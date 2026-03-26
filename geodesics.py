import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim # Import for optimizer, e.g., Adam
import argparse # Added for command-line arguments
import torch.nn as nn
from tqdm import tqdm # Import tqdm for progress bar
# Import VAE components from ensemble_vae.py
from ensemble_vae import VAE, GaussianPrior, GaussianEncoder, GaussianDecoder, new_decoder, new_encoder, vae_load
import plotly.graph_objects as go


class EnergyMinimizer:
    def __init__(self, decoder, curve_method_instance, optimizer_class=optim.Adam, lr=0.01):
        """Initializes the energy minimizer."""
        self.decoder = decoder
        self.device = curve_method_instance.device
        self.curve_method = curve_method_instance
        self.optimizer = optimizer_class([self.curve_method.parameters], lr=lr)
    
    def minimize_energy(self, num_iterations=100):
        """Minimizes the curve's energy via gradient descent."""
        for i in tqdm(range(num_iterations), desc=f"Minimizing Energy for {type(self.curve_method).__name__}", leave=False):

            self.optimizer.zero_grad()
            
            # Calculate energy using the curve's specific implementation
            energy = self.curve_method.calculate_energy(self.decoder) 
            
            energy.backward()
            self.optimizer.step()
            
        return self.curve_method.get_full_curve_points()


class CurveMethod:
    def __init__(self, x1, x2, N=10, device='cpu', dim=2):
        self.N = N
        self.device = device
        self.x1 = torch.as_tensor(x1, dtype=torch.float32, device=device)
        self.x2 = torch.as_tensor(x2, dtype=torch.float32, device=device)
        self.dim = dim
    

    def get_full_curve_points(self):
        """Returns all points defining the curve. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_full_curve_points method.")

    def calculate_energy(self, decoders, montecarlo_sample=500):
        """Approximates the curve energy as sum of squared distances between reconstructed images."""
        curve_points = self.get_full_curve_points()
        N = curve_points.shape[0]
        
        if N < 2:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        reconstructions = torch.stack([decoders(curve_points,i).mean for i in range(decoders.num_decoders)])

        if decoders.num_decoders == 1: # One decoder : fallback to previous implementation
            decoder = decoders
            curve_points = self.get_full_curve_points()
            reconstructed_images = decoder(curve_points).mean
            # Calculate squared Euclidean distance between consecutive reconstructed images
            # Assuming images are (batch_size, C, H, W)
            segment_image_diffs = reconstructed_images[1:] - reconstructed_images[:-1]
            squared_image_distances = torch.sum(segment_image_diffs**2, dim=list(range(1, segment_image_diffs.ndim)))
            energy = torch.sum(squared_image_distances)
            return energy
        

        total_energy = 0

        for i_idx in range(N - 1):
            
            l_idx = np.random.randint(0, decoders.num_decoders, size=(montecarlo_sample,))
            k_idx = np.random.randint(0, decoders.num_decoders, size=(montecarlo_sample,))

            # f_l(c(t_i))
            img_i = reconstructions[l_idx, i_idx]
            # f_k(c(t_{i+1}))
            img_next = reconstructions[k_idx, i_idx + 1]

            diffs = (img_i - img_next).view(montecarlo_sample, -1)
            squared_distances = torch.sum(diffs**2, dim=1)

            total_energy += torch.mean(squared_distances)

        return total_energy

class Piecewise(CurveMethod):
    def __init__(self, x1, x2, N=10, device='cpu', dim=2):
        super().__init__(x1, x2, N, device, dim)
        # Initialize N intermediate points with random coordinates as optimizable parameters.
        self.N = N
        linspace_points = torch.linspace(0, 1, N + 2, device=device).unsqueeze(1)
        interpolated = (1 - linspace_points) * self.x1 + linspace_points * self.x2
        self.parameters = torch.nn.Parameter(interpolated[1:-1])
    
    def get_full_curve_points(self):
        """Assembles the full curve from start, intermediate, and end points."""
        if self.N > 0:
            return torch.cat((self.x1.unsqueeze(0), self.parameters, self.x2.unsqueeze(0)), dim=0)
        else:
            return torch.cat((self.x1.unsqueeze(0), self.x2.unsqueeze(0)), dim=0)


class PolynomialCurve(CurveMethod):
    def __init__(self, x1, x2, N=3, device='cpu', dim=2):
        """
        N: Degree of the model = Number of free parameters - 1.
        dim: Dimensionality of the space (default 2).
        """
        super().__init__(x1, x2, N-1, device, dim)
        # N-1 because of the multiplication by the bridge
        
        # self.N free coefficients (w_k) for the bridge polynomial P(t)
        # Shape: (N, dim)
        self.parameters = torch.nn.Parameter(
            torch.randn(self.N, self.dim, device=self.device, dtype=torch.float32)
        )
        self.num_eval_points = 100

    def get_full_curve_points(self):
        """
        Evaluates: c(t) = (1-t)x1 + t*x2 + t(1-t) * sum_{k=0}^{N-1} (w_k * t^k)
        """
        # 1. Setup t: (M, 1)
        t = torch.linspace(0, 1, self.num_eval_points, device=self.device).unsqueeze(-1)
        
        # 2. Build power basis: [t^0, t^1, ..., t^{N-1}]
        # powers shape: (N,) -> t_pow shape: (M, N)
        powers = torch.arange(self.N, device=self.device, dtype=torch.float32)
        t_pow = t ** powers 
        
        # 3. Compute P(t) via matrix multiplication
        # (M, N) @ (N, dim) -> (M, dim)
        poly_P_t = torch.matmul(t_pow, self.parameters)
        
        # 4. Apply bridge constraint
        bridge = t * (1 - t)
        
        # 5. Final interpolation
        # (1-t)*x1 + t*x2 + bridge*P(t)
        curve_points = (1 - t) * self.x1 + t * self.x2 + bridge * poly_P_t
        return curve_points



def calculate_and_plot_geodesics_3d(model, device, latent_dim, curve_method_str, num_iterations, lr, N, num_geodesics_to_plot, seed=None):
    """
    Plots geodesics in 3D where the Z-axis represents the L2 norm of the VAE reconstruction.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model.eval()
    decoder = model.decoder

    # 1. Setup Curve Classes
    if curve_method_str == 'piecewise':
        curve_class = Piecewise
    elif curve_method_str == 'polynomial':
        curve_class = PolynomialCurve
    else:
        raise ValueError(f"Unknown curve method: {curve_method_str}")

    # 2. Generate Background Surface (L2 Norm)
    grid_range = np.linspace(-4, 4, 100)
    xx, yy = np.meshgrid(grid_range, grid_range)
    grid_points = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=-1), 
                               dtype=torch.float32, device=device)
    
    with torch.no_grad():
        reconstructed_dist = decoder(grid_points)
        recon_mean = reconstructed_dist.mean
        # Calculate L2 Norm: sqrt(sum(x^2))
        zz = torch.sqrt(torch.sum(recon_mean**2, dim=list(range(1, recon_mean.ndim))))
        zz = zz.cpu().numpy().reshape(xx.shape)

    # Create the 3D Surface
    fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale='Viridis', opacity=0.8, name='L2 Norm Surface')])

    # 3. Calculate and Plot Geodesics
    all_latent_points = torch.randn(2 * num_geodesics_to_plot, latent_dim, device=device)
    
    for i in tqdm(range(num_geodesics_to_plot), desc="Calculating 3D Geodesics"):
        x1, x2 = all_latent_points[2 * i], all_latent_points[2 * i + 1]
        
        curve_method = curve_class(x1, x2, N=N, device=device, dim=latent_dim)
        minimizer = EnergyMinimizer(decoder=decoder, curve_method_instance=curve_method, 
                                    optimizer_class=torch.optim.Adam, lr=lr)
        
        # Optimize
        optimized_points = minimizer.minimize_energy(num_iterations=num_iterations).detach()
        
        # Calculate Z-values for the geodesic points specifically
        with torch.no_grad():
            pts_recon = decoder(optimized_points).mean
            pts_z = torch.sqrt(torch.sum(pts_recon**2, dim=list(range(1, pts_recon.ndim)))).cpu().numpy()
        
        opt_np = optimized_points.cpu().numpy()

        # Add Geodesic Line (elevated slightly by +0.05 to prevent clipping through surface)
        fig.add_trace(go.Scatter3d(
            x=opt_np[:, 0], y=opt_np[:, 1], z=pts_z + 0.05,
            mode='lines',
            line=dict(color='white', width=4),
            name=f'Geodesic {i+1}'
        ))

        # Add Start/End Markers
        fig.add_trace(go.Scatter3d(
            x=[opt_np[0, 0], opt_np[-1, 0]], 
            y=[opt_np[0, 1], opt_np[-1, 1]], 
            z=[pts_z[0] + 0.1, pts_z[-1] + 0.1],
            mode='markers',
            marker=dict(color='black', size=4),
            showlegend=False
        ))

    # 4. Layout Improvements
    fig.update_layout(
        title=f"3D Geodesic Paths on L2 Norm Surface ({curve_method_str})",
        scene=dict(
            xaxis_title='Latent Dim 1',
            yaxis_title='Latent Dim 2',
            zaxis_title='L2 Norm (Energy)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5) # Flatten Z slightly for better visibility
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()


def calculate_and_plot_geodesics(model, device, latent_dim, curve_method_str, num_iterations, lr, N, num_geodesics_to_plot, output_filename=None, seed=None):
    """
    Calculates and plots geodesics on the latent space of a VAE.

    Args:
        model (VAE): The trained VAE model.
        device (str): The device to run computations on ('cpu' or 'cuda').
        latent_dim (int): The dimension of the latent space.
        curve_method_str (str): The curve representation method ('piecewise' or 'polynomial').
        num_iterations (int): Number of optimization iterations.
        lr (float): Learning rate for the optimizer.
        N (int): Number of intermediate points or coefficients for the curve method.
        num_geodesics_to_plot (int): Number of geodesics to calculate and plot.
        output_filename (str, optional): If provided, saves the plot to this file.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using random seed: {seed}")

    model.eval() # Set model to evaluation mode

    decoder = model.decoder # This is the GaussianDecoder instance
    cbar_label = 'L2 Norm of Reconstructed Image' # Update label for background plot
    N_val = N
    # Determine curve_method class and N value
    if curve_method_str == 'piecewise':
        curve_class = Piecewise
    elif curve_method_str == 'polynomial':
        curve_class = PolynomialCurve
    else:
        raise ValueError(f"Unknown curve method: {curve_method_str}")

    # Generate random latent points for all start and end points of the geodesics
    # We need 2 * num_geodesics_to_plot individual points
    all_latent_points = torch.randn(2 * num_geodesics_to_plot, latent_dim, device=device)

    # 4. Plot Results
    plt.figure(figsize=(10, 8))
    
    # Background visualization of the metric's "cost"
    grid_range = np.linspace(-4, 4, 100)
    # Adjust grid range based on sampled points if they fall outside -4,4
    # For now, keep it fixed for simplicity, assuming latent points are often around 0.
    xx, yy = np.meshgrid(grid_range, grid_range) 
    grid_points = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=-1), dtype=torch.float32, device=device)
    
    with torch.no_grad(): # No gradients needed for plotting background
        # Get reconstructed images and calculate their L2 norm for the background plot
        reconstructed_images_dist = decoder(grid_points)
        reconstructed_images_mean = reconstructed_images_dist.mean
        # Calculate L2 norm over the image dimensions (C, H, W)
        zz = torch.sqrt(torch.sum(reconstructed_images_mean**2, dim=list(range(1, reconstructed_images_mean.ndim)))).cpu().numpy().reshape(xx.shape)


    plt.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.5) # Increased levels for smoother background
    plt.colorbar().set_label(cbar_label)

    # Plot all sampled latent points
    plt.plot(all_latent_points[:, 0].cpu(), all_latent_points[:, 1].cpu(), 'ko', markersize=8, label='Sampled Latent Points')

    # Iterate through the specified number of pairs and compute geodesics
    first_geodesic_plotted = False # Flag to ensure 'Optimized Geodesics' label appears only once in legend
    # Loop num_geodesics_to_plot times, taking two points from all_latent_points for each geodesic
    for i in tqdm(range(num_geodesics_to_plot), desc="Calculating Geodesics"):
        x1 = all_latent_points[2 * i]
        x2 = all_latent_points[2 * i + 1]

        # 2. Initialize Curve for this pair
        curve_method = curve_class(x1, x2, N=N_val, device=device, dim=latent_dim)
        
        # 3. Run Optimization for this pair
        minimizer = EnergyMinimizer(
            decoder=decoder,
            curve_method_instance=curve_method,
            optimizer_class=optim.Adam,
            lr=lr
        )
        optimized_curve_points = minimizer.minimize_energy(num_iterations=num_iterations).detach().cpu().numpy()

        # Plot optimized curve for this pair
        plt.plot(optimized_curve_points[:, 0], optimized_curve_points[:, 1], 'w-', linewidth=1.5, alpha=0.7, label='Optimized Geodesics' if not first_geodesic_plotted else "")
        first_geodesic_plotted = True # Only add label once for the legend

    plt.title(f'{num_geodesics_to_plot} Geodesics between Random Pairs with {curve_class.__name__} and VAE Decoder')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')

    if output_filename:
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")

    plt.show()

def compute_cov_matrix(D):
    """
    Calculates CoV matrix

    # D shape: (M, N, N) 
        M: Ensemble Count
        N: Number of points
    """
    mean = D.mean(axis=0)
    std  = D.std(axis=0)
    cov = np.divide(std, mean, out=np.zeros_like(std), where=mean!=0)
    cov = std/mean

    return cov

def generate_dist_mat(z,M,N,models,curve_method_str="piecewise",num_curve=100,num_iter=1000,lr=1e-3,full_matrix=0):
    """
    Generates distance matrix for an array of points x
    z: (N,latent_dim) array of points
    M: Ensemble Count
    N: Number of points
    models: M models with same amount of decoders
    """

    dist_mat = np.zeros((M,N,N))

    #compute geodesic distance for each unique distance for each model
    for m in range(M):
        for i in range(N):
            for j in range(i + 1, N):
                dist_mat[m,i,j]=compute_geodesic(z[i],z[j],models[m],curve_method_str,num_curve,num_iter,lr)
        if full_matrix: #can skip since we don't care about permutations
            dist_mat[m,:,:] = dist_mat[m,:,:] + dist_mat[m,:,:].T #symmetric matrix

    return dist_mat

def compute_avg(z,models,N=10,num_curve=100,num_iter=1000,lr=1e-3,curve_method_str="piecewise"):
    M = len(models)

    #Compute distance matrix
    dist_mat= generate_dist_mat(z,M,N,models,curve_method_str,num_curve=100,num_iter=100,lr=1e-3)
    
    #Compute CoV matrix
    cov = compute_cov_matrix(dist_mat)

    mask = np.triu(np.ones((N, N), dtype=bool), k=1) #only different points
    cov_avg= cov[mask].mean()

    return cov_avg

def compute_geodesic(z1,z2,model,curve_method_str="piecewise",num_curve=100,num_iter=100,lr=1e-3):
    """
    Temporary computation for geodesic between two point vectors.
    """
    if curve_method_str == 'piecewise':
        curve_class = Piecewise
        N_val = num_curve
    elif curve_method_str == 'polynomial':
        curve_class = Polynomial_3
        if num_curve != 2:
            print(f"Warning: Polynomial_3 requires N=2, but got N={num_curve}. Setting N=2.")
            N_val = 2
        else:
            N_val = num_curve
    elif curve_method_str == 'euclidian':
        pass
    else:
        raise ValueError(f"Unknown curve method: {curve_method_str}")

    if curve_method_str != 'euclidian':
        #Compute the geodesic distance using the decoder
        curve_method = curve_class(z1, z2, N=N_val, device=device, dim=model.decoder[0].in_features)
        minimizer = EnergyMinimizer(
                decoder=model.decoder,
                curve_method_instance=curve_method,
                optimizer_class=optim.Adam,
                lr=lr
            )
        curve_points = minimizer.minimize_energy(num_iterations=num_iter).detach().cpu().numpy()
        return np.linalg.norm(np.diff(curve_points, axis=0), axis=1).sum()
    else: return np.linalg.norm(np.diff([z1,z2], axis=0), axis=1) #compute eucledian distance

if __name__ == "__main__":
    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # 1.1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optimize a curve's energy using different metrics and curve representations.")
    parser.add_argument('--vae_model_path', type=str, default='experiment/model.pt',
                        help='Path to the trained VAE model state_dict.')
    parser.add_argument('--latent-dim', type=int, default=2, help='Dimension of the VAE latent space (latent_dim).')
    parser.add_argument('--curve_method', type=str, default='piecewise', choices=['piecewise', 'polynomial'],
                        help='Choose the curve representation: "piecewise" (Piecewise) or "polynomial" (Polynomial_3).')
    parser.add_argument('--num_iterations', type=int, default=300,
                        help='Number of optimization iterations.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--N', type=int, default=30,
                        help='Number of intermediate points for Piecewise or coefficients for Polynomial_3 (N=2 for cubic).')
    parser.add_argument('--num_geodesics_to_plot', type=int, default=25,
                        help='Number of geodesics (pairs of points) to calculate and plot.')
    parser.add_argument('--output_filename', type=str, default='geodesics_standalone.png',
                        help='Filename to save the plot.')

    args = parser.parse_args()

    # 1.2. Load VAE model
    model, parameters = vae_load(args.vae_model_path, device)
    latent_dim = parameters['latent_dim']
    
    calculate_and_plot_geodesics(
        model=model,
        device=device,
        latent_dim=latent_dim,
        curve_method_str=args.curve_method,
        num_iterations=args.num_iterations,
        lr=args.lr,
        N=args.N,
        num_geodesics_to_plot=args.num_geodesics_to_plot,
        output_filename=args.output_filename,
        seed=args.seed
    )