import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim # Import for optimizer, e.g., Adam
import argparse # Added for command-line arguments
import torch.nn as nn
from tqdm import tqdm # Import tqdm for progress bar
# Import VAE components from ensemble_vae.py
from ensemble_vae import VAE, GaussianPrior, GaussianEncoder, GaussianDecoder, new_decoder, new_encoder


class EnergyMinimizer:
    def __init__(self, decoder, curve_method_instance, optimizer_class=optim.Adam, lr=0.01):
        """Initializes the energy minimizer."""
        self.decoder = decoder
        self.device = curve_method_instance.device
        self.curve_method = curve_method_instance
        self.optimizer = optimizer_class([self.curve_method.parameters], lr=lr)
    
    def minimize_energy(self, num_iterations=100):
        """Minimizes the curve's energy via gradient descent."""
        # print(f"Starting energy minimization for {type(self.curve_method).__name__} curve...")
        for i in range(num_iterations):
            self.optimizer.zero_grad()
            
            # Calculate energy using the curve's specific implementation
            energy = self.curve_method.calculate_energy(self.decoder) 
            
            energy.backward()
            self.optimizer.step()
            
            # if (i + 1) % (num_iterations // 10 if num_iterations >= 10 else 1) == 0 or i == 0: # Print progress
            #     print(f"Iteration {i+1}/{num_iterations}, Energy: {energy.item():.4f}")
        
        # print("Minimization complete.")
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

    def calculate_energy(self, decoders, montecarlo_sample=10000):
        """Approximates the curve energy as sum of squared distances between reconstructed images."""
        curve_points = self.get_full_curve_points()
        
        N = curve_points.shape[0]
        
        if N < 2:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        reconstructions = torch.stack([decoders(curve_points,i) for i in range(decoders.num_decoders)])
        
        total_energy = 0

        for i_idx in range(N - 1):
            
            l_idx = np.random.randint(0, len(decoders), size=montecarlo_sample)
            k_idx = np.random.randint(0, len(decoders), size=montecarlo_sample)

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
        if self.N > 0:
            initial_points = torch.randn(self.N, self.dim, device=self.device, dtype=torch.float32)
        else:
            initial_points = torch.empty(0, self.dim, device=self.device, dtype=torch.float32)

        self.parameters = torch.nn.Parameter(initial_points)
    
    def get_full_curve_points(self):
        """Assembles the full curve from start, intermediate, and end points."""
        if self.N > 0:
            return torch.cat((self.x1.unsqueeze(0), self.parameters, self.x2.unsqueeze(0)), dim=0)
        else:
            return torch.cat((self.x1.unsqueeze(0), self.x2.unsqueeze(0)), dim=0)


class Polynomial_3(CurveMethod):
    def __init__(self, x1, x2, N=2, device='cpu', dim=2): # N=2 for the two free vector coefficients
        super().__init__(x1, x2, N, device, dim)
        
        if self.N != 2:
            raise ValueError(f"Polynomial_3 (cubic polynomial) requires N=2 for coefficients, but got N={self.N}.")

        # Initialize the two free polynomial coefficients (w3, w2) as optimizable parameters.
        initial_coeffs = torch.randn(self.N, self.dim, device=self.device, dtype=torch.float32)
        self.parameters = torch.nn.Parameter(initial_coeffs)

        self.num_eval_points = 100 

    
    def get_full_curve_points(self):
        """
        Evaluates the cubic polynomial c(t) = w3*t³ + w2*t² + w1*t + w0.
        Coefficients w3 and w2 are the optimizable parameters.
        Coefficients w1 and w0 are derived from endpoint constraints c(0)=x1, c(1)=x2.
        """
        w3, w2 = self.parameters[0], self.parameters[1]

        # Derive fixed coefficients from endpoint constraints.
        w0 = self.x1
        w1 = self.x2 - self.x1 - w3 - w2

        t = torch.linspace(0, 1, self.num_eval_points, device=self.device, dtype=torch.float32)

        # Evaluate polynomial c(t) for all t values.
        t3 = t.pow(3).unsqueeze(-1)
        t2 = t.pow(2).unsqueeze(-1)
        t1 = t.unsqueeze(-1)
        curve_points = t3 * w3 + t2 * w2 + t1 * w1 + w0
        return curve_points


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

    # Determine curve_method class and N value
    if curve_method_str == 'piecewise':
        curve_class = Piecewise
        N_val = N
    elif curve_method_str == 'polynomial':
        curve_class = Polynomial_3
        if N != 2:
            print(f"Warning: Polynomial_3 requires N=2, but got N={N}. Setting N=2.")
            N_val = 2
        else:
            N_val = N
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
    latent_dim = args.latent_dim
    # Instantiate VAE components using the copied new_encoder_net and new_decoder_net
    prior = GaussianPrior(latent_dim)
    decoder_module = GaussianDecoder(new_decoder(latent_dim))
    encoder_module = GaussianEncoder(new_encoder(latent_dim))
    model = VAE(prior, decoder_module, encoder_module).to(device)
    model.load_state_dict(torch.load(args.vae_model_path, map_location=device))
    
    calculate_and_plot_geodesics(
        model=model,
        device=device,
        latent_dim=args.latent_dim,
        curve_method_str=args.curve_method,
        num_iterations=args.num_iterations,
        lr=args.lr,
        N=args.N,
        num_geodesics_to_plot=args.num_geodesics_to_plot,
        output_filename=args.output_filename,
        seed=args.seed
    )