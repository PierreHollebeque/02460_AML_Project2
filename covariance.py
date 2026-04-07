
import torch, numpy as np, matplotlib.pyplot as plt, os
from ensemble_vae import vae_load, subsample
from torchvision import datasets, transforms
import seaborn as sns
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from geodesics import compute_geodesic


def encode_images_to_latent_means(image_pair, model, device):
    """
    Encode a single image pair through a model and extract latent means.
    
    Parameters:
    image_pair: [torch.Tensor]
        Tensor of shape (2, 1, 28, 28) - a pair of MNIST images
    model: [VAE]
        A single VAE model
    device: [torch.device]
        Device to run on
        
    Returns:
    x1, x2: [torch.Tensor, torch.Tensor]
        Latent means for the two images (shape: latent_dim)
    """
    with torch.no_grad():
        model.eval()
        img1 = image_pair[0].unsqueeze(0).to(device)  # (1, 1, 28, 28)
        img2 = image_pair[1].unsqueeze(0).to(device)  # (1, 1, 28, 28)
        
        q1 = model.encoder(img1)
        q2 = model.encoder(img2)
        
        x1 = q1.mean.squeeze(0)  # latent_dim
        x2 = q2.mean.squeeze(0)  # latent_dim
    
    return x1, x2


def select_image_pairs(test_data, num_pairs, random_seed=None):
    """
    Select random pairs of distinct images from test data.
    
    Parameters:
    test_data: [torch.utils.data.Dataset]
        Dataset containing (image, target) tuples
    num_pairs: [int]
        Number of image pairs to select
    random_seed: [int, optional]
        Random seed for reproducibility
        
    Returns:
    image_pairs: [torch.Tensor]
        Tensor of shape (num_pairs, 2, 1, 28, 28) containing image pairs
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # Get all images from test_data (it's a TensorDataset from subsample)
    # test_data.tensors[0] contains the images
    all_images = test_data.tensors[0]  # Shape: (num_train_data, 1, 28, 28)
    
    num_available = all_images.shape[0]
    image_pairs = []
    
    for _ in range(num_pairs):
        # Select two distinct random indices
        indices = torch.randperm(num_available)[:2]
        img1 = all_images[indices[0]]
        img2 = all_images[indices[1]]
        image_pairs.append(torch.stack([img1, img2]))
    
    image_pairs = torch.stack(image_pairs)  # Shape: (num_pairs, 2, 1, 28, 28)
    return image_pairs



def plot_cov(
    all_models, D_values, device,
    num_latent_points=10, number_parameters_geodesic_list=None,
    num_iter=100, lr=1e-3, 
    methods=("euclidean", "piecewise"),
    output_file="cov_plot.pdf",
    csv_file="cov_results.csv"
):
    

    sns.set_style("whitegrid")
    sns.set_context("paper")

    # latent_dim = all_models[0][0].prior.latent_dim
    # z = torch.randn(num_latent_points,2, latent_dim, device=device)

    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    num_train_data = 2048
    num_classes = 3
    test_data = subsample(test_tensors.data, test_tensors.targets, num_train_data, num_classes)
    
    # Select image pairs (fixed across all models for fair comparison)
    image_pairs = select_image_pairs(test_data, num_latent_points, random_seed=42)
    
    # Get latent dimension from first model
    latent_dim = all_models[0][0].prior.latent_dim
    print(f"[plot_cov] Latent dimension: {latent_dim}")
    print(f"[plot_cov] Selected {num_latent_points} image pairs from MNIST test set")
    
    rows = []

    total_jobs = len(methods) * len(D_values)
    job = 0

    print("\n[plot_cov] Starting CoV computation")
    print(f"[plot_cov] methods = {list(methods)}")
    print(f"[plot_cov] decoder counts = {D_values}")
    print(f"[plot_cov] total CoV evaluations = {total_jobs}\n")

    for i, method in enumerate(methods):
        current_N = number_parameters_geodesic_list[i] if number_parameters_geodesic_list else 10

        for j, d in enumerate(D_values):
            job += 1
            print(f"[plot_cov] ({job}/{total_jobs}) method='{method}', decoders={d}")

            # For each model in this decoder count, encode the image pairs and compute geodesics
            stats = cov_matrix_stats(
                image_pairs=image_pairs,
                models=all_models[j],
                num_latent_points=num_latent_points,
                number_parameters_geodesic=current_N, 
                num_iter=num_iter,
                lr=lr,
                curve_method_str=method,
                device=device
            )

            row = {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "decoder_count": d,
                "model_count": len(all_models[j]),
                "latent_dim": latent_dim,
                "num_latent_points": num_latent_points,
                "number_parameters_geodesic": current_N, # Store the specific N used
                "num_iter": num_iter,
                "lr": lr,
                "pair_count": stats["pair_count"],
                "dist_mean": stats["dist_mean"],
                "dist_std": stats["dist_std"],
                "cov_mean": stats["cov_mean"],
                "cov_std": stats["cov_std"],
                "energy_start_mean": stats["energy_start_mean"],
                "energy_start_std": stats["energy_start_std"],
                "energy_end_mean": stats["energy_end_mean"],
                "energy_end_std": stats["energy_end_std"],
                "energy_min_mean": stats["energy_min_mean"],
                "energy_min_std": stats["energy_min_std"],
                "energy_mean_mean": stats["energy_mean_mean"],
                "energy_mean_std": stats["energy_mean_std"],
            }

            rows.append(row)

            print(f"[plot_cov] ({job}/{total_jobs}) finished -> mean CoV = {stats['cov_mean']:.6f}\n")

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"[plot_cov] Saved CSV to: {csv_file}")

    plot_cov_from_csv(
        csv_file=csv_file,
        output_file=output_file
    )

    return df

def plot_cov_from_csv(csv_file, output_file="cov_plot.pdf", y_col="cov_mean", error_col="cov_std"):
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_context("paper")

    df = pd.read_csv(csv_file)

    methods = list(df["method"].unique())
    D_values = sorted(df["decoder_count"].unique())

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    fig, ax = plt.subplots(figsize=(3.4, 3.2))

    x = np.arange(len(D_values))

    # Spread methods side-by-side for the same decoder count
    n_methods = len(methods)
    total_width = 0.8
    bar_width = total_width / max(1, n_methods)

    for i, method in enumerate(methods):
        df_m = df[df["method"] == method].sort_values("decoder_count")
        y = df_m[y_col].to_numpy()
        yerr = df_m[error_col].to_numpy()

        offset = (i - (n_methods - 1) / 2) * bar_width
        x_method = x + offset

        ax.errorbar(
            x_method,
            y,
            yerr=yerr,
            fmt="-o",
            label=method,
            color=colors[i % len(colors)],
            elinewidth=0.8,
            capsize=3,
            linewidth=1.5
        )

    ax.set_xlabel("Number of decoders")
    ax.set_ylabel("Average CoV")
    ax.set_title("Average CoV of pairwise distances", fontsize=10, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(D_values)

    ax.legend(
        frameon=True,
        fontsize=7,
        title="Method",
        title_fontsize=7,
        loc="best"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.show()

    print(f"[plot_cov_from_csv] Saved figure to: {output_file}")



def load_models_for_cov(root_folder, D_values, num_models_per_D, device):
    print(f"\n[INFO] Loading models from root folder: {root_folder}")
    print(f"[INFO] Target decoder counts D: {D_values}")
    print(f"[INFO] Models per D (num_models_per_D): {num_models_per_D}\n")

    all_models = []

    for d in D_values:
        subfolder = os.path.join(root_folder, f"{d}_decoders")
        print(f"[INFO] Checking folder: {subfolder}")

        if not os.path.isdir(subfolder):
            raise FileNotFoundError(f"[ERROR] Missing folder: {subfolder}")

        all_files = sorted(os.listdir(subfolder))
        model_files = [f for f in all_files if f.endswith(".pt")]

        print(f"[INFO] Found {len(model_files)} .pt files")

        model_files = model_files[:num_models_per_D]

        if len(model_files) < num_models_per_D:
            raise ValueError(
                f"[ERROR] Folder {subfolder} only contains {len(model_files)} model files, but num_models_per_D={num_models_per_D}"
            )

        print(f"[INFO] Using files: {model_files}")

        models_d = []
        for i, fname in enumerate(model_files):
            model_path = os.path.join(subfolder, fname)
            print(f"[LOAD] ({i+1}/{num_models_per_D}) Loading: {model_path}")

            model, _ = vae_load(model_path, device)
            model.eval()

            models_d.append(model)

        print(f"[DONE] Loaded {len(models_d)} models for D={d}\n")

        all_models.append(models_d)

    print("[SUCCESS] Finished loading all models.\n")
    return all_models



def compute_cov_matrix(D):
    """
    Calculates CoV matrix
    D shape: (M, N_pairs)
        M: Ensemble Count (number of models)
        N_pairs: Number of latent point pairs
    """
    mean = D.mean(axis=0)
    std  = D.std(axis=0)
    # Handle division by zero for mean, placing 0 where mean is 0
    cov = np.divide(std, mean, out=np.zeros_like(std, dtype=float), where=mean!=0)
    
    return cov

def generate_dist_mat(
    image_pairs, num_latent_points, models,
    curve_method_str="piecewise",
    number_parameters_geodesic=10,
    num_iter=1000,
    lr=1e-3,
    device="cpu"
):
    """
    Generate distance matrix from latent means for each model separately.
    
    For each model:
    - Encode the image pairs to get latent means
    - Compute geodesic distances between latent means
    
    Parameters:
    image_pairs: [torch.Tensor]
        Shape (num_pairs, 2, 1, 28, 28) - pairs of MNIST images (same across all models)
    num_latent_points: [int]
        Number of image pairs
    models: [list of VAE]
        List of VAE models
    ...other params...
    
    Returns:
    dict with distance and energy matrices, shape (num_models, num_pairs)
    """
    num_models_per_D = len(models)
    dist_mat = np.zeros((num_models_per_D, num_latent_points))
    energy_start_mat = np.full((num_models_per_D, num_latent_points), np.nan)
    energy_end_mat = np.full((num_models_per_D, num_latent_points), np.nan)
    energy_min_mat = np.full((num_models_per_D, num_latent_points), np.nan)
    energy_mean_mat = np.full((num_models_per_D, num_latent_points), np.nan)

    for m in tqdm(range(num_models_per_D), desc="Models"):
        model = models[m]
        pair_bar = tqdm(total=num_latent_points, desc=f"Pairs (model {m+1}/{num_models_per_D})", leave=False)

        for k in range(num_latent_points):
            # Get the image pair (same for all models)
            image_pair = image_pairs[k]  # Shape: (2, 1, 28, 28)
            
            # Encode through this model and extract latent means
            x1, x2 = encode_images_to_latent_means(image_pair, model, device)
            
            # Compute geodesic distance between latent means for this model
            meta = compute_geodesic(
                x1, x2, model,
                curve_method_str=curve_method_str,
                number_parameters_geodesic=number_parameters_geodesic,
                num_iter=num_iter,
                lr=lr,
                device=device,
                return_metadata=True
            )

            dist_mat[m, k] = meta["distance"]
            energy_start_mat[m, k] = meta["energy_start"]
            energy_end_mat[m, k] = meta["energy_end"]
            energy_min_mat[m, k] = meta["energy_min"]
            energy_mean_mat[m, k] = meta["energy_mean"]

            pair_bar.update(1)

        pair_bar.close()

    return {
        "dist_mat": dist_mat,
        "energy_start_mat": energy_start_mat,
        "energy_end_mat": energy_end_mat,
        "energy_min_mat": energy_min_mat,
        "energy_mean_mat": energy_mean_mat
    }

def cov_matrix_stats(
    image_pairs, models,
    num_latent_points=10,
    number_parameters_geodesic=100,
    num_iter=1000,
    lr=1e-3,
    curve_method_str="piecewise",
    device="cpu"
):
    results = generate_dist_mat(
        image_pairs=image_pairs,
        num_latent_points=num_latent_points,
        number_parameters_geodesic=number_parameters_geodesic,
        models=models,
        curve_method_str=curve_method_str,
        num_iter=num_iter,
        lr=lr,
        device=device
    )

    dist_mat = results["dist_mat"]
    cov = compute_cov_matrix(dist_mat)

    # With the new structure, dist_mat and cov are already the values we need
    dist_vals = dist_mat
    cov_vals = cov
    energy_start_vals = results["energy_start_mat"]
    energy_end_vals = results["energy_end_mat"]
    energy_min_vals = results["energy_min_mat"]
    energy_mean_vals = results["energy_mean_mat"]

    summary = {
        "dist_mean": float(np.nanmean(dist_vals)),
        "dist_std": float(np.nanstd(dist_vals)),
        "cov_mean": float(np.nanmean(cov_vals)),
        "cov_std": float(np.nanstd(cov_vals)),
        "energy_start_mean": float(np.nanmean(energy_start_vals)),
        "energy_start_std": float(np.nanstd(energy_start_vals)),
        "energy_end_mean": float(np.nanmean(energy_end_vals)),
        "energy_end_std": float(np.nanstd(energy_end_vals)),
        "energy_min_mean": float(np.nanmean(energy_min_vals)),
        "energy_min_std": float(np.nanstd(energy_min_vals)),
        "energy_mean_mean": float(np.nanmean(energy_mean_vals)),
        "energy_mean_std": float(np.nanstd(energy_mean_vals)),
        "pair_count": int(num_latent_points), # Number of pairs computed
        "M": len(models),
        "num_latent_points": num_latent_points,
        "dist_mat": dist_mat,
        "cov_mat": cov
    }

    return summary
