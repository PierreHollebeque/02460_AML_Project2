
import torch, numpy as np, matplotlib.pyplot as plt, os
from ensemble_vae import vae_load
import seaborn as sns
from geodesics import cov_matrix_stats

import pandas as pd
from datetime import datetime

def plot_cov(
    all_models, D_values, device,
    N=10, num_curve=100,
    num_iter=100, lr=1e-3,
    methods=("euclidean", "piecewise"),
    output_file="cov_plot.pdf",
    csv_file="cov_results.csv"
):
    

    sns.set_style("whitegrid")
    sns.set_context("paper")

    latent_dim = all_models[0][0].prior.latent_dim
    z = torch.randn(N, latent_dim, device=device)

    rows = []

    total_jobs = len(methods) * len(D_values)
    job = 0

    print("\n[plot_cov] Starting CoV computation")
    print(f"[plot_cov] methods = {list(methods)}")
    print(f"[plot_cov] decoder counts = {D_values}")
    print(f"[plot_cov] total CoV evaluations = {total_jobs}\n")

    for i, method in enumerate(methods):
        for j, d in enumerate(D_values):
            job += 1
            print(f"[plot_cov] ({job}/{total_jobs}) method='{method}', decoders={d}")

            stats = cov_matrix_stats(
                z=z,
                models=all_models[j],
                N=N,
                num_curve=num_curve,
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
                "N_points": N,
                "num_curve": num_curve,
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

    for i, method in enumerate(methods):
        df_m = df[df["method"] == method].sort_values("decoder_count")
        y = df_m[y_col].to_numpy()
        yerr = df_m[error_col].to_numpy()

        ax.errorbar(
            x,
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
                f"[ERROR] Folder {subfolder} only contains {len(model_files)} model files, but M={M}"
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