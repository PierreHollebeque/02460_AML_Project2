# 02460 Advanced Machine Learning Project

This project explores Variational Autoencoders (VAEs) and the geometry of their latent space. The main script, `ensemble_vae.py`, can be used to train models, generate samples, and visualize geodesics.

## Usage

The main script is `ensemble_vae.py`. It accepts several modes and arguments to control its behavior.

### 1. Train a VAE Model

Before calculating geodesics or generating samples, you must train a model.

```bash
# Train a model and save the results in the 'experiment' folder
python ensemble_vae.py train --experiment-folder experiment --epochs-per-decoder 50
```

This command will train a VAE for 50 epochs and save the trained model as `experiment/model.pt`.

### 2. Calculate and Plot Geodesics

The `geodesics` mode allows you to visualize the structure of the latent space by plotting the shortest paths (geodesics) between pairs of random points. The "distance" is measured within the space of reconstructed images.

#### Prerequisites

Ensure you have a trained model (see Step 1).

#### Basic Usage

To run the calculation with default parameters (10 curves, `piecewise` method):

```bash
python ensemble_vae.py geodesics --experiment-folder experiment
```

The plot will be displayed and saved as `geodesics.png`.

#### Important Arguments for Geodesics

* `--experiment-folder <path>`: **(Required)** Path to the folder containing the trained `model.pt`.
* `--num-curves <int>`: Number of geodesics to calculate and plot. (Default: 10)
* `--curve-method <method>`: Curve representation method. Choices: `piecewise` or `polynomial`. (Default: `piecewise`)
* `--N <int>`: Number of intermediate points for `piecewise` or coefficients for `polynomial` (must be 2 for `polynomial`). (Default: 30)
* `--num-iterations <int>`: Number of iterations for the optimization of each geodesic. (Default: 300)
* `--output-file <filename>`: Filename to save the plot. (Default: `geodesics.png`)

#### Advanced Examples

1.  **Using the polynomial method:**
    ```bash
    python ensemble_vae.py geodesics --experiment-folder experiment --curve-method polynomial --N 2
    ```

2.  **Plotting 25 curves with more iterations and saving to a custom file:**
    ```bash
    python ensemble_vae.py geodesics --experiment-folder experiment --num-curves 25 --num-iterations 500 --output-file detailed_geodesics.png
    ```

