#!/bin/bash
#BSUB -J AML
#BSUB -q gpuv100      
#BSUB -gpu "num=1:mode=exclusive_process"        
#BSUB -W 08:00            
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -e AML_%J.err
#BSUB -o AML_%J.out

# 1. Clear everything
module purge

# 2. Load the specific requirements for your venv (Compiler and CUDA)
module load sqlite3/3.45.1 gcc/12.3.0-binutils-2.40
module load cuda/11.8

# 3. Load Python (ensure this matches the version used to create the venv)
module load python3/3.10.13

# 4. Activate the virtual environment
source ../.venv/bin/activate



# -- Exécution --
../.venv/bin/python3 ensemble_vae.py covariance \
    --experiment-folder cav_models \
    --D 1 2 3 \
    --M 10 \
    --num-geodesics 10 \
    --N 10 20 4\
    --num-iterations 300 \
    --cov-methods euclidean piecewise polynomial \
    --cov-output-file cov_plot_job.pdf \
    --cov-csv-file cov_results_job.csv \
    --device cuda