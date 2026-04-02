#!/bin/bash
#BSUB -J AML
#BSUB -q gpuv100          # Utilise la file GPU V100
#BSUB -gpu "num=1"        # Sollicite 1 GPU
#BSUB -W 08:00            # Temps maximum : 8 heures
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -e logs/precip_%J.err
#BSUB -o logs/precip_%J.out

# -- Préparation --
# Créer le dossier logs s'il n'existe pas pour éviter que le job ne fail
mkdir -p logs

# Nettoyer les modules pour éviter les conflits
module purge

# Charger Python et CUDA (versions stables et compatibles sur DCC)
module load python3/3.10.13
module load cuda/11.8

# -- Activation de l'environnement --
# Correction de la syntaxe : on pointe vers le .venv qui est deux niveaux au-dessus
source ../.venv/bin/activate

# -- Exécution --
python3 ensemble_vae.py covariance \
    --experiment-folder cav_models \
    --D 1 2 3 \
    --M 10 \
    --num-geodesics 10 \
    --N 10 \
    --num-iterations 300 \
    --cov-methods euclidean piecewise polynomial \
    --cov-output-file cov_plot.pdf \
    --device cuda