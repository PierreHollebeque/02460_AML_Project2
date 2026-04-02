#!/bin/bash
#BSUB -J precipitation
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -e sleeper_%J.err
#BSUB -o sleeper_%J.out

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613
