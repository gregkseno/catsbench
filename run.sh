#!/bin/bash
#SBATCH --job-name=train-toy
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source activate dot_bench
python -m src.run experiment=csbm_ar_gaussian_to_swiss_roll