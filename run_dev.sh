#!/bin/bash
#SBATCH --job-name=train-toy
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source activate dot_bench
# python -m src.run experiment=light_sb_d_gaussian_to_swiss_roll
python -m src.run experiment=csbm_gaussian_to_swiss_roll
