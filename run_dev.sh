#!/bin/bash
#SBATCH --job-name=train-toy
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source activate dot_bench
# HYDRA_FULL_ERROR=1 python -m src.run experiment=dlight_sb_gaussian_to_swiss_roll
# HYDRA_FULL_ERROR=1 python -m src.run experiment=csbm_benchmark
HYDRA_FULL_ERROR=1 python -m src.run experiment=dlight_sb_m_gaussian_to_swiss_roll
