#!/bin/bash
#SBATCH --job-name=benchmark_images
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source activate dot_bench
SEED=1

HYDRA_FULL_ERROR=1 python -m src.run -m \
  seed=${SEED} data.num_workers=1 data.pin_memory=true \
  experiment=dlight_sb/benchmark_images/images_g002
    

python -m src.run seed=1 experiment=csbm/gaussian_to_swiss_roll/default method.mse_loss_coeff=1.0 method.kl_loss_coeff=0.0 method.ce_loss_coeff=0