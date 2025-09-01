#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2667
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

source activate dot_bench
HYDRA_FULL_ERROR=1 python -m src.run experiment=dlight_sb_benchmark_images
# HYDRA_FULL_ERROR=1 python -m src.run experiment=csbm_benchmark data.dim=2 method.ce_loss_coeff=0.0 trainer.limit_train_batches=100000 method.num_first_iterations=1 prior.num_timesteps=127 prior.num_skip_steps=1
# HYDRA_FULL_ERROR=1 python -m src.run experiment=dlight_sb_m_gaussian_to_swiss_roll prior.num_timesteps=127 prior.num_skip_steps=1
