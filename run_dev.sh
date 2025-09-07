#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=25GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source activate dot_bench
SEED=1

# HYDRA_FULL_ERROR=1 python -m src.run -m \
#   hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
#   hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
#   hydra.launcher.submitit_folder='${paths.log_dir}/sweeps/${now:%Y-%m-%d_%H-%M-%S}/.submitit' \
#   seed=${SEED} data.num_workers=1 data.pin_memory=true \
#   experiment=dlight_sb_m/benchmark/d64_u001_t63_kl,dlight_sb_m/benchmark/d64_u001_t63_mse

python -m src.run seed=1 experiment=csbm/gaussian_to_swiss_roll/default method.mse_loss_coeff=1.0 method.kl_loss_coeff=0.0 method.ce_loss_coeff=0