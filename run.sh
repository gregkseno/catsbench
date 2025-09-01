#!/bin/bash
#SBATCH --job-name=bench_light_sb
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-26%4
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=16-00:00:00

source activate dot_bench
seed=1

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01

elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 &
    python -m src.run experiment=dlight_sb_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 11 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 12 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 13 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 14 ]; then
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 15 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 16 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 17 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 18 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 19 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 20 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 21 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 22 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 23 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 24 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 25 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$SLURM_ARRAY_TASK_ID" -eq 26 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$seed" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

fi

wait
