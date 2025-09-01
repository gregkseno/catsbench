#!/bin/bash
set -euo pipefail

SEED=1
DATA="/filestore/benchmark/data"
LOGS="/filestore/benchmark/logs"

if [ "$1" -eq 0 ]; then
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01

elif [ "$1" -eq 1 ]; then
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01

elif [ "$1" -eq 2 ]; then
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 &
    python -m src.run experiment=dlight_sb_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01

elif [ "$1" -eq 3 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 4 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 5 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 6 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 7 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 8 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 9 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 10 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 11 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 12 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 13 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 14 ]; then
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=csbm_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 15 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 16 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 17 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 18 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=2 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 19 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 20 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 21 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 22 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=16 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 23 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.02 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 24 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=gaussian prior.alpha=0.05 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 25 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.005 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

elif [ "$1" -eq 26 ]; then
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=63 prior.num_skip_steps=2 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=1.0 method.mse_loss_coeff=0.0 &
    python -m src.run experiment=dlight_sb_m_benchmark seed="$SEED" paths.data_dir="$DATA" paths.log_dir="$LOGS" data.dim=64 prior.prior_type=uniform prior.alpha=0.01 prior.num_timesteps=15 prior.num_skip_steps=8 method.kl_loss_coeff=0.0 method.mse_loss_coeff=1.0

fi

wait
