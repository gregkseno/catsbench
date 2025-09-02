#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

source activate dot_bench
SEED=1

python -m src.run -m \
  seed="$SEED" \
  hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
  experiment=dlight_sb/benchmark/d64_g002,dlight_sb/benchmark/d64_g005,dlight_sb/benchmark/d64_u0005,dlight_sb/benchmark/d64_u001