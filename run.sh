#!/bin/bash
#SBATCH --job-name=bench_dlight
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-14%4
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=16-00:00:00

source activate dot_bench
export PYTHONPATH="$PWD:$PYTHONPATH"
SEED=1

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb/benchmark/d2_g002,dlight_sb/benchmark/d2_g005,dlight_sb/benchmark/d2_u0005,dlight_sb/benchmark/d2_u001
    ;;
  1)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb/benchmark/d16_g002,dlight_sb/benchmark/d16_g005,dlight_sb/benchmark/d16_u0005,dlight_sb/benchmark/d16_u001
    ;;
  2)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb/benchmark/d64_g002,dlight_sb/benchmark/d64_g005,dlight_sb/benchmark/d64_u0005,dlight_sb/benchmark/d64_u001
    ;;
  3)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d2_g002_t63_kl,dlight_sb_m/benchmark/d2_g002_t63_mse,dlight_sb_m/benchmark/d2_g002_t15_kl,dlight_sb_m/benchmark/d2_g002_t15_mse
    ;;
  4)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d2_g005_t63_kl,dlight_sb_m/benchmark/d2_g005_t63_mse,dlight_sb_m/benchmark/d2_g005_t15_kl,dlight_sb_m/benchmark/d2_g005_t15_mse
    ;;
  5)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d2_u0005_t63_kl,dlight_sb_m/benchmark/d2_u0005_t63_mse,dlight_sb_m/benchmark/d2_u0005_t15_kl,dlight_sb_m/benchmark/d2_u0005_t15_mse
    ;;
  6)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d2_u001_t63_kl,dlight_sb_m/benchmark/d2_u001_t63_mse,dlight_sb_m/benchmark/d2_u001_t15_kl,dlight_sb_m/benchmark/d2_u001_t15_mse
    ;;
  7)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d16_g002_t63_kl,dlight_sb_m/benchmark/d16_g002_t63_mse,dlight_sb_m/benchmark/d16_g002_t15_kl,dlight_sb_m/benchmark/d16_g002_t15_mse
    ;;
  8)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d16_g005_t63_kl,dlight_sb_m/benchmark/d16_g005_t63_mse,dlight_sb_m/benchmark/d16_g005_t15_kl,dlight_sb_m/benchmark/d16_g005_t15_mse
    ;;
  9)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d16_u0005_t63_kl,dlight_sb_m/benchmark/d16_u0005_t63_mse,dlight_sb_m/benchmark/d16_u0005_t15_kl,dlight_sb_m/benchmark/d16_u0005_t15_mse
    ;;
  10)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d16_u001_t63_kl,dlight_sb_m/benchmark/d16_u001_t63_mse,dlight_sb_m/benchmark/d16_u001_t15_kl,dlight_sb_m/benchmark/d16_u001_t15_mse
    ;;
  11)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d64_g002_t63_kl,dlight_sb_m/benchmark/d64_g002_t63_mse,dlight_sb_m/benchmark/d64_g002_t15_kl,dlight_sb_m/benchmark/d64_g002_t15_mse
    ;;
  12)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d64_g005_t63_kl,dlight_sb_m/benchmark/d64_g005_t63_mse,dlight_sb_m/benchmark/d64_g005_t15_kl,dlight_sb_m/benchmark/d64_g005_t15_mse
    ;;
  13)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d64_u0005_t63_kl,dlight_sb_m/benchmark/d64_u0005_t63_mse,dlight_sb_m/benchmark/d64_u0005_t15_kl,dlight_sb_m/benchmark/d64_u0005_t15_mse
    ;;
  14)
    python -m src.run -m \
      seed="$SEED" \
      hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.launcher.backend=loky \
      experiment=dlight_sb_m/benchmark/d64_u001_t63_kl,dlight_sb_m/benchmark/d64_u001_t63_mse,dlight_sb_m/benchmark/d64_u001_t15_kl,dlight_sb_m/benchmark/d64_u001_t15_mse
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac