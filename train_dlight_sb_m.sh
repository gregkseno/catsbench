#!/bin/bash
#SBATCH --job-name=bench_all
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-31%4
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=16-00:00:00
#SBATCH --output=logs/bench_dlight_sb_m_%A_%a.out

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=5

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_g002_t63_kl
    ;;
  1)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_g002_t63_mse
    ;;
  2)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_g002_t15_kl
    ;;
  3)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_g002_t15_mse
    ;;
  4)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_g005_t63_kl
    ;;
  5)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_g005_t63_mse
    ;;
  6)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_g005_t15_kl
    ;;
  7)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_g005_t15_mse
    ;;
  8)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_u0005_t63_kl
    ;;
  9)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_u0005_t63_mse
    ;;
  10)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_u0005_t15_kl
    ;;
  11)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_u0005_t15_mse
    ;;
  12)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_u001_t63_kl
    ;;
  13)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_u001_t63_mse
    ;;
  14)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_u001_t15_kl
    ;;
  15)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d2_u001_t15_mse
    ;;
  16)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_g002_t63_kl
    ;;
  17)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_g002_t63_mse
    ;;
  18)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_g002_t15_kl
    ;;
  19)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_g002_t15_mse
    ;;
  20)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_g005_t63_kl
    ;;
  21)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_g005_t63_mse
    ;;
  22)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_g005_t15_kl
    ;;
  23)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_g005_t15_mse
    ;;
  24)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_u0005_t63_kl
    ;;
  25)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_u0005_t63_mse
    ;;
  26)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_u0005_t15_kl
    ;;
  27)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_u0005_t15_mse
    ;;
  28)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_u001_t63_kl
    ;;
  29)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_u001_t63_mse
    ;;
  30)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_u001_t15_kl
    ;;
  31)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d16_u001_t15_mse
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac