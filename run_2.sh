#!/bin/bash
#SBATCH --job-name=bench_dlight
#SBATCH --partition=ais-gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --time=16-00:00:00

source activate dot_bench
SEED=1
export HYDRA_FULL_ERROR=1
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run -m \
      seed=${SEED} data.num_workers=1 data.pin_memory=true \
      experiment=dlight_sb/benchmark_images/images_g001
    ;;
  1)
    python -m src.run -m \
      seed=${SEED} data.num_workers=1 data.pin_memory=true \
      experiment=dlight_sb/benchmark_images/images_g002
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac