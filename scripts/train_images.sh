#!/bin/bash
#SBATCH --job-name=bench_dlight
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2743
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00

module load gpu/cuda-12.3

source activate dot_bench
SEED=1
export HYDRA_FULL_ERROR=1
# case "${SLURM_ARRAY_TASK_ID}" in
#   0)
#     python -m src.run \
#       seed=${SEED} data.num_workers=7 data.pin_memory=true \
#       experiment=dlight_sb/benchmark_images/images_g001
#     ;;
#   1)
#     python -m src.run \
#       seed=${SEED} data.num_workers=7 data.pin_memory=true \
#       experiment=dlight_sb/benchmark_images/images_g002
#     ;;
#   *)
#     echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
#     exit 2
#     ;;
# esac

python -m src.run \
  seed=${SEED} data.num_workers=7 data.pin_memory=false \
  experiment=csbm/benchmark_image/image_g001 \
  +trainer.check_val_every_n_epoch=1
