#!/bin/bash
#SBATCH --job-name=train-celeba
#SBATCH --partition=ais-gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=6-00:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench

SEED=5
EXPERIMENTS=(u0005 u001)
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

python -m src.run \
  seed=${SEED} data.num_workers=2 data.pin_memory=true \
  experiment=alpha_csbm/celeba/${EXPERIMENT}
