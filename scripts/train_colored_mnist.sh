#!/bin/bash
#SBATCH --job-name=train-colored-mnist
#SBATCH --partition=ais-gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5%2
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=6-00:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench

SEED=5
EXPERIMENTS=(t2 t4 t10 t25 t50 t100)
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

python -m src.run \
  seed=${SEED} data.num_workers=4 data.pin_memory=true \
  experiment=alpha_csbm/colored_mnist/${EXPERIMENT}
