#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00

source activate dot_bench

python -m src.run data.num_workers=3 trainer.devices=4 \
    experiment=csbm/colored_mnist/t10 