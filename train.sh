#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source activate dot_bench

python -m src.run data.num_workers=3 \
    experiment=dlight_sb/benchmark_hd/d64_g002