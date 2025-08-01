#!/bin/bash
#SBATCH --job-name=train-toy
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2665
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16-00:00:00

source activate dot_bench
python -m src.run experiment=light_sb_d_benchmark +trainer.max_epochs=10 \
    data.dim=16 \
    prior.prior_type=uniform \
    prior.alpha=0.005