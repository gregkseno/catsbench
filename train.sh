#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2966
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=12:00:00

module load gpu/cuda-12.3
module load compilers/gcc-12.2.0
source activate dot_bench

HYDRA_FULL_ERROR=1 python -m src.run \
    experiment=dlight_sb_m/benchmark_hd/d2_g002_t63_kl seed=5
