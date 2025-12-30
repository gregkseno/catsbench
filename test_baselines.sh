#!/bin/bash
#SBATCH --job-name=test-baselines
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source activate dot_bench

papermill \
    notebooks/benchmark_baselines.ipynb \
    notebooks/benchmark_baselines_log.ipynb