#!/bin/bash
#SBATCH --job-name=benchmark_images
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source activate dot_bench
SEED=1

python -m src.run \
  seed=${SEED} data.num_workers=7 trainer.devices=1 \
  experiment=dlight_sb/benchmark_image/image_g001 task_name=test \
  ckpt_path=/trinity/home/g.ksenofontov/Projects/dot_bench/logs/article/benchmark_images/dlight_sb/g001/last.ckpt