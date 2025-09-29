#!/bin/bash
#SBATCH --job-name=benchmark_images
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2667
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00

source activate dot_bench
SEED=1

python -m src.run \
  seed=${SEED} data.num_workers=7 data.pin_memory=true trainer.devices=2 \
  experiment=dlight_sb/benchmark_images/images_g001 task_name=test \
  ckpt_path=/trinity/home/g.ksenofontov/Projects/dot_bench/logs/article/images/dlight_sb/g001/last.ckpt