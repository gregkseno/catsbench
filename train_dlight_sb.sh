#!/bin/bash
#SBATCH --job-name=bench_dlight_sb
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2%1
#SBATCH --nodes=1
#SBATCH --mem=70GB
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/bench_dlight_sb_%A_%a.out

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=42

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=70 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      +data.reversed=true \
      experiment=dlight_sb/benchmark_hdg/d2_g002,dlight_sb/benchmark_hdg/d2_g005,dlight_sb/benchmark_hdg/d2_u0005,dlight_sb/benchmark_hdg/d2_u001
    ;;
  1)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=70 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      +data.reversed=true \
      experiment=dlight_sb/benchmark_hdg/d16_g002,dlight_sb/benchmark_hdg/d16_g005,dlight_sb/benchmark_hdg/d16_u0005,dlight_sb/benchmark_hdg/d16_u001
    ;;
  2)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=70 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      +data.reversed=true \
      experiment=dlight_sb/benchmark_hdg/d64_g002,dlight_sb/benchmark_hdg/d64_g005,dlight_sb/benchmark_hdg/d64_u0005,dlight_sb/benchmark_hdg/d64_u001
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac