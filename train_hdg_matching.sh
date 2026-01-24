#!/bin/bash
#SBATCH --job-name=train-hdg-matching
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11%1
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=6-00:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=5
METHOD=dlight_sb_m # (dlight_sb_m csbm alpha_csbm)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d2_g002_t63_kl,${METHOD}/benchmark_hdg/d2_g002_t63_mse,${METHOD}/benchmark_hdg/d2_g002_t15_kl,${METHOD}/benchmark_hdg/d2_g002_t15_mse
    ;;
  1)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d2_g005_t63_kl,${METHOD}/benchmark_hdg/d2_g005_t63_mse,${METHOD}/benchmark_hdg/d2_g005_t15_kl,${METHOD}/benchmark_hdg/d2_g005_t15_mse
    ;;
  2)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d2_u0005_t63_kl,${METHOD}/benchmark_hdg/d2_u0005_t63_mse,${METHOD}/benchmark_hdg/d2_u0005_t15_kl,${METHOD}/benchmark_hdg/d2_u0005_t15_mse
    ;;
  3)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d2_u001_t63_kl,${METHOD}/benchmark_hdg/d2_u001_t63_mse,${METHOD}/benchmark_hdg/d2_u001_t15_kl,${METHOD}/benchmark_hdg/d2_u001_t15_mse
    ;;
  4)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d16_g002_t63_kl,${METHOD}/benchmark_hdg/d16_g002_t63_mse,${METHOD}/benchmark_hdg/d16_g002_t15_kl,${METHOD}/benchmark_hdg/d16_g002_t15_mse
    ;;
  5)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d16_g005_t63_kl,${METHOD}/benchmark_hdg/d16_g005_t63_mse,${METHOD}/benchmark_hdg/d16_g005_t15_kl,${METHOD}/benchmark_hdg/d16_g005_t15_mse
    ;;
  6)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d16_u0005_t63_kl,${METHOD}/benchmark_hdg/d16_u0005_t63_mse,${METHOD}/benchmark_hdg/d16_u0005_t15_kl,${METHOD}/benchmark_hdg/d16_u0005_t15_mse
    ;;
  7)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d16_u001_t63_kl,${METHOD}/benchmark_hdg/d16_u001_t63_mse,${METHOD}/benchmark_hdg/d16_u001_t15_kl,${METHOD}/benchmark_hdg/d16_u001_t15_mse
    ;;
  8)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d64_g002_t63_kl,${METHOD}/benchmark_hdg/d64_g002_t63_mse,${METHOD}/benchmark_hdg/d64_g002_t15_kl,${METHOD}/benchmark_hdg/d64_g002_t15_mse
    ;;
  9)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d64_g005_t63_kl,${METHOD}/benchmark_hdg/d64_g005_t63_mse,${METHOD}/benchmark_hdg/d64_g005_t15_kl,${METHOD}/benchmark_hdg/d64_g005_t15_mse
    ;;
  10)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d64_u0005_t63_kl,${METHOD}/benchmark_hdg/d64_u0005_t63_mse,${METHOD}/benchmark_hdg/d64_u0005_t15_kl,${METHOD}/benchmark_hdg/d64_u0005_t15_mse
    ;;
  11)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hdg/d64_u001_t63_kl,${METHOD}/benchmark_hdg/d64_u001_t63_mse,${METHOD}/benchmark_hdg/d64_u001_t15_kl,${METHOD}/benchmark_hdg/d64_u001_t15_mse
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac