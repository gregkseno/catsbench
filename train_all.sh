#!/bin/bash
#SBATCH --job-name=bench_all
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-26%4
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=16-00:00:00
#SBATCH --output=logs/bench_all_%A_%a.out

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=5

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=dlight_sb/benchmark/d2_g002,dlight_sb/benchmark/d2_g005,dlight_sb/benchmark/d2_u0005,dlight_sb/benchmark/d2_u001
    ;;
  1)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=dlight_sb/benchmark/d16_g002,dlight_sb/benchmark/d16_g005,dlight_sb/benchmark/d16_u0005,dlight_sb/benchmark/d16_u001
    ;;
  2)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=dlight_sb/benchmark/d64_g002,dlight_sb/benchmark/d64_g005,dlight_sb/benchmark/d64_u0005,dlight_sb/benchmark/d64_u001
    ;;
  3)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d2_g002_t63_kl,alpha_csbm/benchmark/d2_g002_t63_mse,alpha_csbm/benchmark/d2_g002_t15_kl,alpha_csbm/benchmark/d2_g002_t15_mse
    ;;
  4)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d2_g005_t63_kl,alpha_csbm/benchmark/d2_g005_t63_mse,alpha_csbm/benchmark/d2_g005_t15_kl,alpha_csbm/benchmark/d2_g005_t15_mse
    ;;
  5)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d2_u0005_t63_kl,alpha_csbm/benchmark/d2_u0005_t63_mse,alpha_csbm/benchmark/d2_u0005_t15_kl,alpha_csbm/benchmark/d2_u0005_t15_mse
    ;;
  6)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d2_u001_t63_kl,alpha_csbm/benchmark/d2_u001_t63_mse,alpha_csbm/benchmark/d2_u001_t15_kl,alpha_csbm/benchmark/d2_u001_t15_mse
    ;;
  7)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d16_g002_t63_kl,alpha_csbm/benchmark/d16_g002_t63_mse,alpha_csbm/benchmark/d16_g002_t15_kl,alpha_csbm/benchmark/d16_g002_t15_mse
    ;;
  8)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d16_g005_t63_kl,alpha_csbm/benchmark/d16_g005_t63_mse,alpha_csbm/benchmark/d16_g005_t15_kl,alpha_csbm/benchmark/d16_g005_t15_mse
    ;;
  9)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d16_u0005_t63_kl,alpha_csbm/benchmark/d16_u0005_t63_mse,alpha_csbm/benchmark/d16_u0005_t15_kl,alpha_csbm/benchmark/d16_u0005_t15_mse
    ;;
  10)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d16_u001_t63_kl,alpha_csbm/benchmark/d16_u001_t63_mse,alpha_csbm/benchmark/d16_u001_t15_kl,alpha_csbm/benchmark/d16_u001_t15_mse
    ;;
  11)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d64_g002_t63_kl,alpha_csbm/benchmark/d64_g002_t63_mse,alpha_csbm/benchmark/d64_g002_t15_kl,alpha_csbm/benchmark/d64_g002_t15_mse
    ;;
  12)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d64_g005_t63_kl,alpha_csbm/benchmark/d64_g005_t63_mse,alpha_csbm/benchmark/d64_g005_t15_kl,alpha_csbm/benchmark/d64_g005_t15_mse
    ;;
  13)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d64_u0005_t63_kl,alpha_csbm/benchmark/d64_u0005_t63_mse,alpha_csbm/benchmark/d64_u0005_t15_kl,alpha_csbm/benchmark/d64_u0005_t15_mse
    ;;
  14)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=alpha_csbm/benchmark/d64_u001_t63_kl,alpha_csbm/benchmark/d64_u001_t63_mse,alpha_csbm/benchmark/d64_u001_t15_kl,alpha_csbm/benchmark/d64_u001_t15_mse
    ;;
  15)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d2_g002_t63_kl,csbm/benchmark/d2_g002_t63_mse,csbm/benchmark/d2_g002_t15_kl,csbm/benchmark/d2_g002_t15_mse
    ;;
  16)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d2_g005_t63_kl,csbm/benchmark/d2_g005_t63_mse,csbm/benchmark/d2_g005_t15_kl,csbm/benchmark/d2_g005_t15_mse
    ;;
  17)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d2_u0005_t63_kl,csbm/benchmark/d2_u0005_t63_mse,csbm/benchmark/d2_u0005_t15_kl,csbm/benchmark/d2_u0005_t15_mse
    ;;
  18)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d2_u001_t63_kl,csbm/benchmark/d2_u001_t63_mse,csbm/benchmark/d2_u001_t15_kl,csbm/benchmark/d2_u001_t15_mse
    ;;
  19)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d16_g002_t63_kl,csbm/benchmark/d16_g002_t63_mse,csbm/benchmark/d16_g002_t15_kl,csbm/benchmark/d16_g002_t15_mse
    ;;
  20)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d16_g005_t63_kl,csbm/benchmark/d16_g005_t63_mse,csbm/benchmark/d16_g005_t15_kl,csbm/benchmark/d16_g005_t15_mse
    ;;
  21)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d16_u0005_t63_kl,csbm/benchmark/d16_u0005_t63_mse,csbm/benchmark/d16_u0005_t15_kl,csbm/benchmark/d16_u0005_t15_mse
    ;;
  22)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d16_u001_t63_kl,csbm/benchmark/d16_u001_t63_mse,csbm/benchmark/d16_u001_t15_kl,csbm/benchmark/d16_u001_t15_mse
    ;;
  23)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d64_g002_t63_kl,csbm/benchmark/d64_g002_t63_mse,csbm/benchmark/d64_g002_t15_kl,csbm/benchmark/d64_g002_t15_mse
    ;;
  24)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d64_g005_t63_kl,csbm/benchmark/d64_g005_t63_mse,csbm/benchmark/d64_g005_t15_kl,csbm/benchmark/d64_g005_t15_mse
    ;;
  25)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d64_u0005_t63_kl,csbm/benchmark/d64_u0005_t63_mse,csbm/benchmark/d64_u0005_t15_kl,csbm/benchmark/d64_u0005_t15_mse
    ;;
  26)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=csbm/benchmark/d64_u001_t63_kl,csbm/benchmark/d64_u001_t63_mse,csbm/benchmark/d64_u001_t15_kl,csbm/benchmark/d64_u001_t15_mse
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac