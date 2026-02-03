#!/bin/bash
#SBATCH --job-name=train-hd
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2743
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=6-00:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=5
METHOD=dlight_sb # (dlight_sb cnot)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hd/d2_g002,${METHOD}/benchmark_hd/d2_g005,${METHOD}/benchmark_hd/d2_u0005,${METHOD}/benchmark_hd/d2_u001
    ;;
  1)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hd/d16_g002,${METHOD}/benchmark_hd/d16_g005,${METHOD}/benchmark_hd/d16_u0005,${METHOD}/benchmark_hd/d16_u001
    ;;
  2)
    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=1 data.pin_memory=true 'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit' \
      experiment=${METHOD}/benchmark_hd/d64_g002,${METHOD}/benchmark_hd/d64_g005,${METHOD}/benchmark_hd/d64_u0005,${METHOD}/benchmark_hd/d64_u001
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac