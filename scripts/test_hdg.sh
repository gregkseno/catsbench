#!/bin/bash
#SBATCH --job-name=test-hdg
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2743
#SBATCH --gpus=1
#SBATCH --array=0-11%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=2
METHOD=dlight_sb # (dlight_sb cnot)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d2_g002 task_name=test ckpt_path=auto
    ;;
  1)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d2_g005 task_name=test ckpt_path=auto
    ;;
  2)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d2_u0005 task_name=test ckpt_path=auto
    ;;
  3)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d2_u001 task_name=test ckpt_path=auto
    ;;
  4)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d16_g002 task_name=test ckpt_path=auto
    ;;
  5)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d16_g005 task_name=test ckpt_path=auto
    ;;
  6)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d16_u0005 task_name=test ckpt_path=auto
    ;;
  7)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d16_u001 task_name=test ckpt_path=auto
    ;;
  8)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d64_g002 task_name=test ckpt_path=auto
    ;;
  9)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d64_g005 task_name=test ckpt_path=auto
    ;;
  10)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d64_u0005 task_name=test ckpt_path=auto
    ;;
  11)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hdg/d64_u001 task_name=test ckpt_path=auto
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac