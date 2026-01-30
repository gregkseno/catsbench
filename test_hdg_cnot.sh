#!/bin/bash
#SBATCH --job-name=test-hdg-cnot
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --array=0-11%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --time=04-00:00:00
#SBATCH --reservation=HPC-2743

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=1

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d2_g002
    ;;
  1)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d2_g005
    ;;
  2)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d2_u001
    ;;
  3)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d2_u0005
    ;;
  4)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d16_g002
    ;;
  5)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d16_g005
    ;;
  6)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d16_u001
    ;;
  7)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d16_u0005
    ;;
  8)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d64_g002
    ;;
  9)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d64_g005
    ;;
  10)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d64_u001
    ;;
  11)
    python -m src.run \
      seed=${SEED} data.num_workers=1 data.pin_memory=false experiment=cnot/benchmark_hdg/d64_u0005
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac