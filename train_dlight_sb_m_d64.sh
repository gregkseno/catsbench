#!/bin/bash
#SBATCH --job-name=bench_all
#SBATCH --partition=ais-gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=16-00:00:00
#SBATCH --output=logs/bench_dlight_sb_m_d64_%A_%a.out

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=2

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_g002_t63_kl \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5
    ;;
  1)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_g002_t63_mse \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  2)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_g002_t15_kl \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  3)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_g002_t15_mse \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  4)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_g005_t63_kl \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  5)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_g005_t63_mse \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  6)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_g005_t15_kl \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  7)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_g005_t15_mse \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  8)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_u0005_t63_kl \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  9)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_u0005_t63_mse \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  10)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_u0005_t15_kl \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  11)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_u0005_t15_mse \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  12)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_u001_t63_kl \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  13)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_u001_t63_mse \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  14)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_u001_t15_kl \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  15)
    python -m src.run \
      seed=${SEED} data.num_workers=7 data.pin_memory=true \
      experiment=dlight_sb_m/benchmark/d64_u001_t15_mse \
      trainer=multi_gpu trainer.max_epochs=5 trainer.limit_train_batches=20000 \
      trainer.val_check_interval=null +trainer.check_val_every_n_epoch=5 
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac