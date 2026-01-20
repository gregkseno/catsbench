from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.cuda.get_device_capability = lambda x: (7, None)
except ImportError:
    pass
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from src.utils.ranked_logger import RankedLogger
from src.utils import instantiate_callbacks, instantiate_loggers


if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    if major >= 8: 
        torch.set_float32_matmul_precision("high")

def _detect_config_dir() -> str:
    candidates = []
    ds_home = os.environ.get('DS_PROJECT_HOME')
    if ds_home: candidates.append(os.path.join(ds_home, 'configs'))

    here = os.path.abspath(__file__)
    parent = os.path.dirname(os.path.dirname(here))  # go up 2 levels
    candidates.append(os.path.join(parent, 'configs'))

    for p in candidates:
        if os.path.isdir(p): return p

    tried = ' | '.join(candidates)
    raise RuntimeError(f'[Hydra] Config directory not found. Tried: {tried}')

log = RankedLogger(__name__, rank_zero_only=True)
CONFIG_DIR = _detect_config_dir()

@hydra.main(version_base='1.1', config_path=CONFIG_DIR, config_name='config.yaml')
def main(config: DictConfig):
    if config.get('seed'):
        L.seed_everything(config.seed, workers=True)
    
    if config.data.num_workers > 0 and config.get('trainer.strategy') is None:
        try:
            import torch.multiprocessing as mp
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # NOTE: hydra will instantiate all subobjects of the object recursively
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
    log.info(f'Instantiating datamodule <{config.data._target_}>...')
    datamodule: LightningDataModule = instantiate(config.data)
    
    log.info(f'Instantiating method <{config.method._target_}>...')
    #print(config)
    method: LightningModule = instantiate(config.method)

    log.info('Instantiating callbacks...')
    callbacks: List[Callback] = instantiate_callbacks(config.get('callbacks'))

    log.info('Instantiating loggers...')
    loggers: List[Logger] = instantiate_loggers(config.get('logger'))
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_container(config))

    log.info(f'Instantiating trainer <{config.trainer._target_}>...')
    trainer: Trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers)
    
    if config.task_name == 'train':
        log.info('Starting training!')
        trainer.fit(model=method, datamodule=datamodule, ckpt_path=config.get('ckpt_path'))
    elif config.task_name == 'test':
        assert config.get('ckpt_path') is not None, 'The `ckpt_path` must be provided for testing!'
        ckpt_path = config.get('ckpt_path')
        if ckpt_path == 'auto':
            log.info('Auto-detecting the latest checkpoint path...')
            log_dir = config.paths.log_dir
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            data_choice = hydra_cfg.runtime.choices.data
            method_choice = hydra_cfg.runtime.choices.method
            experiment_choice = hydra_cfg.runtime.choices.experiment
            exp_dir = os.path.join(
                log_dir, 'runs', data_choice, method_choice, experiment_choice, str(config.seed)
            )
            log.info(f'data_choice: {data_choice}')
            log.info(f'method_choice: {method_choice}')
            log.info(f'experiment_choice: {experiment_choice}')
            log.info(f'config.seed: {config.seed}')
            # select the latest subdir by folder name of format 'YYYY-MM-DD_HH-MM-SS'
            # and if this folder contains train.log file
            subdirs = [os.path.join(exp_dir, subdir) for subdir in os.listdir(exp_dir)]
            subdirs = [subdir for subdir in subdirs if 'train.log' in os.listdir(subdir)]
            latest_subdir = max(subdirs, key=lambda x: os.path.basename(x))
            ckpts = os.listdir(os.path.join(latest_subdir, 'checkpoints'))
            ckpts = [ckpt for ckpt in ckpts if ckpt.startswith('epoch_')]
            last_ckpt = max(ckpts, key=lambda x: int(x.split('_')[1].split('.')[0]))
            ckpt_path = os.path.join(latest_subdir, 'checkpoints', last_ckpt)       
        
        log.info(f'Starting testing with ckpt_path: {ckpt_path}.')
        trainer.test(model=method, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        raise ValueError(f'Unknown task name: {config.task_name}!')

if __name__ == '__main__':
    main()
