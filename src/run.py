from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from src.utils.logging.console import RankedLogger
from src.utils import instantiate_callbacks, instantiate_loggers

try:
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.cuda.get_device_capability = lambda x: (7, None)
except ImportError:
    pass


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

    # NOTE: hydra will instantiate all subobjects of the object recursively
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
    log.info(f'Instantiating datamodule <{config.data._target_}>...')
    datamodule: LightningDataModule = instantiate(config.data)
    
    log.info(f'Instantiating method <{config.method._target_}>...')
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
        log.info('Starting testing!')
        trainer.test(model=method, datamodule=datamodule, ckpt_path=config.get('ckpt_path'))
    else:
        raise ValueError(f'Unknown task name: {config.task_name}!')

if __name__ == '__main__':
    main()
