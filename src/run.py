from typing import List

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger, CometLogger

from src.utils.logging.console import RankedLogger
from src.utils import instantiate_callbacks, instantiate_loggers


log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base='1.1', config_path='../configs', config_name='config.yaml')
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
