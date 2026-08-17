from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig

from src.methods.base import BaseMethod
from src.utils import instantiate_callbacks, instantiate_loggers
from src.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class BaseWorkflow(ABC):
    """Owns the Lightning training procedure around a method."""

    def __init__(
        self,
        trainer_config: DictConfig,
        callbacks_config: Optional[DictConfig],
        logger_config: Optional[DictConfig],
        output_dir: str,
    ) -> None:
        self.trainer_config = trainer_config
        self.callbacks_config = callbacks_config
        self.logger_config = logger_config
        self.output_dir = output_dir

    def make_trainer(self, hparams: Mapping[str, Any]) -> Trainer:
        """Instantiate a fresh Trainer and its callbacks and loggers."""
        log.info("Instantiating callbacks...")
        callbacks = instantiate_callbacks(self.callbacks_config)

        log.info("Instantiating loggers...")
        loggers = instantiate_loggers(self.logger_config, self.output_dir)
        for logger in loggers:
            logger.log_hyperparams(hparams)

        log.info(f"Instantiating trainer <{self.trainer_config._target_}>...")
        return instantiate(
            self.trainer_config,
            callbacks=callbacks,
            logger=loggers,
        )

    @abstractmethod
    def run(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        ckpt_path: Optional[str],
        hparams: Mapping[str, Any],
    ) -> Trainer:
        """Run the training procedure and return its Trainer."""

