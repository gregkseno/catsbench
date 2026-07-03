from typing import List, Optional

import os
import uuid
import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.ranked_logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(
    logger_cfg: DictConfig, output_dir: Optional[str] = None
) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    loggers: List[Logger] = []

    if not logger_cfg:
        logger.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    logger_id = None
    if output_dir:
        run_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(run_dir, exist_ok=True)
        logger_id_path = os.path.join(run_dir, "logger_id")
        try:
            with open(logger_id_path, "x", encoding="utf-8") as file:
                file.write(uuid.uuid4().hex)
        except FileExistsError:
            pass
        with open(logger_id_path, encoding="utf-8") as file:
            logger_id = file.read().strip()
        if not logger_id:
            raise RuntimeError(f"Logger ID file is empty: {logger_id_path}")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            kwargs = {}
            if logger_id and lg_conf._target_.endswith("WandbLogger"):
                kwargs["id"] = logger_id
            elif logger_id and lg_conf._target_.endswith("CometLogger"):
                kwargs.update(experiment_key=logger_id, mode="get_or_create")
            loggers.append(hydra.utils.instantiate(lg_conf, **kwargs))

    return loggers

def get_run_directory_from_checkpoint(
    ckpt_path: Optional[str], default_dir: str, base_dir: Optional[str] = None
) -> str:
    """Return the checkpoint's run directory or the default Hydra directory."""
    search_dir = os.path.join(base_dir, default_dir) if base_dir else default_dir

    if ckpt_path == "auto":
        run_parent = os.path.dirname(os.path.expanduser(search_dir))
        if not os.path.isdir(run_parent):
            return default_dir
        candidates = [
            os.path.join(run_parent, name)
            for name in os.listdir(run_parent)
            if os.path.isfile(
                os.path.join(run_parent, name, "checkpoints", "last.ckpt")
            )
        ]
        run_dir = max(candidates, key=os.path.basename, default=search_dir)
        return os.path.relpath(run_dir, base_dir) if base_dir else run_dir
    if not ckpt_path or ckpt_path in {"best", "last"} or "://" in ckpt_path:
        return default_dir

    checkpoint = os.path.expanduser(ckpt_path)
    checkpoint_dir = os.path.dirname(checkpoint)
    if (
        checkpoint.endswith(".ckpt")
        and os.path.basename(checkpoint_dir) == "checkpoints"
    ):
        run_dir = os.path.dirname(checkpoint_dir)
        return os.path.relpath(run_dir, base_dir) if base_dir else run_dir
    return default_dir
