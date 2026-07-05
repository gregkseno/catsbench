import os
from copy import deepcopy
from typing import Any, Literal, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.hub import download_url_to_file

from lightning.pytorch import Trainer
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .base import BaseMetricsCallback
from ..data.batch import Batch
from ..data.codec import BaseCodec
from ..methods import BaseMethod


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"


class FID(FrechetInceptionDistance):
    def __init__(
        self,
        feature: Union[int, nn.Module] = 2048,
        reset_real_features: bool = False,
        normalize: bool = True,
        input_img_size: tuple[int, int, int] = (3, 299, 299),
        feature_extractor_weights_path: Optional[str] = 'checkpoints/fid_weights.ckpt',
        **kwargs: Any,
    ) -> None:
        if feature_extractor_weights_path is not None:
            if not os.path.exists(feature_extractor_weights_path):
                os.makedirs(os.path.dirname(feature_extractor_weights_path), exist_ok=True)
                print(f"Downloading FID weights to {feature_extractor_weights_path}...")
                download_url_to_file(
                    url=FID_WEIGHTS_URL, 
                    dst=feature_extractor_weights_path,
                    progress=True
                )
        super().__init__(
            feature=feature,
            reset_real_features=reset_real_features,
            normalize=normalize,
            input_img_size=input_img_size,
            feature_extractor_weights_path=feature_extractor_weights_path,
            **kwargs,
        )


class CMMD(Metric):
    def __init__(
        self,
        reset_real_features: bool = False,
        normalize: bool = True,
        embedding_extractor_model: str = CLIP_MODEL_NAME,
    ) -> None:
        super().__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(embedding_extractor_model)
        self.model = CLIPVisionModelWithProjection.from_pretrained(embedding_extractor_model).eval()
        self.input_image_size = self.image_processor.crop_size["height"]
        self.normalize = normalize
        self.reset_real_features = reset_real_features
        self.is_resetted = False
    
        self.add_state("real_images", default=torch.zeros(0))
        self.add_state("fake_images", default=torch.zeros(0))

    def _mmd(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        sigma: int = 
        10, scale: int = 1000
    ) -> torch.Tensor:

        x_sqnorms = torch.diag(torch.matmul(x, x.T))
        y_sqnorms = torch.diag(torch.matmul(y, y.T))

        gamma = 1 / (2 * sigma**2)
        k_xx = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
        )
        k_xy = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
        )
        k_yy = torch.mean(
            torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
        )

        return scale * (k_xx + k_yy - 2 * k_xy)

    def update(self, imgs: torch.Tensor, real: bool) -> None:
        if not self.reset_real_features and self.is_resetted and real:
            # We dont want to update the real features 
            # after reset if reset_real_features is False
            return
        imgs = (imgs / 255).float() if not self.normalize else imgs
        imgs = F.interpolate(
            imgs, size=(self.input_image_size, self.input_image_size), mode="bicubic"
        )
        inputs = self.image_processor(
            images=imgs,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_embs = self.model(**inputs).image_embeds
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        if real:
            self.real_images = torch.cat([self.real_images, image_embs], dim=0)
        else:
            self.fake_images = torch.cat([self.fake_images, image_embs], dim=0)

    def compute(self) -> torch.Tensor:
        return self._mmd(self.real_images, self.fake_images)
    
    def reset(self) -> None:
        self.is_resetted = True
        if not self.reset_real_features:
            real_images = deepcopy(self.real_images)
            super().reset()
            self.real_images = real_images
        else:
            super().reset()    

    @property
    def device(self):
        return next(self.parameters()).device


class ImageMetricsCallback(BaseMetricsCallback):
    codec: Optional[BaseCodec] = None

    def __init__(self) -> None:
        super().__init__()
        self.metrics: dict[str, Metric] = {}
        self.test_metrics: dict[str, Metric] = {}

    def _setup_callback(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        stage: Literal["fit", "validate", "test"],
    ) -> None:
        if self.codec is not None and self.metrics and (
            stage != "test" or self.test_metrics
        ):
            return

        if self.codec is None:
            assert hasattr(trainer.datamodule, "codec"), (
                "Wrong datamodule! It should have `codec` attribute"
            )
            self.codec = trainer.datamodule.codec

        if not self.metrics:
            self.metrics = {
                "fid": FID(),
                "lpips": LearnedPerceptualImagePatchSimilarity(normalize=True),
                "mse": MeanSquaredError(),
            }

        if stage == "test" and not self.test_metrics:
            self.test_metrics = {"cmmd": CMMD()}

    @torch.no_grad()
    def _update_metrics(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        batch: Batch,
        batch_idx: int,
        stage: Literal["train", "val", "test"] = "train",
    ) -> None:
        assert self.codec is not None
        if getattr(pl_module, "fb", None) == "backward":
            return

        x_start, x_end = batch.encoded
        x_start_image, x_end_image = batch.raw

        self.codec.to(x_start.device)
        pred_x_end = pl_module.sample(x_start)
        pred_x_end_image = self.codec.decode_to_image(pred_x_end).detach()

        self.metrics["fid"].update(x_end_image, real=True)
        self.metrics["fid"].update(pred_x_end_image, real=False)
        self.metrics["mse"].update(pred_x_end_image, x_start_image)

        if stage == "test":
            self.metrics["lpips"].update(pred_x_end_image, x_start_image)
            self.test_metrics["cmmd"].update(x_end_image, real=True)
            self.test_metrics["cmmd"].update(pred_x_end_image, real=False)

    def _compute_and_log_metrics(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        stage: Literal["train", "val", "test"] = "train",
    ) -> None:
        if getattr(pl_module, "fb", None) == "backward":
            return

        for name, metric in self.metrics.items():
            if name == "lpips" and stage != "test":
                continue
            pl_module.log(
                f"{stage}/{name}_forward",
                metric.compute(),
                sync_dist=False,
            )
            metric.reset()

        if stage == "test":
            for name, metric in self.test_metrics.items():
                pl_module.log(
                    f"{stage}/{name}_forward",
                    metric.compute(),
                    sync_dist=False,
                )
                metric.reset()
