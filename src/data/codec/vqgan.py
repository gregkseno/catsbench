import torch
from omegaconf import OmegaConf

from ...vq_diffusion.taming.models.vqgan import VQModel
from .base import BaseCodec


class VQGANCodec(BaseCodec):
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
    ) -> None:
        super().__init__()
        self.config = OmegaConf.load(config_path).model.params
        self.model = VQModel(**self.config).eval()
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.centroids = self.model.quantize.embedding.weight.data

    @torch.no_grad()
    def encode_to_cats(self, images: torch.Tensor) -> torch.Tensor:
        images = 2 * images - 1
        _, _, (_, _, cats) = self.model.encode(images)
        return cats.reshape(images.shape[0], -1).long()

    @torch.no_grad()
    def decode_to_image(self, cats: torch.Tensor) -> torch.Tensor:
        shape = (
            cats.shape[0], 
            int(self.config.embed_dim ** 0.5), 
            int(self.config.embed_dim ** 0.5), 
            int(self.config.ddconfig.z_channels)
        )
        z_q = self.model.quantize.get_codebook_entry(cats, shape)
        images = self.model.decode(z_q)
        images = torch.clamp(images, -1., 1.)
        images = (images + 1.) / 2.
        return images
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
