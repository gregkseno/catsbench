from omegaconf import OmegaConf
import torch
from torch import nn

from src.methods.codecs.modules.blocks import Encoder, Decoder
from src.methods.codecs.modules.vector_quantizer import VectorQuantizer2 as VectorQuantizer

class VQModel(nn.Module):
    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
class VQGAN(nn.Module):
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
    ) -> None:
        super().__init__()
        self.config = OmegaConf.load(config_path).model.params
        self.model = VQModel(**self.config)
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.eval()
        self.centroids = self.model.quantize.embedding.weight.data

    @torch.no_grad()
    def encode_to_cats(self, images: torch.Tensor) -> torch.Tensor:
        images = 2 * images - 1
        _, _, (_, _, cats) = self.model.encode(images)
        cats = cats.reshape(images.shape[0], -1)
        return cats.long()
    
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
    def device(self):
        return next(self.parameters()).device