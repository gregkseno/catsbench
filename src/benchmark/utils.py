import torch
import numpy as np

from PIL import Image
from matplotlib.figure import Figure


def broadcast(tensor: torch.Tensor, num_add_dims: int, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim += tensor.dim() + 1
    shape = [*tensor.shape[:dim], *([1] * num_add_dims), *tensor.shape[dim:]]
    return tensor.reshape(*shape)

def log_space_product(log_matrix1: torch.Tensor, log_matrix2: torch.Tensor) -> torch.Tensor: 
    log_matrix1 = log_matrix1[..., :, None]
    log_matrix2 = log_matrix2[..., None, :, :]
    return torch.logsumexp(log_matrix1 + log_matrix2, dim=-2)

def sample_separated_means(num_potentials, dim, num_categories, min_dist=5, max_attempts=5000):
    means = []
    attempts = 0
    low, high = 5, num_categories - 5

    while len(means) < num_potentials and attempts < max_attempts:
        candidate = torch.randint(low, high, (dim,))
        if all(torch.norm(candidate - m.float()) >= min_dist for m in means):
            means.append(candidate)
        attempts += 1

    if len(means) < num_potentials:
        raise RuntimeError(f"Could only generate {len(means)} points with min_dist={min_dist}")
    
    return torch.stack(means)

def fig2img(fig: Figure) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis = 2)
    return Image.frombytes("RGBA", (w, h), buf.tobytes())
