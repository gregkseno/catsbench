from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms, datasets

def _continuous_to_discrete(
    batch: Union[torch.Tensor, np.ndarray], 
    num_categories: int,
    quantize_range: Optional[Tuple[Union[int, float], Union[int, float]]] = None
):
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).contiguous()
    if quantize_range is None:
        quantize_range = (-3, 3)
    bin_edges = torch.linspace(
        quantize_range[0], 
        quantize_range[1], 
        num_categories - 1
    )
    discrete_batch = torch.bucketize(batch, bin_edges)
    return discrete_batch

class DiscreteUniformDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_categories: int = 100, train: bool = True
    ):
        dataset = 6 * torch.rand(size=(num_samples, dim)) - 3
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = _continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset  

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

class DiscreteGaussianDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_categories: int = 100, train: bool = True
    ):          
        dataset = torch.randn(size=[num_samples, dim])
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = _continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

class DiscreteColoredMNISTDataset(Dataset):
    def __init__(
        self, 
        target_digit: int, 
        data_dir: str, 
        train: bool = True, 
        img_size: int = 32
    ):
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda image: self._get_random_colored_images(image))
        ])
        
        dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
        dataset = torch.stack(
            [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == target_digit],
            dim=0
        )
        dataset = (255 * dataset).to(dtype=torch.int64)
        self.dataset = dataset      

    def _get_random_colored_images(self, image: torch.Tensor):
        hue = 360 * torch.rand(1)
        image_min = 0
        image_diff = (image - image_min) * (hue % 60) / 60
        image_inc = image_diff
        image_dec = image - image_diff
        colored_image = torch.zeros((3, image.shape[1], image.shape[2]))
        H_i = torch.round(hue / 60) % 6 # type: ignore
        
        if H_i == 0:
            colored_image[0] = image
            colored_image[1] = image_inc
            colored_image[2] = image_min
        elif H_i == 1:
            colored_image[0] = image_dec
            colored_image[1] = image
            colored_image[2] = image_min
        elif H_i == 2:
            colored_image[0] = image_min
            colored_image[1] = image
            colored_image[2] = image_inc
        elif H_i == 3:
            colored_image[0] = image_min
            colored_image[1] = image_dec
            colored_image[2] = image
        elif H_i == 4:
            colored_image[0] = image_inc
            colored_image[1] = image_min
            colored_image[2] = image
        elif H_i == 5:
            colored_image[0] = image
            colored_image[1] = image_min
            colored_image[2] = image_dec
        
        return colored_image