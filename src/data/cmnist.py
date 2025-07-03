import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


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

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

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