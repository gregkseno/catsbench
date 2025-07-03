import sys
from typing import Any, Literal, Optional, Tuple

import numpy as np
from PIL import Image
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from tqdm.auto import tqdm
tqdm.pandas()


class ImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

class CelebaDataset(Dataset):
    transform: Optional[transforms.Compose] = None
    
    def __init__(
        self, 
        sex: Literal['male', 'female', 'both'], 
        data_dir: str,
        size: Optional[int] = None, 
        train: bool = True,
        split: int | float = 162771, # from original dataset
        use_quantized: bool = True,
        return_names: bool = False
    ):
        self.train = train
        self.use_quantized = use_quantized
        self.size = size
        self.return_names = return_names
        self.data_dir= data_dir

        subset = pd.read_csv(os.path.join(data_dir, 'celeba', 'list_attr_celeba.csv'))

        if isinstance(split, int): 
            # this logic mathches setup of previously trained models
            subset = subset.iloc[:split] if train else subset.iloc[split:]
            if sex == 'male':
                subset = subset[subset['Male'] != -1]
            elif sex == 'female':
                subset = subset[subset['Male'] == -1]
            else:
                subset = subset
        else:
            # this logic mathches asbm setup
            male_subset = subset[subset['Male'] != -1]
            female_subset = subset[subset['Male'] == -1]
            male_split_index, female_split_index = int(len(male_subset) * split), int(len(female_subset) * split)
            
            male_subset = male_subset[:male_split_index] if train else male_subset.iloc[male_split_index:]
            female_subset = female_subset[:female_split_index] if train else female_subset.iloc[female_split_index:]

            if sex == 'male':
                subset = male_subset
            elif sex == 'female':
                subset = female_subset
            else:
                subset = pd.concat([male_subset, female_subset], ignore_index=True)
                subset = subset.sort_values(by='image_id').reset_index(drop=True)


        if use_quantized:
            sub_folder = 'quantized'
            subset['image_id'] = subset['image_id'].str.removesuffix('.jpg') + '.npy'
        else:
            sub_folder = 'raw'

        self.image_names = subset['image_id']
        self.dataset = [os.path.join(data_dir, 'celeba', 'img_align_celeba', sub_folder, image) for image in self.image_names.tolist()]

    def __getitem__(self, index):
        if self.train and self.use_quantized:
            image = torch.from_numpy(np.load(self.dataset[index]))
        else:
            transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
            ])
            image = Image.open(self.dataset[index])
            image = image.convert('RGB')
            image = transform(image)

        if self.return_names:
           return image, self.dataset[index].split('/')[-1]
        return image

    def __len__(self):
        return len(self.dataset)
    
    def get_by_filename(self, index):
        transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
        ])
        # image = self.image_names[self.image_names == index].item()
        image = Image.open(os.path.join(self.data_dir, 'celeba', 'img_align_celeba', 'raw', index))
        image = image.convert('RGB')
        image = transform(image)
        return image
    
    @staticmethod
    def quantize_train(
        model: nn.Module, 
        data_dir: str,
        size: int = 128, 
        batch_size: int = 32,
    ):
        load_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        data_dir = os.path.join(data_dir, 'celeba', 'img_align_celeba')
        save_path = os.path.join(data_dir, 'quantized')
        dataset = ImageFolder(data_dir, transform=load_transform, allow_empty=True) # allow_eppty because it will be handled in next line
        if 'quantized' in dataset.classes:
            raise FileExistsError('Folder with quantized images already exists!')
        else:
            os.makedirs(save_path, exist_ok=True)

        dataloader = DataLoader(dataset, batch_size=batch_size)
        for images, image_paths in tqdm(dataloader, file=sys.stdout):
            images = images.to(model.device)
            encoded_images = model.encode_to_cats(images).cpu().detach().numpy()
            for encoded_image, image_path in zip(encoded_images, image_paths):
                file_name = image_path.split('/')[-1].split('.')[0]
                image_path = os.path.join(save_path, file_name)
                np.save(image_path, encoded_image)  