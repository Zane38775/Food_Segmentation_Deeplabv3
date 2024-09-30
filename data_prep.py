import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FoodSegDataset(Dataset):
    def __init__(self, data_dir, img_dir, mask_dir, split='train', transform=None):
        self.img_dir = os.path.join(img_dir, split)
        self.mask_dir = os.path.join(mask_dir, split)

        image_set_path = os.path.join(data_dir, 'ImageSets', f"{split}.txt")
        with open(image_set_path, "r") as f:
            self.images = [line.strip() for line in f]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + '.png')

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 确保图像和掩码具有相同的尺寸
        image = np.array(image)
        mask = np.array(mask)
        
        # 如果尺寸不同，将掩码调整为与图像相同的尺寸
        if image.shape[:2] != mask.shape:
            mask = Image.fromarray(mask).resize((image.shape[1], image.shape[0]), Image.NEAREST)
            mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()