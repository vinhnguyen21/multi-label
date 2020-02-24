import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.utils.data as utils
# from src.transformation import autoaugment
class RetinalDataset(utils.Dataset):
    def __init__(self, inputs, labels, image_folder, size,transform):
        self.inputs = inputs
        self.labels = labels
        self.image_folder = image_folder
        self.size = size
        self.transform = transform
    def __getitem__(self, index):
        image = self.inputs[index]
        image_path = os.path.join(self.image_folder, image)
        image_label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image, self.size)
        else:
            image = image.resize(self.size)
            return transforms.ToTensor(image), torch.from_numpy(np.array(image_label).astype(np.float32))
        return image, torch.from_numpy(np.array(image_label).astype(np.float32))
    def __len__(self):
        return len(self.inputs)
