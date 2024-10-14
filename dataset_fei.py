import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from PIL import Image


import os
import cv2
from torch.utils.data import Dataset



class PUFDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.categories = ["puf1", "puf2", "puf3", "puf4", "puf5"]  
        # self.categories = ["0", "1", "2", "3", "4"] 
        # self.categories = ["class1", "class2", "class3", "class4", "class5"]
        self.transform = transform

        if train:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'test')

        for i, category in enumerate(self.categories):
            category_path = os.path.join(data_path, category)
            for item in os.listdir(category_path):
                item_path = os.path.join(category_path, item)
                self.image_paths.append(item_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # image = cv2.imread(image_path, 0)
        image = np.array(Image.open(image_path).convert('L'))
        # print(image)
        # image = (image > 128).astype(np.uint8)
        # print(image.shape)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

        
    
        
        
                 
        
