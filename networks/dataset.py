import os
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch  
from torch.utils.data import Dataset 

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

class FeatureDataset(Dataset):
    def __init__(self, dataset_path, type='train', transform=None, target_transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform
        data = np.load(dataset_path, 'r')
        self.features = torch.tensor(data['X_' + type], dtype=torch.float32)
        self.n_classes = len(np.unique(data['y_' + type]))
        self.labels = torch.tensor(data['y_' + type], dtype=torch.long)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx, :]
        label = self.labels[idx]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        return features, label
    

    def get_dim(self):
        return self.features.shape[1]
    
    def get_n_classes(self):
        return self.n_classes