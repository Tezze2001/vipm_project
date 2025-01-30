import torch
import os
import random
import numpy as np
from PIL import Image
from torchvision import datasets, transforms


# Funzione per preprocessare un'immagine
def preprocess_image(image_path, target_size, augment, seed=None):
    if seed is not None:
        # Imposta il seed per la riproducibilità
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if augment:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),  # Flip orizzontale con probabilità 50%
            transforms.RandomRotation(degrees=15),  # Rotazione casuale di massimo 15 gradi
            transforms.RandomVerticalFlip(p=0.5),  # Flip verticale con probabilità 50%
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    image = Image.open(image_path).convert('RGB')
    return transform(image)


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, labels_dict, base_dir, num_classes, target_size=(224, 224), augment=False):
        self.labels_dict = labels_dict
        self.base_dir = base_dir
        self.target_size = target_size
        self.augment = augment
        self.num_classes = num_classes

        self.image_paths = [
            os.path.join(base_dir, img) for img in os.listdir(base_dir)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Ordina le immagini per garantire consistenza
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Caricamento immagine
        image_path = self.image_paths[idx]

        # Estrai l'ID dell'immagine dal nome file (es. train_059371_<trasformazione>.jpg -> 059371)
        image_name = os.path.basename(image_path)
        image_id = image_name.split('_')[1]

        # Ottieni l'etichetta corrispondente dall'ID
        label = self.labels_dict[image_id]

        # Carica e preprocessa l'immagine
        image = preprocess_image(image_path, self.target_size, self.augment)

        # Codifica l'etichetta in one-hot encoding
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1

        return image, one_hot_label
