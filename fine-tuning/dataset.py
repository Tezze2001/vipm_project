import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms


# Funzione per preprocessare un'immagine con supporto per trasformazioni personalizzate
def preprocess_image(image_path, target_size, transform_list):
    image = Image.open(image_path).convert('RGB')
    transformed_images = []
    for transform in transform_list:
        transformed_images.append(transform(image))
    return transformed_images


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_names, labels, base_dir, num_classes, target_size=(224, 224), augment=False, custom_transforms=None):
        self.image_names = image_names
        self.labels = labels
        self.base_dir = base_dir
        self.target_size = target_size
        self.augment = augment
        self.num_classes = num_classes
        self.custom_transforms = custom_transforms or []

        # Trasformazioni di default
        self.default_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),  # Flip orizzontale
                transforms.RandomRotation(degrees=15),  # Rotazione casuale
                transforms.RandomVerticalFlip(p=0.5),  # Flip verticale
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)  # Rumore gaussiano
            ])
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Caricamento immagine
        image_path = f"{self.base_dir}/{self.image_names[idx]}"

        # Crea una lista di trasformazioni da applicare
        transform_list = [self.default_transform]
        if self.augment_transform:
            transform_list.append(self.augment_transform)
        transform_list.extend(self.custom_transforms)  # Aggiungi trasformazioni personalizzate

        # Preprocessa l'immagine con tutte le trasformazioni
        transformed_images = preprocess_image(image_path, self.target_size, transform_list)

        # Codifica dell'etichetta in one-hot encoding
        label = self.labels[idx]
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1

        # Restituisci tutte le immagini trasformate con la stessa etichetta
        return transformed_images, [one_hot_label] * len(transformed_images)
