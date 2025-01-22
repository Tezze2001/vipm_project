import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50

class ImageFeatureExtractor:
    def __init__(self, model, target_layer, input_size=(224, 224), device=None):
        self.model = self._create_feature_extractor(model, target_layer)
        self.input_size = input_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _create_feature_extractor(self, model, target_layer):
        """ Crea un estrattore di feature fino al livello desiderato. """
        layers = dict(model.named_children())
        if target_layer not in layers:
            raise ValueError(f"Il layer {target_layer} non esiste nel modello. I layer disponibili sono: {list(layers.keys())}")

        feature_layers = []
        for name, layer in model.named_children():
            feature_layers.append(layer)
            if name == target_layer:
                break

        return nn.Sequential(*feature_layers)

    def preprocess_image(self, image_path):
        """ Preprocessa un'immagine. """
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    def extract_features(self, image_names, indir, n = None):
        """ Estrae le feature per tutte le immagini nella lista. """
        features = []

        
        if n is not None:
            final_image_names = image_names[:n]
        else:
            final_image_names = image_names

        for i, image_name in enumerate(final_image_names):
            print(f"Elaborazione immagine {i + 1}/{len(image_names)}: {image_name}")
            image_path = os.path.join(indir, image_name)
            img = self.preprocess_image(image_path).to(self.device)
            with torch.no_grad():
                feature = self.model(img)
            features.append(feature.cpu().numpy().squeeze())
        return np.array(features)

    def normalize_features(self, features):
        """ Normalizza le feature. """
        return features / np.linalg.norm(features, axis=1, keepdims=True)

    def get_normalized_features(self, image_names, indir, n=None):
        features = self.extract_features(image_names, indir, n=n)
        return self.normalize_features(features)
    

class ResNet50FeatureExtractor:
    def __init__(self, path_file_names, path_directory, weights = "IMAGENET1K_V1"):
        self.path_file_names = path_file_names
        self.path_directory = path_directory

        self.model = resnet50(weights=weights)
        self.layer_name = 'avgpool'
        self.model.eval()
        
    def extract_features(self, save_path, normalized=True, n = None):
        feature_extractor = ImageFeatureExtractor(self.model, target_layer=self.layer_name)
        
        data = pd.read_csv(self.path_file_names, header=None, names=['image_name', 'label'])

        image_names = data['image_name'].tolist()
        labels = data['label'].values
        
        if normalized:
            features = feature_extractor.get_normalized_features(image_names,  self.path_directory, n)
        else:
            features = feature_extractor.extract_features(image_names, self.path_directory, n)
        
        np.savez(save_path, X=features, y=labels)



