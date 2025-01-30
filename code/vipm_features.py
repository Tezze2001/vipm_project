import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from abc import ABC

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage import color
from skimage.feature import local_binary_pattern
from torchvision import transforms
from torchvision.models import resnet50


class BaseFeatureExtractor(ABC):
    """Classe base astratta per l'estrazione di feature."""

    def __init__(self, is_neural=False, device=None, name="base"):
        self.is_neural = is_neural
        self.name = name
        if is_neural:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess_image(self, image_path):
        """Preprocessa un'immagine. Da implementare nelle sottoclassi."""
        pass

    def _extract_features(self, image):
        """Estrae le feature dall'immagine. Da implementare nelle sottoclassi."""
        pass

    def _normalize_features(self, features):
        """Normalizza le feature."""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        return features / np.linalg.norm(features, axis=1, keepdims=True)

    def _extract_list_features(self, image_names, indir, n=None):
        """Estrae le feature per tutte le immagini nella lista."""
        features = []
        final_image_names = image_names[:n] if n is not None else image_names

        start_time = time.time()
        for i, image_name in enumerate(final_image_names):
            if i % 100 == 0:
                print(f"Elaborazione immagine {i + 1}/{len(final_image_names)} - Tempo rimanente stimato: {(time.time() - start_time) / (i + 1) * (len(final_image_names) - i):.2f} secondi")
            image_path = os.path.join(indir, image_name)

            if self.is_neural:
                img = self._preprocess_image(image_path).to(self.device)
                with torch.no_grad():
                    feature = self.model(img)
                features.append(feature.cpu().numpy().squeeze())
            else:
                image = self._preprocess_image(image_path)
                feature = self._extract_features(image)
                features.append(feature)

        return np.array(features)

    def _get_normalized_features(self, image_names, indir, n=None):
        """Estrae e normalizza le feature."""
        features = self._extract_list_features(image_names, indir, n=n)
        return self._normalize_features(features)

    def get_features(self, csv, indir, outdir, normalize=False, n=None, file_name=None):
        """Salva le feature su disco o le carica se già disponibili."""
        # Nome del file basato sul nome della classe, sul parametro di normalizzazione e sul nome del csv
        if file_name is not None:
            feature_file = os.path.join(outdir, file_name)
        else:
            nome_csv = os.path.basename(csv).split('.')[0]
            feature_file = os.path.join(outdir, f"{nome_csv}_{self.name}_features{'_normalized' if normalize else ''}.npz")

        # Se il file esiste, caricalo
        if os.path.exists(feature_file):
            print(f"Caricamento delle feature da {feature_file}")
            npzfile = np.load(feature_file)
            return npzfile['X'], npzfile['y'], npzfile['image_names']
        else:
            data = pd.read_csv(csv, header=None, names=['image_name', 'label'])
            image_names = data['image_name'].tolist()
            labels = data['label'].values
            
            # Altrimenti, estrai le feature
            print("File non trovato. Estrazione delle feature...")
            features = self._get_normalized_features(image_names, indir, n=n) if normalize else self._extract_list_features(
                image_names, indir, n=n)

            # Salva su disco
            os.makedirs(outdir, exist_ok=True)
            np.savez(feature_file, X=features, y=labels, image_names=image_names)
            print(f"Feature salvate in {feature_file}")
            return features, labels, image_names
        
    def get_features_single_image(self, image):
        """Estrae le feature da una singola immagine."""
        img = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(img)
        return feature.cpu().numpy().squeeze()


class NeuralFeatureExtractor(BaseFeatureExtractor):
    """Implementazione specifica per l'estrazione di feature da reti neurali generiche."""

    def __init__(self, model, target_layer, input_size=(224, 224), device=None, name="neural"):
        super().__init__(is_neural=True, device=device, name=name)
        self.model = self._create_feature_extractor(model, target_layer)
        self.model.to(self.device)
        self.model.eval()
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _create_feature_extractor(self, model, target_layer):
        """Crea un estrattore di feature fino al livello desiderato."""
        layers = dict(model.named_children())
        if target_layer not in layers:
            raise ValueError(f"Il layer {target_layer} non esiste nel modello. "
                             f"I layer disponibili sono: {list(layers.keys())}")

        feature_layers = []
        for name, layer in model.named_children():
            feature_layers.append(layer)
            if name == target_layer:
                break

        return nn.Sequential(*feature_layers)

    def _preprocess_image(self, image_path):
        """Preprocessa l'immagine per l'input alla rete neurale."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)


class ResNet50FeatureExtractor(NeuralFeatureExtractor):
    """Implementazione specifica per l'estrazione di feature da ResNet50."""

    def __init__(self, weights="IMAGENET1K_V1", device=None):
        model = resnet50(weights=weights)
        super().__init__(model=model, target_layer='avgpool', input_size=(224, 224), device=device, name="resnet50")


class TraditionalFeatureExtractor(BaseFeatureExtractor):
    """Implementazione specifica per l'estrazione di feature tradizionali."""

    def __init__(self, name, preprocess_image_func=None):
        super().__init__(is_neural=False, name=name)
        self._preprocess_image_func = preprocess_image_func

    def _preprocess_image(self, image_path):
        """Preprocessa l'immagine per l'estrazione delle feature."""
        image = Image.open(image_path)
        if image is None:
            raise ValueError(f"Immagine {image_path} non valida")
        if self._preprocess_image_func is not None:
            return self._preprocess_image_func(image)
        return image


class RGBMeanFeatureExtractor(TraditionalFeatureExtractor):
    """Feature extractor che calcola i valori medi RGB dalle immagini."""

    def __init__(self, preprocess_image_func=None):
        super().__init__(name="rgb_mean", preprocess_image_func=preprocess_image_func)

    def _extract_features(self, image):
        """Estrae i valori medi RGB dall'immagine."""
        try:
            R, G, B = image.split()
            R_mean = np.mean(np.array(R))
            G_mean = np.mean(np.array(G))
            B_mean = np.mean(np.array(B))
            return np.array([R_mean, G_mean, B_mean])
        except Exception as e:
            print(f"Errore durante l'elaborazione dell'immagine: {e}")
            return None


class LBPFeatureExtractor(TraditionalFeatureExtractor):
    """Feature extractor che usa Local Binary Patterns."""

    def __init__(self, radius=24, n_points=8, preprocess_image_func=None):
        super().__init__(name="lbp", preprocess_image_func=preprocess_image_func)
        self.radius = radius
        self.n_points = n_points

    def _extract_features(self, image):
        """Estrae feature LBP dall'immagine."""
        try:
            image = np.array(image.convert("L"))
            image = (image * 255).astype(np.uint8)

            h, w = image.shape
            h_start = (h - 255) // 2
            w_start = (w - 255) // 2
            image = image[h_start:h_start + 255, w_start:w_start + 255]

            lbp = local_binary_pattern(image, self.n_points, self.radius, method="uniform")

            lbp_hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, self.n_points + 3),
                range=(0, self.n_points + 2)
            )
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-6)

            return lbp_hist
        except Exception as e:
            print(f"Errore nell'estrazione delle feature LBP: {e}")
            return None


class LABFeatureExtractor(TraditionalFeatureExtractor):
    """Feature extractor che usa lo spazio colore LAB."""

    def __init__(self, bins=256, preprocess_image_func=None):
        super().__init__(name="lab", preprocess_image_func=preprocess_image_func)
        self.bins = bins

    def _extract_features(self, image):
        """Estrae feature dallo spazio colore LAB."""
        try:
            lab_image = color.rgb2lab(image)

            l_channel = lab_image[:, :, 0]
            a_channel = lab_image[:, :, 1]
            b_channel = lab_image[:, :, 2]

            l_hist, _ = np.histogram(l_channel.ravel(), bins=self.bins, range=(0, 100))
            a_hist, _ = np.histogram(a_channel.ravel(), bins=self.bins, range=(-128, 127))
            b_hist, _ = np.histogram(b_channel.ravel(), bins=self.bins, range=(-128, 127))

            l_hist = l_hist.astype("float") / (l_hist.sum() + 1e-6)
            a_hist = a_hist.astype("float") / (a_hist.sum() + 1e-6)
            b_hist = b_hist.astype("float") / (b_hist.sum() + 1e-6)

            return np.concatenate([l_hist, a_hist, b_hist])
        except Exception as e:
            print(f"Errore nell'estrazione delle feature LAB: {e}")
            return None
