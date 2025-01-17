import os
import numpy as np
import pandas as pd
from torchvision.models import ResNet50_Weights
from vipm_features import RGBMeanFeatureExtractor, LBPFeatureExtractor, LABFeatureExtractor, ResNet50FeatureExtractor
# Carica il file CSV
def load_csv(csv_path):
    data = pd.read_csv(csv_path, header=None, names=['image_name', 'label'])
    return data['image_name'].tolist(), data['label'].tolist()

# Percorsi
csv_path = '../dataset/train_small.csv'   
indir = '../dataset/train_set'  # Modifica in base alla posizione delle immagini
outdir = '../features'  # Modifica in base alla posizione delle feature
os.makedirs(outdir, exist_ok=True)

# Inizializza gli estrattori di feature
extractors = [
    RGBMeanFeatureExtractor(),
    LBPFeatureExtractor(radius=3, n_points=8),
    LABFeatureExtractor(bins=32),
    ResNet50FeatureExtractor()
]

# Carica le immagini dal CSV
image_names, labels = load_csv(csv_path)

# Itera sugli estrattori di feature
for extractor in extractors:
    print(f"\nTesting extractor: {extractor.name}")

    try:
        features = extractor.get_features(csv=csv_path, indir=indir, outdir=outdir, normalize=True)
        print(f"Estratte {features.shape[0]} feature di dimensione {features.shape[1]} con {extractor.name}")
        print(f"Prime 5 feature: {features[:5]})")
    except Exception as e:
        print(f"Errore durante l'elaborazione con {extractor.name}: {e}")

print("\nTesting completato.")
