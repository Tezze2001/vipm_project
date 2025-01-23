import os
import numpy as np
import pandas as pd
from vipm_features import RGBMeanFeatureExtractor, LBPFeatureExtractor, LABFeatureExtractor, ResNet50FeatureExtractor
from vipm_dataset_cleaner import DatasetCleaner  # Assumendo che il cleaner sia salvato in un modulo separato

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

# Lista per raccogliere tutte le feature estratte
all_features = []

# Itera sugli estrattori di feature
for extractor in extractors:
    print(f"\nTesting extractor: {extractor.name}")

    try:
        features, _, _ = extractor.get_features(csv=csv_path, indir=indir, outdir=outdir, normalize=True)
        print(f"Estratte {features.shape[0]} feature di dimensione {features.shape[1]} con {extractor.name}")
        print(f"Prime 5 feature: {features[:5]}")
        all_features.append(features)
    except Exception as e:
        print(f"Errore durante l'elaborazione con {extractor.name}: {e}")

# Combina tutte le feature in un unico array
combined_features = np.concatenate(all_features, axis=1)
print(f"\nDimensione totale delle feature combinate: {combined_features.shape}")

# Test finale con il DatasetCleaner
print("\nAvvio del test finale con il cleaner...")

# Crea un'istanza del cleaner
cleaner = DatasetCleaner(features=combined_features, class_info=labels, clean_criterion=DatasetCleaner.clean_criterion_isolation_forest)

# Pulisci il dataset globalmente
accepted_indices_global = cleaner.clean_dataset_global(contamination=0.1)
print(f"\nImmagini accettate globalmente: {len(accepted_indices_global)} su {len(labels)}")
print(f"Indici accettati: {accepted_indices_global}")
print(f"Rifuati: {list(set(range(len(labels))) - set(accepted_indices_global))}")

print("\nTesting del cleaner completato.")
