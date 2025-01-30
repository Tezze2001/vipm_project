import os
import numpy as np
import pandas as pd
from vipm_features import RGBMeanFeatureExtractor, LBPFeatureExtractor, LABFeatureExtractor, ResNet50FeatureExtractor
from vipm_dataset_cleaner import DatasetCleaner 
from vipm_image_retrieval import ImageRetrievalKNN, ImageRetrievalBestFit

# Carica il file CSV
def load_csv(csv_path):
    data = pd.read_csv(csv_path, header=None, names=['image_name', 'label'])
    return data['image_name'].tolist(), data['label'].tolist()

# Percorsi
csv_path = '../dataset/train_small.csv'   
csv_unlabeled = '../dataset/train_unlabeled.csv'
indir = '../dataset/train_set' 
outdir = '../features' 
os.makedirs(outdir, exist_ok=True)


# Carica le immagini dal CSV
image_names, labels = load_csv(csv_path)
image_names_unlabeled, _ = load_csv(csv_unlabeled)

extractor = ResNet50FeatureExtractor()
features_unlabeled, _, _ = extractor.get_features(csv=csv_unlabeled, indir=indir, outdir=outdir, normalize=True)
features_small, _, _ = extractor.get_features(csv=csv_path, indir=indir, outdir=outdir, normalize=True)


percent_20_random_features_unlabeled = np.random.choice(features_unlabeled.shape[0], int(features_unlabeled.shape[0]*0.2), replace=False)
retrival = ImageRetrievalKNN(dataset=features_unlabeled[percent_20_random_features_unlabeled], queryset=features_small, queryset_labels=labels)
indices, predictions = retrival.retrieve_images()
print(indices, predictions)

retrival_best_fit = ImageRetrievalBestFit(dataset=features_unlabeled[percent_20_random_features_unlabeled], queryset=features_small, queryset_labels=labels, n_neighbors=5)
indices_best_fit, predictions_best_fit = retrival_best_fit.retrieve_images()
print(indices_best_fit, predictions_best_fit)


