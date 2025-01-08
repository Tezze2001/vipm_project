import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
from torchvision.models import resnet50

# Percorsi dei file
csv_tr = './dataset/train_small.csv'
csv_te = './dataset/val_info.csv'
indir_tr = './dataset/train_set'
indir_te = './dataset/val_set'

# Caricamento dei dati dal CSV (Training)
data_tr = pd.read_csv(csv_tr, header=None, names=['image', 'label'])
image_names_tr = data_tr['image'].tolist()
labels_tr = data_tr['label'].values

# Caricamento dei dati dal CSV (Validation)
data_te = pd.read_csv(csv_te, header=None, names=['image', 'label'])
image_names_te = data_te['image'].tolist()
labels_te = data_te['label'].values


# Funzione per preprocessare un'immagine
def preprocess_image(image_path, target_size):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


# Estrazione delle feature
num_tr = len(image_names_tr)
num_te = len(image_names_te)

# Carica il modello ResNet50 pre-addestrato
base_model = resnet50(weights="IMAGENET1K_V1")  # Cambia il peso se necessario
layer_name = 'avgpool'
base_model.eval()


# Estrai fino al layer desiderato ("avgpool")
class FeatureExtractor(nn.Module):
    def __init__(self, model, target_layer):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])  # Fino a "layer4"
        self.pool = model.avgpool  # Livello "avgpool"

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)  # Appiattisci le feature


model = FeatureExtractor(base_model, layer_name)

# Dimensione input del modello
input_size = (224, 224)


# Funzione per estrarre le feature dal modello (identica al tuo codice precedente)
def extract_features(model, image_names, indir, target_size, device):
    model = model.to(device)
    features = []
    for i, image_name in enumerate(image_names):
        print(f"Elaborazione immagine: {i + 1}/{len(image_names)}")
        image_path = os.path.join(indir, image_name)
        img = preprocess_image(image_path, target_size).to(device)
        with torch.no_grad():
            feature = model(img)
        features.append(feature.cpu().numpy().squeeze())
    return np.array(features)


# Estrazione delle feature
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Estrazione delle feature...")

# Training set
feat_tr = extract_features(model, image_names_tr, indir_tr, input_size, device)

# Test set
feat_te = extract_features(model, image_names_te, indir_te, input_size, device)

# Normalizzazione delle feature
feat_tr /= np.linalg.norm(feat_tr, axis=1, keepdims=True)
feat_te /= np.linalg.norm(feat_te, axis=1, keepdims=True)

# Salvataggio delle feature
np.savez('features_cibo_avgpool_resnet50.npz', feat_tr=feat_tr, labels_tr=labels_tr, feat_te=feat_te,
         labels_te=labels_te)

# Classificazione con 1-NN
feat_tr = feat_tr.reshape(feat_tr.shape[0], -1)  # Appiattisci le feature del training set
feat_te = feat_te.reshape(feat_te.shape[0], -1)

D = cdist(feat_te, feat_tr, metric='euclidean')
idx_pred_te = np.argmin(D, axis=1)
lab_pred_te = labels_tr[idx_pred_te]

# Accuratezza globale
accuracy = np.mean(lab_pred_te == labels_te)
print(f"Accuratezza globale: {accuracy * 100:.2f}%")

# Matrice di confusione
conf_matrix = confusion_matrix(labels_te, lab_pred_te)

# Calcolo accuratezza per classe
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
for i, acc in enumerate(class_accuracies):
    print(f"Classe {i}: Accuratezza = {acc * 100:.2f}%")

# Visualizzazione matrice di confusione
plt.figure(figsize=(16, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu",
            cbar_kws={'label': 'Numero di Campioni'}, annot_kws={"size": 10})
plt.title(f"Accuratezza Globale: {accuracy * 100:.2f}% - Layer: {layer_name}")
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.savefig('confusion_matrix_resnet50.png')
# plt.show()
