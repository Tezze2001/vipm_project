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
from scipy.stats import mode
from torchvision.models import resnet50
from sklearn.model_selection import StratifiedKFold
# from torchsummary import torchsummary

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

# torchsummary.summary(base_model, (3, 224, 224))


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

# Configurazione per 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Variabili per calcolare l'accuratezza globale media e per classe
global_accuracies = []
class_accuracies_list = []
layer_name = "resnet_layer_name"  # Nome del layer per la visualizzazione

for fold, (train_idx, val_idx) in enumerate(skf.split(feat_tr, labels_tr)):
    print(f"Fold {fold + 1}/{n_splits}")
    
    # Suddivisione del training e validation set
    X_train, X_val = feat_tr[train_idx], feat_tr[val_idx]
    y_train, y_val = labels_tr[train_idx], labels_tr[val_idx]
    
    # Appiattisci le feature
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    
    # Calcolo delle distanze tra validation e training
    D = cdist(X_val, X_train, metric='euclidean')
    
    # Classificazione con k-NN
    k = 5
    k_nearest_indices = np.argpartition(D, k, axis=1)[:, :k]
    k_nearest_labels = y_train[k_nearest_indices]
    lab_pred_val = mode(k_nearest_labels, axis=1)[0].flatten()
    
    # Accuratezza globale per il fold
    accuracy = np.mean(lab_pred_val == y_val)
    global_accuracies.append(accuracy)
    print(f"Accuratezza fold {fold + 1}: {accuracy * 100:.2f}%")
    
    # Matrice di confusione per il fold
    conf_matrix = confusion_matrix(y_val, lab_pred_val)
    
    # Accuratezza per classe
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    class_accuracies_list.append(class_accuracies)
    for i, acc in enumerate(class_accuracies):
        print(f"  Classe {i}: Accuratezza = {acc * 100:.2f}%")

# Accuratezza media globale
mean_global_accuracy = np.mean(global_accuracies)
print(f"\nAccuratezza media globale: {mean_global_accuracy * 100:.2f}%")

# Accuratezza media per classe
mean_class_accuracies = np.mean(class_accuracies_list, axis=0)
for i, acc in enumerate(mean_class_accuracies):
    print(f"Classe {i}: Accuratezza media = {acc * 100:.2f}%")

# Visualizzazione matrice di confusione dell'ultimo fold
plt.figure(figsize=(16, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu",
            cbar_kws={'label': 'Numero di Campioni'}, annot_kws={"size": 10})
plt.title(f"Accuratezza Globale Ultimo Fold: {accuracy * 100:.2f}% - Layer: {layer_name}")
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.savefig('confusion_matrix_resnet50_crossval.png')
# plt.show()