import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd

# Carica il modello ResNet50 pre-addestrato
model = models.resnet50(pretrained=True)
model.eval()  # Mette il modello in modalità di valutazione

# Funzione per caricare e preparare l'immagine
def load_and_prepare_image(image_path):
    img = Image.open(image_path)  # Carica l'immagine
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  # Trasformazioni necessarie
    
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Aggiungi la dimensione del batch
    return img_tensor

# Funzione per classificare una serie di immagini
def classify_images_in_directory(directory_path, file_names):
    results = []
    for img_file in file_names:
        #img_path = os.path.join(directory_path, './' + img_file)  # Costruisci il percorso completo
        img_tensor = load_and_prepare_image(directory_path + '/' + img_file)  # Prepara l'immagine
        
        # Previsioni sul modello
        with torch.no_grad():
            outputs = model(img_tensor)
        
        _, predicted_class = torch.max(outputs, 1)  # Trova la classe con il punteggio più alto
        
        results.append(predicted_class.item())
        # Stampa il risultato per questa immagine
        print(f"Image: {img_file} - Predected class ID: {predicted_class.item()}")

# Percorso della directory contenente le immagini
directory_path = "../dataset/small_dataset"  # Sostituisci con il percorso della tua cartella

data_train_labelled = pd.read_csv("../dataset/train_small.csv", header=None)
#print(list(data_train_labelled[0]))
# Classifica le immagini
classify_images_in_directory(directory_path, list(data_train_labelled[0]))
