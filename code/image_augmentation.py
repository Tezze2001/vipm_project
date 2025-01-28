import os
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import random
import pandas as pd


# Funzione per creare directory se non esistono
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Funzione per applicare trasformazioni e salvare immagini
def apply_transformations_and_save(csv_path, input_dir, output_dir, sample_percentage=100, seed=None):
    # Imposta il seed per la riproducibilità
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Carica il CSV con nomi delle immagini e label
    data = pd.read_csv(csv_path, header=None, names=['image', 'label'])

    # Seleziona un campione casuale basato sulla percentuale specificata per le immagini da sporcare
    if sample_percentage < 100:
        sampled_data = data.sample(frac=sample_percentage / 100, random_state=seed)
    else:
        sampled_data = data

    # Trasformazione per aggiungere rumore gaussiano
    transform_noise = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),  # Rumore gaussiano con std=0.1
        transforms.ToPILImage()
    ])

    # Crea directory di output
    create_directory(output_dir)

    # Scansiona tutte le immagini specificate nel CSV
    for _, row in data.iterrows():
        filename = row['image']
        input_path = os.path.join(input_dir, filename)

        if not os.path.exists(input_path):
            print(f"Immagine non trovata: {input_path}, saltata.")
            continue

        # Carica l'immagine
        image = Image.open(input_path).convert("RGB")
    
        # Salva sempre l'immagine originale
        base_filename, ext = os.path.splitext(filename)
        original_path = os.path.join(output_dir, f"{base_filename}_original{ext}")
        image.save(original_path)

        # Se l'immagine è nel campione, applica le trasformazioni
        if filename in sampled_data['image'].values:
            transformations = []
            if random.random() > 0.70:  
                noisy_image = transform_noise(image)
                transformations.append(("noisy", noisy_image))
            if random.random() > 0.70: 
                bilateral_size = random.choice([3, 5, 7])  # Scelta casuale della dimensione del filtro
                bilateral_filtered_image = image.filter(ImageFilter.ModeFilter(size=bilateral_size))
                transformations.append(("bilateral", bilateral_filtered_image))
            if random.random() > 0.70: 
                gaussian_radius = random.uniform(1, 3)  # Scelta casuale del raggio (tra 1 e 3)
                gaussian_filtered_image = image.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
                transformations.append(("gaussian", gaussian_filtered_image))

            # Salva solo le immagini con trasformazioni applicate
            for name, transformed_image in transformations:
                transformed_path = os.path.join(output_dir, f"{base_filename}_{name}{ext}")
                transformed_image.save(transformed_path)


# Percorsi delle directory
input_dir = "dataset//train_set"
csv_tr = './dataset/train_small.csv'
output_dir = "dataset/augmented_train_set_test2"

# Crea la directory di output
create_directory(output_dir)

# Applica le trasformazioni e salva le immagini con seed
apply_transformations_and_save(csv_tr, input_dir, output_dir, sample_percentage=50, seed=42)

print("Trasformazioni completate e immagini salvate.")
