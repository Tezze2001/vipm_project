import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np


# Carica l'immagine originale
image_path = "dataset/val_set/val_000036.jpg"
image = Image.open(image_path).convert("RGB")

# Normalizza e aggiunge rumore gaussiano
transform_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.2)  # Rumore gaussiano con std=0.2
])

# Applica i filtri e la trasformazione
noisy_image = transform_noise(image).clamp(0, 1).permute(1, 2, 0).numpy()  # Rumore gaussiano
bilateral_filtered_image = image.filter(ImageFilter.ModeFilter(size=7))  # Filtro bilaterale approssimato
gaussian_filtered_image = image.filter(ImageFilter.GaussianBlur(radius=2))  # Filtro gaussiano

# Visualizza l'immagine originale e le versioni trasformate
plt.figure(figsize=(15, 8))

# Immagine originale
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

# Immagine con rumore gaussiano
plt.subplot(2, 3, 2)
plt.title("Image with Gaussian Noise")
plt.imshow(noisy_image)
plt.axis("off")

# Immagine con filtro bilaterale
plt.subplot(2, 3, 3)
plt.title("Image with Bilateral Filter")
plt.imshow(bilateral_filtered_image)
plt.axis("off")

# Immagine con filtro gaussiano
plt.subplot(2, 3, 4)
plt.title("Image with Gaussian Blur")
plt.imshow(gaussian_filtered_image)
plt.axis("off")

plt.tight_layout()
plt.show()
