import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import os
import sys
import numpy as np

project_path = os.path.abspath("../../code")
sys.path.append(project_path)
from vipm_features import ResNet50FeatureExtractor

project_path = os.path.abspath("../../networks")
sys.path.append(project_path)
from models import *
from vipm_features import *
from vipm_pipeline import *
from dataset import *
import torch

LOKY_MAX_CPU_COUNT = 16

outdir = '../features' 
os.makedirs(outdir, exist_ok=True)

def load_file_as_list(file_path):
    try:
        return [line.strip() for line in open(file_path, 'r', encoding='utf-8')]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

class_labels = load_file_as_list("class_list.txt")
    
# Carica le immagini dal CSV
extractor = ResNet50FeatureExtractor()

one_layer_model = ClassifierNetwork(2048, 251)
one_layer_optimizer = torch.optim.Adam(one_layer_model.parameters(), lr=0.01)
one_layer_scheduler = torch.optim.lr_scheduler.StepLR(one_layer_optimizer, step_size=5, gamma=0.1)
one_layer_model_option = ModelOptions(torch.nn.CrossEntropyLoss(), one_layer_optimizer, one_layer_scheduler, input_dim = 2048)
nn = NeuralNetwork(one_layer_model, one_layer_model_option)
# load weights
nn.model.load_state_dict(torch.load("classifier.pth", weights_only=True))

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    image = Image.open(file_path).convert("RGB")
    
    # Cut square center
    cut_square = min(image.size)
    left = (image.size[0] - cut_square) / 2
    top = (image.size[1] - cut_square) / 2
    right = (image.size[0] + cut_square) / 2
    bottom = (image.size[1] + cut_square) / 2
    image = image.crop((left, top, right, bottom))
    
    # Resize
    image = image.resize((224, 224))
    
    img_tk = ImageTk.PhotoImage(image)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    
    classify_image(image)

def classify_image(file_path):
    features =  extractor.get_features_single_image(file_path)
    one_layer_model.eval()
    features = torch.FloatTensor(features)
    # add 1 dimension for batch
    features = features.unsqueeze(0)
    # to device
    features = features.to("cuda")
    with torch.no_grad():
        outputs = one_layer_model(features)
        top_probs, top_indices = torch.topk(outputs, k=10, dim=-1)
        sum_vals = top_probs.sum(dim=-1, keepdim=True)

        # Dividi per la somma per ottenere una proporzione
        top_probs = top_probs / sum_vals  
            
    top_probs = top_probs.squeeze().cpu().numpy()
    top_probs = np.maximum(top_probs, 0)
    top_indices = top_indices.squeeze().cpu().numpy()
    top_classes = [class_labels[i] for i in top_indices]
       
    # Show top 10
    result = ""
    for i in range(10):
        result += f"{top_classes[i]}: {top_probs[i]:.2f}\n"
    result_label.config(text=result)
    
# Create main window
root = tk.Tk()
root.title("Image Classifier")
root.geometry("400x500")

# UI Elements
upload_btn = Button(root, text="Upload Image", command=load_image)
upload_btn.pack(pady=10)

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="Class: ?, Probability: ?")
result_label.pack(pady=20)

# Run app
root.mainloop()
