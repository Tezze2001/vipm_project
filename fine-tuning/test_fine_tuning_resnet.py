import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.models import ResNet50_Weights
from dataset import CustomDataset
from torchvision import transforms
from utility import test_validate, train, validate

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

custom_transforms = [
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.2)  # Rumore gaussiano
]

num_classes = 251
batch_size = 32
learning_rate = 1e-3
epochs = 10

# Creazione dei dataset
train_dataset = CustomDataset(image_names_tr, labels_tr, indir_tr, num_classes=num_classes, augment=True)
test_dataset = CustomDataset(image_names_te, labels_te, indir_te, num_classes=num_classes, augment=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suddivisione in training e validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Caricamento modello ResNet50
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Modifica dell'ultimo layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, num_classes),
)
model = model.to(device)

# Loss e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Early Stopping
best_val_loss = float('inf')
early_stop_counter = 0
curr_epoch = 0
patience = 10  # Numero massimo di epoche senza miglioramenti per fermare l'addetramento

# Training e validazione
# Addestramento con monitoraggio della loss e dell'accuratezza
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy, y_pred, y_true = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    curr_epoch += 1
    print(f"Epoch {epoch + 1}/{epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        best_model_state = model.state_dict()
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("\nEarly stopping triggered. Stopping training.")
        break

    # Scheduler step
    scheduler.step()

# Grafico della loss e dell'accuratezza
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, curr_epoch + 1), train_losses, label='Train Loss')
plt.plot(range(1, curr_epoch + 1), val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Andamento della Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, curr_epoch + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, curr_epoch + 1), val_accuracies, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Andamento dell\'Accuratezza')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_metrics_fine_tuning.png')

# Fase di test
test_loss, test_accuracy_top1, test_accuracy_top5, y_test_pred, y_test_true = test_validate(model, test_loader, criterion, device)

print("\nRisultati sul Test Set:")
print(f"  Test Loss: {test_loss:.4f}, Test Accuracy top1: {test_accuracy_top1 * 100:.2f}%, Test Accuracy top5: {test_accuracy_top5 * 100:.2f}%")

# Matrice di confusione per il test set
y_test_pred_classes = y_test_pred
y_test_true_classes = np.argmax(y_test_true, axis=1)
conf_matrix = confusion_matrix(y_test_true_classes, y_test_pred_classes)

# Visualizzazione matrice di confusione
plt.figure(figsize=(16, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu",
            cbar_kws={'label': 'Numero di Campioni'}, annot_kws={"size": 10})
plt.title(f"Matrice di Confusione - Accuratezza Test: {test_accuracy_top1 * 100:.2f}%")
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.savefig('confusion_matrix_test_resnet50_nn_pytorch.png')
# plt.show()

# Accuratezza per classe per il test set
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
print("\nAccuratezza per Classe sul Test Set:")
for i, acc in enumerate(class_accuracies):
    print(f"  Classe {i}: Accuratezza = {acc * 100:.2f}%")
