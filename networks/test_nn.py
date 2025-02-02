import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from dataset import FeatureDataset
from models import *
from utility import train, validate, test_validate

matplotlib.use('TkAgg')

"""
feature_ex = np.load('dataset/features_extended_10.npz', 'r')
X_train = feature_ex['X_train']
y_train = feature_ex['y_train']
X_val = feature_ex['X_val']
y_val = feature_ex['y_val']
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# Parametri
input_dim = X_train.shape[1]  # Dimensione dell'input
num_classes = len(np.unique(y_train))  # Numero di classi
batch_size = 32
epochs = 100
learning_rate = 0.01

# Conversione in tensori
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# one-hot encoding
y_train = F.one_hot(y_train, num_classes=251)
y_val = F.one_hot(y_val, num_classes=251)
y_train = y_train.float()
y_val = y_val.float()

# Dataset e DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
"""

# Parametri
input_dim = 2048  # Dimensione dell'input
num_classes = 251  # Numero di classi
batch_size = 2048  
epochs = 100
learning_rate = 0.01

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

training_set = FeatureDataset('../dataset/features_extended_20.npz',
                              type='train',
                              target_transform=lambda y: F.one_hot(y, num_classes=num_classes))

validation_set = FeatureDataset('../dataset/features_extended_20.npz',
                                type='val',
                                target_transform=lambda y: F.one_hot(y, num_classes=num_classes))

test_set = FeatureDataset('../dataset/test_features_resnet50.npz',
                          type='test',
                          target_transform=lambda y: F.one_hot(y, num_classes=num_classes))

test_set_degraded = FeatureDataset('../dataset/test_degraded_features_resnet50.npz',
                                type='test',
                                target_transform=lambda y: F.one_hot(y, num_classes=num_classes))

train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
test_degraded_loader = DataLoader(test_set_degraded, batch_size=batch_size, shuffle=False)

# Inizializza il modello, la loss e l'optimizer
model = ClassifierNetwork(training_set.get_dim(), training_set.get_n_classes()).to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Decrementa lr ogni 5 epoche

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
plt.savefig('training_validation_metrics.png')

# Fase di test pulito
test_loss, test_accuracy_top1, test_accuracy_top5, y_test_pred, y_test_true = test_validate(model, test_loader, criterion, device)

print("\nRisultati sul Test Set pulito:")
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


# Fase di test su immagini degradate
test_loss, test_accuracy_top1, test_accuracy_top5, y_test_pred, y_test_true = test_validate(model, test_degraded_loader, criterion, device)

print("\nRisultati sul Test Set degradato:")
print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy_top1 * 100:.2f}%, Test Accuracy top5: {test_accuracy_top5 * 100:.2f}%")

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
