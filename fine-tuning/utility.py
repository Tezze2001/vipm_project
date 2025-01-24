import numpy as np
import torch
import torch.nn as nn
import os

# Funzione di addestramento
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
        optimizer.zero_grad()
        outputs = model(X_batch)
            
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total += y_batch.size(0)
        correct += (predicted == y_batch.argmax(dim=1)).sum().item()

    accuracy = correct / total
    return running_loss / len(train_loader), accuracy


# Funzione di validazione
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch.argmax(dim=1)).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = correct / total
    return running_loss / len(val_loader), accuracy, np.array(all_preds), np.array(all_labels)

def test_validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            running_loss += loss.item()

            # Top-1 predictions
            predicted_top1 = outputs.argmax(dim=1)

            # Top-5 predictions
            _, predicted_top5 = outputs.topk(5, dim=1)

            total += y_batch.size(0)
            correct_top1 += (predicted_top1 == y_batch.argmax(dim=1)).sum().item()

            # Check if true labels are in the top-5 predictions
            true_labels = y_batch.argmax(dim=1, keepdim=True)
            correct_top5 += (predicted_top5 == true_labels).sum().item()

            all_preds.extend(predicted_top1.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    return running_loss / len(val_loader), top1_accuracy, top5_accuracy, np.array(all_preds), np.array(all_labels)


# Funzione per creare il labels_dict da un DataFrame
def create_labels_dict(data):
    labels_dict = {}
    for _, row in data.iterrows():
        # Estrai l'id dal nome dell'immagine (esempio: train_059371_noisy.jpg -> 059371)
        image_name = os.path.splitext(row['image'])[0]
        image_id = image_name.split('_')[1]  # Prendi la seconda parte del nome diviso da "_"
        labels_dict[image_id] = row['label']
    return labels_dict
