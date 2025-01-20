import numpy as np
import torch
import torch.nn as nn

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