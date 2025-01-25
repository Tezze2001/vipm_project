import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import torch
import os
import sys

project_path = os.path.abspath("../networks")  # Adatta il percorso a dove si trova il tuo progetto
sys.path.append(project_path)

from models import ModelOptions
from vipm_dataset_cleaner import DatasetCleaner
from vipm_image_retrieval import ImageRetrieval
from vipm_features import BaseFeatureExtractor
from vipm_costants import * 
from abc import ABC

class Transformer(ABC):
    def transform(self, **kwargs):
        pass

class Model(ABC):
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
class FeatureExtractor(Transformer):
    def __init__(self, extractor: BaseFeatureExtractor):
        super().__init__()
        self.extractor = extractor
        self.name = self.extractor.name

    def transform(self, **kwargs):
        return self.extractor.get_features(csv=kwargs['csv'], indir=kwargs['indir'], outdir=kwargs['outdir'], normalize=kwargs['normalize'])
    
class Retrieval(Transformer):
    def __init__(self, retrieval_method: ImageRetrieval):
        super().__init__()
        self.retrieval_method = retrieval_method

    def transform(self, **kwargs):
        return self.extractor.get_features()
    
class Cleaner(Transformer):
    def __init__(self, cleaner_method: DatasetCleaner):
        super().__init__()
        self.cleaner_method = cleaner_method
    
    def transform(self, **kwargs):
        return self.cleaner_method.clean_dataset_global(**kwargs)
    
class Splitter(Transformer):
    def __init__(self, X_small_train, y_small_train, X_train, y_train):
        super().__init__()
        self.X_small_train = X_small_train
        self.y_small_train = y_small_train
        self.X_train = X_train
        self.y_train = y_train
    
    def transform(self, **kwargs):
        if test_size == None:
            test_size = test_size=int((.2 * (self.X_train.shape[0] + self.X_small_train.shape[0]) )/ 251)/20

        if test_size == 1:
            X_train, X_test = self.X_train, self.X_small_train
            y_train, y_test = self.y_train, self.y_small_train
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X_small_train, self.y_small_train, test_size=test_size, stratify=self.y_small_train, random_state=42)
            
            X_train = np.concatenate((X_train, self.X_train), axis=0)  # Combine rows
            y_train = np.concatenate((y_train, self.y_train), axis=0)  # Combine

            
        return X_train, X_test, y_train, y_test
    

class KNN(Model):
    def __init__(self, n_neighbors, standardize = True, weights='uniform'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.standardize = standardize
        if self.standardize:
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),              # Standardization step
                ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights=self.weights)) 
            ])
        else:
            self.pipeline = Pipeline([
                ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights=self.weights)) 
            ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)
    

class NeuralNetwork(Model):
    def __init__(self, model: torch.nn.Module, model_options: ModelOptions, log=True):
        super().__init__()
        self.model = model
        self.model_options = model_options
        self.model.to(self.model_options.device)
        self.log = log 
        self.best_model_state = type(self.model)(input_dim = self.model_options.input_dim, num_classes = self.model_options.num_classes)
        self.best_model_state.to(self.model_options.device)

    def __train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.model_options.device), y_batch.to(self.model_options.device).float()
            self.model_options.optimizer.zero_grad()
            outputs = self.model(X_batch)
                
            loss = self.model_options.criterion(outputs, y_batch)
            loss.backward()
            self.model_options.optimizer.step()

            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch.argmax(dim=1)).sum().item()

        accuracy = correct / total
        return running_loss / len(train_loader), accuracy

    def __evaluate(self, val_loader):
        self.best_model_state.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.model_options.device), y_batch.to(self.model_options.device).float()

                outputs = self.best_model_state(X_batch)

                loss = self.model_options.criterion(outputs, y_batch)
                running_loss += loss.item()

                predicted = outputs.argmax(dim=1)
                total += y_batch.size(0)
                correct += (predicted == y_batch.argmax(dim=1)).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = correct / total
        return running_loss / len(val_loader), accuracy, np.array(all_preds), np.array(all_labels)
    
    def fit(self, train_loader, val_loader):
        best_val_loss = float('inf')
        early_stop_counter = 0
        curr_epoch = 0
        

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        for epoch in range(self.model_options.epochs):
            train_loss, train_accuracy = self.__train(train_loader)
            if val_loader != None:
                val_loss, val_accuracy, y_pred, y_true = self.__evaluate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            curr_epoch += 1
            if self.log:
                print(f"Epoch {epoch + 1}/{self.model_options.epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
                if val_loader != None:
                    print(f"  Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1] * 100:.2f}%")

            # Early stopping
            
            if val_loader != None:
                if val_losses[-1] < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    self.best_model_state.load_state_dict(self.model.state_dict())
                else:
                    early_stop_counter += 1
            else:
                self.best_model_state.load_state_dict(self.model.state_dict())


            if early_stop_counter >= self.model_options.patience:
                print("\nEarly stopping triggered. Stopping training.")
                break

            # Scheduler step
            self.model_options.scheduler.step()

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, test_loader):
        mean_loss, accuracy, y_pred, y_test_hot_encoded = self.__evaluate(test_loader)

        y_test = np.argmax(y_test_hot_encoded, axis=1)

        return mean_loss, accuracy, y_pred, y_test
    

class Pipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def start():
        pass