import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import cdist

class DatasetCleaner:
    def __init__(self, features, class_info, clean_criterion):
        """
        Inizializza il cleaner con il file CSV, la cartella dei dati e la cartella di output.
        
        Args:
            csv_path (str): Percorso al file CSV contenente i dati.
            data_folder (str): Cartella contenente i file da analizzare.
            output_folder (str): Cartella dove salvare i dati puliti e rifiutati.
            feature_extractor (callable): Funzione opzionale per estrarre le feature dai file.
        """
        self.features = features
        self.class_info = class_info
        self.clean_criterion = clean_criterion
        
    def clean_dataset_global(self, **kwargs):
        """
        Pulisce il dataset utilizzando il modello personalizzato.
        
        Args:
            **kwargs: Parametri extra per il modello personalizzato.
        """

        # Usa il modello personalizzato per determinare cosa è sporco
        predictions = self.clean_criterion(self.features, **kwargs)

        accepted_indices = [index for index, is_accepted in enumerate(predictions) if is_accepted]

        return accepted_indices
    

    def clean_dataset_by_class(self, **kwargs):
        """
        Pulisce il dataset separatamente per ogni classe.
        
        Args:
            **kwargs: Parametri extra per il modello personalizzato.
        """
        # Dizionario per raggruppare le feature per classe
        class_groups = {}
        for idx, class_label in enumerate(self.class_info):
            if class_label not in class_groups:
                class_groups[class_label] = []
            class_groups[class_label].append(idx)

        # Risultati finali (nell'ordine originale)
        accepted_indices = []

        # Applica il classificatore per ogni classe
        for class_label, indices in class_groups.items():
            # Estrai le feature corrispondenti a questa classe
            class_features = [self.features[i] for i in indices]
            
            # Usa il modello personalizzato per classificare
            predictions = self.clean_criterion(class_features, **kwargs)
            
            # Aggiungi gli indici accettati nell'ordine originale
            accepted_indices.extend([indices[i] for i, is_accepted in enumerate(predictions) if is_accepted])

        # Ordina gli indici accettati per ripristinare l'ordine originale
        accepted_indices.sort()

        return accepted_indices


    def summarize_results(self, label=""):
        """
        Mostra un riassunto dei risultati, inclusi i file accettati e rifiutati
        per la prima classe come feedback visivo, in un'unica immagine con Matplotlib.
        """
        total_files = len(self.df)
        rejected_files = self.df["Rejected"].sum()
        clean_files = total_files - rejected_files

        print(f"Totale file processati: {total_files}")
        print(f"File accettati: {clean_files}")
        print(f"File rifiutati: {rejected_files}")

        for n in range(10):
            # Visualizza un feedback visivo per la prima classe
            class_df = self.df[self.df["Label"] == n]
            accepted_files = class_df[class_df["Rejected"] == 0]["File"]
            rejected_files = class_df[class_df["Rejected"] == 1]["File"]

            # Organizza le immagini in una griglia
            fig, axes = plt.subplots(2, max(len(accepted_files), len(rejected_files)), figsize=(15, 6))
            fig.suptitle(f"Classe: {n} - Accettati e Rifiutati - {label}", fontsize=16)
        
            # Mostra immagini accettate
            for i, file in enumerate(accepted_files):
                file_path = os.path.join(self.clean_data_folder, file)
                try:
                    img = Image.open(file_path)
                    axes[0, i].imshow(img)
                    axes[0, i].axis("off")
                    axes[0, i].set_title("Accepted")
                except Exception as e:
                    print(f"Errore nel caricamento dell'immagine {file}: {e}")

            # Mostra immagini rifiutate
            for i, file in enumerate(rejected_files):
                file_path = os.path.join(self.bad_data_folder, file)
                try:
                    img = Image.open(file_path)
                    axes[1, i].imshow(img)
                    axes[1, i].axis("off")
                    axes[1, i].set_title("Rejected")
                except Exception as e:
                    print(f"Errore nel caricamento dell'immagine {file}: {e}")

            # Rimuovi assi vuoti
            for ax_row in axes:
                for ax in ax_row:
                    if not ax.has_data():
                        ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            # Salva immagine in cartella output
            plt.savefig(os.path.join(self.output_folder, f"summary_{n}.png"))
            plt.show()
        return total_files, clean_files, rejected_files

    ### CLEAN CRITERIA
    
    # Funzione per pulire i dati con Isolation Forest
    def clean_criterion_isolation_forest(features, contamination=0.1):
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(features)
        return model.predict(features) != -1  # Ritorna False per gli outlier
    

    def clean_criterion_histogram_distance(features, contamination=0.1, metric='cosine'):
        """
        Cleans dataset based on histogram distances from the mean representation.
        """
        if len(features) == 0:
            return []

        # Calculate distances from mean representation
        mean_representation = np.mean(features, axis=0)
        distances = cdist(features, [mean_representation], metric=metric).flatten()
        
        # Determine threshold based on contamination
        threshold = np.percentile(distances, (1 - contamination) * 100)
        
        return distances > threshold