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
    def __init__(self, csv_path, data_folder, output_folder, features, clean_criterion):
        """
        Inizializza il cleaner con il file CSV, la cartella dei dati e la cartella di output.
        
        Args:
            csv_path (str): Percorso al file CSV contenente i dati.
            data_folder (str): Cartella contenente i file da analizzare.
            output_folder (str): Cartella dove salvare i dati puliti e rifiutati.
            feature_extractor (callable): Funzione opzionale per estrarre le feature dai file.
        """
        self.df = pd.read_csv(csv_path, header=None, names=["File", "Label"])
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.bad_data_folder = os.path.join(output_folder, "rejected")
        self.clean_data_folder = os.path.join(output_folder, "clean")
        self.features = features
        self.clean_criterion = clean_criterion

        # Rimuovi output folder se esiste
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        
        # Crea le cartelle di output
        for folder in [self.output_folder, self.bad_data_folder, self.clean_data_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def clean_dataset_global(self, **kwargs):
        """
        Pulisce il dataset utilizzando il modello personalizzato.
        
        Args:
            **kwargs: Parametri extra per il modello personalizzato.
        """
        feature_list = []
        file_paths = []

        # Usa il modello personalizzato per determinare cosa è sporco
        predictions = self.clean_criterion(self.features, **kwargs)

        # Salva i risultati
        self.df = self.df[self.df["File"].isin(file_paths)]  # Escludi file non processabili
        self.df["Rejected"] = predictions
        self.df.to_csv(os.path.join(self.output_folder, "train_info.csv"), index=False)

        # Copia i file nelle cartelle corrette
        for file_name, is_rejected in zip(file_paths, self.df["Rejected"]):
            src = os.path.join(self.data_folder, file_name)
            dest_folder = self.bad_data_folder if is_rejected else self.clean_data_folder
            dest = os.path.join(dest_folder, file_name)

            shutil.copy(src, dest)

        print("Pulizia completata. Risultati salvati in:", self.output_folder)

    def clean_dataset_by_class(self, **kwargs):
        """
        Pulisce il dataset separatamente per ogni classe.
        
        Args:
            **kwargs: Parametri extra per il modello personalizzato.
        """
        feature_list = []
        file_paths = []

        # Itera su ogni classe
        for label in self.df["Label"].unique():
            
            # Filtro i dati per la classe corrente
            class_df = self.df[self.df["Label"] == label]
            
            class_feature_list = []
            class_file_paths = []

            # Estrai le feature per ogni file della classe
            for index, row in class_df.iterrows():
                file_name = row["File"]
                file_path = os.path.join(self.data_folder, file_name)
                if os.path.exists(file_path):
                    features = self.features(file_path)
                    if features is not None:
                        class_feature_list.append(features)
                        class_file_paths.append(file_name)

            # Converte le feature in array numpy per la classe
            class_feature_array = np.array(class_feature_list)

            # Usa il modello personalizzato per determinare cosa è sporco per la classe corrente
            predictions = self.clean_criterion(class_feature_array, **kwargs)

           # Aggiungi i risultati alla dataframe della classe
           
            self.df.loc[self.df["Label"] == label, "Rejected"] = predictions


            # Copia i file nelle cartelle corrette per la classe
            for file_name in class_file_paths:
                is_rejected = self.df.loc[self.df["File"] == file_name, "Rejected"].values[0]
                src = os.path.join(self.data_folder, file_name)
                dest_folder = self.bad_data_folder if is_rejected else self.clean_data_folder
                dest = os.path.join(dest_folder, file_name)
                shutil.copy(src, dest)
            

        # Salva i risultati complessivi
        self.df.to_csv(os.path.join(self.output_folder, "train_info.csv"), index=False)
        print("Pulizia completata. Risultati salvati in:", self.output_folder)


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
        return model.predict(features) == -1  # Ritorna True per gli outlier
    

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
    
    
    ### EVALUATE CLEANER

    def evaluate_resnet_50(path_input, device=None):

        # Percorsi dei file
        csv_tr = path_input + '/train_info.csv'
        csv_te = './dataset/val_info.csv'
        indir_tr = './dataset/train_set'
        indir_te = './dataset/val_set'
        val_features_file = 'val_features_resnet50.npz'  # Nome file di salvataggio

        # Caricamento dei dati dal CSV (Training)
        data_tr = pd.read_csv(csv_tr, header=None, names=['image', 'label', 'rejected'])
        image_names_tr = data_tr[data_tr['rejected'] == "False"]['image'].tolist()
        labels_tr = data_tr[data_tr['rejected'] == "False"]['label'].values

        # Caricamento dei dati dal CSV (Validation)
        data_te = pd.read_csv(csv_te, header=None, names=['image', 'label'])
        image_names_te = data_te['image'].tolist()
        labels_te = data_te['label'].values

        # Funzione per preprocessare un'immagine
        def preprocess_image(image_path, target_size):
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = Image.open(image_path).convert('RGB')
            return transform(image).unsqueeze(0)

        # Carica il modello ResNet50 pre-addestrato
        base_model = resnet50(weights="IMAGENET1K_V1")  # Cambia il peso se necessario
        layer_name = 'avgpool'
        base_model.eval()

        # Estrai fino al layer desiderato ("avgpool")
        class FeatureExtractor(nn.Module):
            def __init__(self, model, target_layer):
                super(FeatureExtractor, self).__init__()
                self.features = nn.Sequential(*list(model.children())[:-2])  # Fino a "layer4"
                self.pool = model.avgpool  # Livello "avgpool"

            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                return torch.flatten(x, 1)  # Appiattisci le feature

        model = FeatureExtractor(base_model, layer_name)

        # Dimensione input del modello
        input_size = (224, 224)

        # Funzione per estrarre le feature dal modello
        def extract_features(model, image_names, indir, target_size, device):
            model = model.to(device)
            features = []
            for i, image_name in enumerate(image_names):
                # print(f"Elaborazione immagine: {i + 1}/{len(image_names)}")
                image_path = os.path.join(indir, image_name)
                img = preprocess_image(image_path, target_size).to(device)
                with torch.no_grad():
                    feature = model(img)
                features.append(feature.cpu().numpy().squeeze())
            return np.array(features)

        # Estrazione delle feature
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Estrazione delle feature...")

        # Training set
        feat_tr = extract_features(model, image_names_tr, indir_tr, input_size, device)

        # Controlla se il file delle feature di validation esiste
        if os.path.exists(val_features_file):
            print("Caricamento delle feature di validation salvate...")
            data = np.load(val_features_file)
            feat_te = data['feat_te']
            labels_te = data['labels_te']
        else:
            print("Calcolo delle feature di validation...")
            feat_te = extract_features(model, image_names_te, indir_te, input_size, device)

            # Salvataggio delle feature di validation
            print("Salvataggio delle feature di validation...")
            np.savez(val_features_file, feat_te=feat_te, labels_te=labels_te)

        # Normalizzazione delle feature
        feat_tr /= np.linalg.norm(feat_tr, axis=1, keepdims=True)
        feat_te /= np.linalg.norm(feat_te, axis=1, keepdims=True)

        # Classificazione con 1-NN
        feat_tr = feat_tr.reshape(feat_tr.shape[0], -1)  # Appiattisci le feature del training set
        feat_te = feat_te.reshape(feat_te.shape[0], -1)

        # Calcolo delle distanze tra validation e training
        D = cdist(feat_te, feat_tr, metric='euclidean')
        
        # Classificazione con k-NN
        k = 5
        k_nearest_indices = np.argpartition(D, k, axis=1)[:, :k]
        k_nearest_labels = labels_tr[k_nearest_indices]
        k_nearest_labels = k_nearest_labels.astype(int)
        lab_pred_te = mode(k_nearest_labels, axis=1)[0].flatten()
        accuracy_5 = np.mean(lab_pred_te == labels_te)
        
        k = 10
        k_nearest_indices = np.argpartition(D, k, axis=1)[:, :k]
        k_nearest_labels = labels_tr[k_nearest_indices]
        k_nearest_labels = k_nearest_labels.astype(int)
        lab_pred_te = mode(k_nearest_labels, axis=1)[0].flatten()
        accuracy_10 = np.mean(lab_pred_te == labels_te)
        
        k = 50
        k_nearest_indices = np.argpartition(D, k, axis=1)[:, :k]
        k_nearest_labels = labels_tr[k_nearest_indices]
        k_nearest_labels = k_nearest_labels.astype(int)
        lab_pred_te = mode(k_nearest_labels, axis=1)[0].flatten()
        accuracy_50 = np.mean(lab_pred_te == labels_te)
        
        k = 100
        k_nearest_indices = np.argpartition(D, k, axis=1)[:, :k]
        k_nearest_labels = labels_tr[k_nearest_indices]
        k_nearest_labels = k_nearest_labels.astype(int)
        lab_pred_te = mode(k_nearest_labels, axis=1)[0].flatten()
        accuracy_100 = np.mean(lab_pred_te == labels_te)
        
        k = 150
        k_nearest_indices = np.argpartition(D, k, axis=1)[:, :k]
        k_nearest_labels = labels_tr[k_nearest_indices]
        k_nearest_labels = k_nearest_labels.astype(int)
        lab_pred_te = mode(k_nearest_labels, axis=1)[0].flatten()
        accuracy_150 = np.mean(lab_pred_te == labels_te)
        
        return accuracy_5, accuracy_10, accuracy_50, accuracy_100, accuracy_150
