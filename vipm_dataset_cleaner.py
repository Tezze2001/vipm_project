import os
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from PIL import Image


class DatasetCleaner:
    def __init__(self, csv_path, data_folder, output_folder, feature_extractor, clean_criterion):
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
        self.feature_extractor = feature_extractor
        self.clean_criterion = clean_criterion

        # Rimuovi output folder se esiste
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        
        # Crea le cartelle di output
        for folder in [self.output_folder, self.bad_data_folder, self.clean_data_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def extract_features(self, file_path):
        """
        Estrai feature dai file usando la funzione personalizzata o restituisci None.
        """
        if self.feature_extractor:
            try:
                return self.feature_extractor(file_path)
            except Exception as e:
                print(f"Errore nell'estrazione delle feature per {file_path}: {e}")
                return None
        else:
            print("Nessuna funzione di estrazione delle feature definita.")
            return None

    def clean_dataset_global(self, **kwargs):
        """
        Pulisce il dataset utilizzando il modello personalizzato.
        
        Args:
            **kwargs: Parametri extra per il modello personalizzato.
        """
        feature_list = []
        file_paths = []

        # Estrai le feature per ogni file
        for index, row in self.df.iterrows():
            file_name = row["File"]
            file_path = os.path.join(self.data_folder, file_name)
            if os.path.exists(file_path):
                features = self.extract_features(file_path)
                if features is not None:
                    feature_list.append(features)
                    file_paths.append(file_name)

        # Converte le feature in array numpy
        feature_array = np.array(feature_list)

        # Usa il modello personalizzato per determinare cosa è sporco
        predictions = self.clean_criterion(feature_array, **kwargs)

        # Salva i risultati
        self.df = self.df[self.df["File"].isin(file_paths)]  # Escludi file non processabili
        self.df["Rejected"] = predictions
        self.df.to_csv(os.path.join(self.output_folder, "results.csv"), index=False)

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
                    features = self.extract_features(file_path)
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
            for file_name, is_rejected in zip(class_file_paths, class_df["Rejected"]):
                src = os.path.join(self.data_folder, file_name)
                dest_folder = self.bad_data_folder if is_rejected else self.clean_data_folder
                dest = os.path.join(dest_folder, file_name)
                shutil.copy(src, dest)
            

        # Salva i risultati complessivi
        self.df.to_csv(os.path.join(self.output_folder, "results.csv"), index=False)
        print("Pulizia completata. Risultati salvati in:", self.output_folder)


    def summarize_results(self):
        """
        Mostra un riassunto dei risultati.
        """
        total_files = len(self.df)
        rejected_files = self.df["Rejected"].sum()
        clean_files = total_files - rejected_files

        print(f"Totale file processati: {total_files}")
        print(f"File accettati: {clean_files}")
        print(f"File rifiutati: {rejected_files}")
        
        
    # Esempio di funzione per estrarre feature (calcola la media dei valori nel file)
    def example_feature_extractor(file_path):
        try:
            # get RGB mean from image file
            image = Image.open(file_path)
            R, G, B = image.split()
            R_mean = np.mean(np.array(R))
            G_mean = np.mean(np.array(G))
            B_mean = np.mean(np.array(B))
            return np.array([R_mean, G_mean, B_mean])
        except Exception as e:
            print(f"Errore durante il caricamento di {file_path}: {e}")
            return None

    # Esempio di funzione di pulizia usando Isolation Forest
    def example_clean_criterion(features, contamination=0.1):
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(features)
        return model.predict(features) == -1  # Ritorna True per gli outlier
