from abc import ABC
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

class ImageRetrieval(ABC):
    def __init__(self, dataset, dataset_labels, queryset):
        self.dataset = dataset
        self.dataset_labels = dataset_labels
        self.queryset = queryset
        
    def retrieve_images(self):
        pass

class ImageRetrievalKNNCentroids(ImageRetrieval):
    def __init__(self, dataset, dataset_labels, queryset, algo = 'ball_tree', computes_centroid = True, standardize = True, n_image_per_class=5, weights='uniform'):
        super().__init__(dataset, dataset_labels, queryset)
        self.n_image_per_class = n_image_per_class
        self.standardize = standardize
        self.weights = weights
        self.algo = algo
        self.computes_centroid = computes_centroid

    def compute_centroids(self):
        centroids = []  

        for target_class in np.unique(self.dataset_labels):
            centroid = self.dataset[self.dataset_labels == target_class].mean(axis=0)
            centroids.append(centroid) 
        
        return np.array(centroids)

    def retrieve_images(self):
        if self.computes_centroid:
            db = self.compute_centroids()
        else:
            db = self.dataset

        knn = NearestNeighbors(n_neighbors=self.n_image_per_class, algorithm=self.algo)

        if self.standardize:
            scaler = StandardScaler()
            db = scaler.fit_transform(db)
            queryset = scaler.transform(self.queryset)

        knn.fit(queryset)
        _, indeces_per_class = knn.kneighbors(db)

        idxs = []
        labels = []
        for label, indeces in enumerate(indeces_per_class):
            for i in indeces:
                idxs.append(i)
                labels.append(label)

        # ritorna gli indici da considerare e le loro labels
        return idxs, labels
    

class ImageRetrievalKNN(ImageRetrieval):
    def __init__(self, dataset, dataset_labels, queryset, standardize = True, n_neighbors=5, weights='uniform'):
        super().__init__(dataset, dataset_labels, queryset)
        self.n_neighbors = n_neighbors
        self.standardize = standardize
        self.weights = weights
        
    def retrieve_images(self):
        std_pipeline = Pipeline([
            ('scaler', StandardScaler()),            
            ('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights))
        ])

        pipeline = Pipeline([
            ('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights))
        ])

        if self.standardize:
            std_pipeline.fit(self.dataset, self.dataset_labels)
            predictions = std_pipeline.predict(self.queryset)
        else:
            pipeline.fit(self.dataset, self.dataset_labels)
            predictions = pipeline.predict(self.queryset)
            
        # ritorna gli indici da considerare e le loro labels
        return list(range(len(predictions))), list(map(int, predictions))
    
    
class ImageRetrievalBestFit(ImageRetrieval):
    def __init__(self, dataset, dataset_labels, queryset, standardize=True, n_neighbors=5, distance_metric='euclidean'):
        super().__init__(dataset, dataset_labels, queryset)
        self.n_neighbors = n_neighbors
        self.standardize = standardize
        self.distance_metric = distance_metric
        
    def retrive_images(self):
        # Filtra i dati del dataset per la classe target
        indices = []
        labels = []
        for target_class in np.unique(self.dataset_labels):
            # stampa senza andare a capo per mostrare la classe target
            print(f"Classe target: {target_class}", end='\r')
            target_data = self.dataset[self.dataset_labels == target_class]
            
            # Standardizzazione dei dati se richiesto
            queryset_to_use = self.queryset
            dataset_to_use = target_data
            if self.standardize:
                scaler = StandardScaler()
                queryset_to_use = scaler.fit_transform(self.queryset)
                dataset_to_use = scaler.transform(target_data)
            
            # Calcola le distanze tra il queryset e il dataset della classe target
            distances = cdist(queryset_to_use, dataset_to_use, metric=self.distance_metric)
            
            # Trova gli n_neighbors più vicini
            nearest_indices = np.argsort(np.min(distances, axis=1))[:self.n_neighbors]
            indices.extend(nearest_indices)
            labels.extend([target_class] * self.n_neighbors)
        #remove np int
        indices = list(map(int, indices))
        labels = list(map(int, labels))
        return indices, labels
