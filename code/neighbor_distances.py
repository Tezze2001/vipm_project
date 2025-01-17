import numpy as np

def compute_class_centroids(X, y):
    centroids = {}
    unique_classes = np.unique(y)
    for cls in unique_classes:
        centroids[cls] = X[y == cls].mean(axis=0)
    return centroids

class CentroidWeights:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.centroids = compute_class_centroids(X_train, y_train)
    
    def __centroid_based_weights(self, distances):
        
        n_queries = distances.shape[0]  # Numero di query
        weights = []

        for i in range(n_queries):
            # Ottenere le distanze per la query corrente (i-esimo esempio)
            query_distances = distances[i]
            
            # Per ogni distanza calcolare il peso basato sul centroide
            query_weights = []
            for j, dist in enumerate(query_distances):
                # Determina la classe del vicino j-esimo
                neighbor_class = self.y_train[j]  # Notare che j fa riferimento all'indice del vicino nel training
                centroid = self.centroids[neighbor_class]  # Centroide della classe del vicino
                # Calcola la distanza dal centroide del vicino
                distance_to_centroid = np.linalg.norm(self.X_train[j] - centroid)
                # Calcolo del peso
                weight = 1 / (1 + distance_to_centroid)
                query_weights.append(weight)

            # Aggiungi i pesi per la query corrente
            weights.append(query_weights)

        return np.array(weights)

        
    def __call__(self, distances):
        return self.__centroid_based_weights(distances)
