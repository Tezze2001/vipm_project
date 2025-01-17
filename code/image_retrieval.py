import pandas as pd
import numpy as np
from abc import ABC

class Data(ABC):
    def __init__(self, path):
        self._path = path
        dataset = np.load(self._path)
        self._features = dataset['features']
        self._labels = dataset['label']

class DataSet(Data):
    def __init__(self, path):
        super().__init__(path)

class QuerySet(Data):
    def __init__(self, path):
        super().__init__(path)
    
    

class ImageRetrieval:
    def __init__(self, dataset_path, queryset_path):
        self.__dataset_path = dataset_path
        self.__dataset = np.load(self.__dataset_path)
        self.__queryset_path = queryset_path
        self.__queryset = np.load(self.__queryset_path)


