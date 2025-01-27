import torch
import torch.nn as nn

class ModelOptions:
    def __init__(self, 
                 criterion: torch.nn, 
                 optimizer: torch.optim, 
                 scheduler: torch.optim, 
                 input_dim = 2048, 
                 num_classes = 251, 
                 batch_size = 2048, 
                 epochs = 100, 
                 patience = 10):
        self.criterion = criterion
        self.optimizer = optimizer
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size 
        self.epochs = epochs
        self.patience = patience
        self.scheduler = scheduler
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

# Definizione del modello
class OneLayerNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OneLayerNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class ClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassifierNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

