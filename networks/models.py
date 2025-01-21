import torch.nn as nn

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

