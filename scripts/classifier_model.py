import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes=2):
        
        super().__init__()


        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.classifier(x)
        
    def predict_proba(self, x):
        
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)[:, 1]