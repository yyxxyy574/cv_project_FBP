import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, model, num_classes=5):
        super(Classifier, self).__init__()
        num_features = model.fc.in_features
        self.classfier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.classfier(x)