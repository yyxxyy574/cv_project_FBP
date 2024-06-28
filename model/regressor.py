'''
回归模型
'''

import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, model):
        super(Regressor, self).__init__()
        num_features = model.fc.in_features
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.regressor(x)