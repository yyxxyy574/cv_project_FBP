import torch.nn as nn
from torchvision import models

import sys
sys.path.append("../")
from utils import num_flat_features
from .regressor import Regressor
from .classifier import Classifier

class FBP(nn.Module):
    def __init__(self):
        super(FBP, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.regressor = Regressor(self.backbone)
        self.classifier = Classifier(self.backbone, num_classes=5)
        
    def forward(self, x):
        for name, module in self.backbone.named_children():
            if name != 'fc':
                x = module(x)
        res_regression = self.regressor(x.view(-1, num_flat_features(x)))
        res_classification = self.classifier(x.view(-1, num_flat_features(x)))
        
        return res_regression, res_classification
    
class FBC(nn.Module):
    def __init__(self, backbone, classifier):
        super(FBC, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        for name, module in self.backbone.named_children():
            if name != 'fc':
                x = module(x)
        res_classification = self.classifier(x.view(-1, num_flat_features(x)))
        
        return res_classification