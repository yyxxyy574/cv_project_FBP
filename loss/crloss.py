'''
损失函数设计
为分类损失和回归损失的加权和
'''

import torch.nn as nn

class CRLoss(nn.Module):
    def __init__(self, weight_classifier=0.4, weight_regressor=0.6):
        super(CRLoss, self).__init__()
        self.weight_classifier = weight_classifier
        self.weight_regressor = weight_regressor
        self.criterion_classifier = nn.CrossEntropyLoss()
        self.criterion_regressor = nn.MSELoss()
        
    def forward(self, pred_score, gt_score, pred_class, gt_class):
        loss_classfier = self.criterion_classifier(pred_class, gt_class)
        loss_regressor = self.criterion_regressor(pred_score, gt_score)
        
        return self.weight_classifier * loss_classfier + self.weight_regressor * loss_regressor