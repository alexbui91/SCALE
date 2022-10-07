import torch.nn as nn
import torch.nn.functional as F
import torch

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.3, reduction='sum'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, y_hat, y):
        p = F.log_softmax(y_hat / self.temperature, 1)
        weight = F.softmax(y / self.temperature, 1)
        loss = -(p * weight)
        if self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss

# use this for amazon and yelp # https://bit.ly/3cXP59n https://bit.ly/3zL3M8p
def imbalanceCriteria(labels, device=None, num_classes=2):
    num_train = len(labels)
    weights = []
    for n in range(num_classes):
        num_pos = (labels == n).to(torch.float32).sum()
        weight = num_train / (num_classes * num_pos)
        weights.append(weight)
    criteria = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
    return criteria