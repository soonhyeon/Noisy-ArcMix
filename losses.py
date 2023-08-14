import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Parameter
import math 


class ASDLoss(nn.Module):
    def __init__(self, reduction=True):
        super(ASDLoss, self).__init__()
        if reduction == True:
            self.ce = nn.CrossEntropyLoss()
        
        else:
            self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss

# ArcFace is referred to https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=30.0, m=0.7, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output
