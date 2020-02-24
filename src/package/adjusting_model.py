import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

"""
Generalized mean layer

"""
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

"""
Swish activation function

"""
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
#         self.inplace = inplace
    def forward(self, x, beta=1.12):
        return x * torch.sigmoid(beta * x)
def convert_relu_to_swish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, swish())
        else:
            convert_relu_to_swish(child)