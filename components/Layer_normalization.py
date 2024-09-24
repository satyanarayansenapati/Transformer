import torch as t
import torch.nn as nn


# layer normalization
class LayerNorm(nn.Module):

  def __init__(self,model_dim,eps = 1e-5, annot = False):
    super().__init__()

    self.eps = eps
    self.model_dim = model_dim
    self.annot = annot

    self.gamma = nn.Parameter(t.ones(self.model_dim))
    #if self.annot : print(f'gamma : shape = {self.gamma.shape}')

    self.beta = nn.Parameter(t.zeros(self.model_dim))
    #if self.annot : print(f'beta : shape = {self.beta.shape}')

  def forward(self, x):

    # getting the last dimension on which normalization will be applied
    dim = x.size()[-1]

    # calculating mean
    mean = x.mean(dim = -1, keepdims = True)
    if self.annot : print(f'shape of mean is : shape = {mean.shape}')

    # calculating variance
    var = ((x-mean)**2).mean(dim = -1, keepdims = True)
    if self.annot : print(f'variance shape is : shape = {var.shape}')

    # calculating standard deviation
    std = t.sqrt(var+self.eps)
    if self.annot : print(f'standard deviation shape is : shape = {std.shape}')

    y = (x - mean)/std
    if self.annot : print(f'y = (x-mean)/std shape is : shape = {y.shape}')

    out = self.gamma * y + self.beta
    if self.annot : print(f'shape of normalized output is : shape = {out.shape}')

    return out