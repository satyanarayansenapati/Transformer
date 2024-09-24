import torch.nn as nn

# the ff neural network
class FeedForward(nn.Module):

  def __init__(self,hidden_dim, model_dim, dropout = 0.1, annot = False):
    super(). __init__()

    self.hidden_dim = hidden_dim
    self.model_dim = model_dim
    self.dropout = dropout
    self.annot = annot

    self.norm1 = nn.Linear(self.model_dim, self.hidden_dim)
    self.norm2 = nn.Linear(self.hidden_dim, self.model_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.dropout)

  def forward(self, x):
    x = self.norm1(x)
    if self.annot : print(f'input shape after passing through 1st ff layer : shape = {x.shape}')

    x = self.relu(x)
    x = self.dropout(x)
    x = self.norm2(x)
    if self.annot : print(f'input shape after passing through last ff layer : shape = {x.shape}')
    return x