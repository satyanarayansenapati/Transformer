from components.cross_atten import CrossAtten
from components.multihead_attn import MultiheadAtt
from components.Layer_normalization import LayerNorm
from components.feed_forward import FeedForward

import torch.nn as nn


# multihead attention
class DecoderLayer(nn.Module):

  def __init__(self,model_dim, hidden_dim, num_head, dropout, annot):
    super(). __init__()
    self.model_dim = model_dim
    self.hidden_dim = hidden_dim
    self.num_head = num_head
    self.dropout = dropout
    self.annot = annot

    self.masked_attention = MultiheadAtt(model_dim = self.model_dim, num_head = self.num_head, annot = self.annot)
    self.norm1 = LayerNorm(model_dim=self.model_dim, annot = self.annot)
    self.drop1 = nn.Dropout(p = self.dropout)
    self.cross_attention = CrossAtten(model_dim=self.model_dim, num_head=self.num_head,annot = self.annot)
    self.norm2 = LayerNorm(model_dim = self.model_dim, annot = self.annot)
    self.drop2 = nn.Dropout(p = self.dropout)
    self.ff = FeedForward(hidden_dim=self.hidden_dim, model_dim = self.model_dim, dropout = self.dropout,annot = self.annot)
    self.norm3 = LayerNorm(model_dim = self.model_dim, annot = self.annot)
    self.drop3 = nn.Dropout(p = self.dropout)


  def forward(self,x,y,mask):

    res_y = y

    if self.annot : print('\n\n--------------------Passing through masked attention--------------------------\n')
    y = self.masked_attention(y,mask)

    if self.annot : print('--------------------Dropout#1--------------------------\n')
    y = self.drop1(y)

    if self.annot : print('--------------------Normalization(Res + Out)#1--------------------------\n')
    y = self.norm1(res_y + y)
    res_y = y

    if self.annot : print('--------------------Cross Attention--------------------------\n')
    out = self.cross_attention(x=x,y=y)

    if self.annot : print('--------------------Dropout#2--------------------------\n')
    out = self.drop2(out)

    if self.annot : print('--------------------Normalization(Res + Out)#2--------------------------\n')
    out = self.norm2(out + res_y)
    res_y = out

    if self.annot: print('-------------------FF Network--------------------------\n')
    out = self.ff(out)

    if self.annot : print('--------------------Dropout#3--------------------------\n')
    out = self.drop3(out)

    if self.annot : print('--------------------Normalization(Res + Out)#3--------------------------\n')
    out = self.norm3(out + res_y)

    if self.annot : print('--------------------Final Output from Encoder is delivered--------------------------\n\n')

    return out
  

class SeqDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y,mask = inputs

        for module in self._modules.values():
            y = module(x,y,mask)

        return y
    

class Decoder(nn.Module):

  def __init__(self,model_dim, hidden_dim, num_head, dropout,annot = False, num_layers = 1):
    super().__init__()
    self.layers = SeqDecoder(*[DecoderLayer(model_dim = model_dim,hidden_dim= hidden_dim, num_head = num_head,dropout=dropout,annot = True) for _ in range(num_layers)] )

  def forward(self,x,y,mask):
    y = self.layers(x,y, mask)

    return y