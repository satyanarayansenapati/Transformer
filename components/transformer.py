import torch.nn as nn
import torch as t

from components.encoder import Encoder
from components.decoder import Decoder
from components.pos_embed import PosEmbed

class Transformer(nn.Module):

  def __init__(self, d_model,hidden_dim,dropout, num_head,num_layers, annot):
    super().__init__()
    
    self.d_model = d_model
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.num_head = num_head
    self.annot = annot
    self.num_layer = num_layers


    self.encoder = Encoder(d_model = self.d_model, 
                           hidden_dim=self.hidden_dim,
                           dropout=self.dropout,
                           num_head=self.num_head,
                           num_layers=self.num_layer,
                           annot = self.annot)
    
    self.decoder = Decoder(model_dim=self.d_model,
                           hidden_dim=self.hidden_dim,
                           num_head=self.num_head,
                           dropout=self.dropout,
                           annot = self.annot, 
                           num_layers=self.num_layer)
 

  def forward(self, x, y, mask):
    batch, seq_len, input_dim = x.size()

    if self.annot : print(f"Adding postional embeddings")
    pos_embed = PosEmbed(seq_len, input_dim, annot = self.annot)
    pos_embed = pos_embed.forward()
    x = t.add(x, pos_embed)
    if self.annot : print(f'Shape of input X : {x.shape}')
    if self.annot : print(f'Shape of input Y : {y.shape}')
    if self.annot : print(f'Shape of input mask : {mask.shape}')
    x = self.encoder(x)
    if self.annot : print(f'Shape of output from encoder : {x.shape}')
    y = self.decoder(x,y,mask)
    if self.annot : print(f'Shape of output from decoder : {y.shape}')
    return y