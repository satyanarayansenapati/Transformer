from components.multihead_attn import MultiheadAtt
from components.Layer_normalization import LayerNorm
from components.feed_forward import FeedForward

import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self,d_model,hidden_dim,dropout, num_head, annot, mask):
        super(EncoderLayer,self). __init__()
        self.annot = annot
        self.attention = MultiheadAtt(model_dim=d_model,num_head=num_head,annot=self.annot)
        self.norm1 = LayerNorm(model_dim=d_model,annot= self.annot)
        self.dropout1 = nn.Dropout(p = dropout)
        self.ff = FeedForward(hidden_dim=hidden_dim,model_dim=d_model,dropout=0.15,annot=self.annot)
        self.norm2 = LayerNorm(model_dim=d_model,annot= self.annot)
        self.dropout2 = nn.Dropout(p = dropout)
        

    def forward(self,x):
        residual = x

        # passing through multihead attention
        if self.annot : print(f'\n\n----------------------Passing through multihead attention-------------------\n')
        x = self.attention(x, mask = None)

        if self.annot :print('--------------------1st dropout-------------------------------------------------\n')
        x = self.dropout1(x)

        if self.annot :print('--------normalization (residual connection + output from attention heads)-------\n')
        x = self.norm1(x + residual)
        residual = x

        if self.annot :print('-----------------------Passing through feed forward network---------------------\n')
        x = self.ff(x)

        if self.annot :print('-------------------------------Second dropout------------------------------------\n')
        x = self.dropout2(x)

        if self.annot :print('-------------Normalization(Residual + ff layer putput)--------------------------\n')
        x = self.norm2(x + residual)
        if self.annot :print('----------------------------Final Output from Encoder-------------------------\n\n')
        return x


class Encoder(nn.Module):

    def __init__(self,d_model,hidden_dim,dropout, num_head,num_layers,annot, mask = None ):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout, num_head=num_head, annot=annot, mask=mask) for _ in range(num_layers)])
        #self.la = num_layers
    def forward(self,x):
        x = self.layers(x)
        return x