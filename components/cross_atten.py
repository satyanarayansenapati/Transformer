import torch.nn as nn

import torch as t
import torch.nn as nn
import torch.nn.functional as f
import math


# scaled dot product function
def scaled_dot_product(q,k,v, mask = None):
  # scaling factor
  d = q.size()[-1]

  qk = t.matmul(q,k.transpose(-1,-2))

  if mask is not None:
    print(f'-----------Adding Mask------------\nmask size = {mask.size()}')
    qk += mask

  # divinding it by scaling factor
  qk_scaled = qk/math.sqrt(d)
  #print('qk scaled shape : ', qk_scaled.shape)

  # applying softmax
  attention = f.softmax(qk_scaled, dim = -1)

  # multiplying with v
  v = t.matmul(attention,v)

  return v,attention


class CrossAtten(nn.Module):

  def __init__(self, model_dim, num_head, annot):
    super().__init__()
    self.model_dim = model_dim
    self.num_head = num_head
    self.annot = annot

    self.kv_layer = nn.Linear(self.model_dim, 2*self.model_dim)
    self.q_layer = nn.Linear(self.model_dim, self.model_dim)
    self.linear = nn.Linear(self.model_dim, self.model_dim)

  def forward(self, x, y):
    batch,seq_len,model_dim = x.size()
    if self.annot : print(f'input X shape : {x.shape}')

    # passing through kv_layer
    kv = self.kv_layer(x)
    if self.annot : print(f'Dimension of input X after passing through KV_layer : {kv.shape}')

    # reshaping for multi head
    kv = kv.reshape(batch, seq_len, self.num_head, -1)
    if self.annot : print(f'Dimension of KV matrix after reshaping : {kv.shape}')

    # swapping the dimensions
    kv = kv.permute(0,2,1,3)
    if self.annot : print(f'Dimension of KV matrix after swapping : {kv.shape}')

    # divinding k,v from KV matrix
    k,v = kv.chunk(2, dim = -1)
    if self.annot : print(f'Creating K, V matrix from KV matrix. Shape of K and V is : {k.shape}')

    # passing through q layer
    q = self.q_layer(y)
    if self.annot : print(f'Dimension of Y after passing through Q_layer : {q.shape}')

    # reshaping for multihead
    q = q.reshape(batch, seq_len, self.num_head,-1)
    if self.annot : print(f'Q matrix shape after reshaping it for multihead attention : {q.shape}')

    # swapping the dimensions
    q = q.permute(0,2,1,3)
    if self.annot : print(f'Q matrix shape after reshaping it for multihead attention : {q.shape}')

    # scaled dot product
    v, _ = scaled_dot_product(q,k,v,mask= None)
    if self.annot : print(f'Cross attention calculated. Shape of V matrix : {v.shape}')

    # concate the result from each head
    v = v.reshape(batch, seq_len,-1)
    if self.annot : print(f'Combining the results from each head, the final shape of V matrix : {v.shape}')

    # passing it through linear layer
    out = self.linear(v)
    if self.annot : print(f'Final shape of V matrix from multi head cross attention : {out.shape}')

    return out