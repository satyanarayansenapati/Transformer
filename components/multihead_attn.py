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

class MultiheadAtt(nn.Module):

  def __init__(self, model_dim, num_head,annot=False): # Added a default value for mask
    super().__init__()

    self.model_dim = model_dim
    self.num_head = num_head

    #assert (self.model_dim//self.num_head)== 0, "Model dimension must be divisible by number of heads"

    self.annot  = annot

    # linear transform layer to convert input dim to 3*model_dim
    self.qkv_layer = nn.Linear(self.model_dim, 3*self.model_dim)

    # linear layer
    self.linear = nn.Linear(self.model_dim, self.model_dim)

  def forward(self, x, mask=None): # Added mask as an argument with a default value

    batch, seq_len, input_dim = x.size()

    if self.annot : print(f'the input to multihead : shape = {x.shape}')

    # preparing for qkv layer
    x = self.qkv_layer(x)
    if self.annot : print(f'after passing through qkv layer : shape = {x.shape}')

    # we will split the last dimension as per number of heads and rearrange the vector
    x = x.reshape(batch, seq_len, self.num_head, 3 *(self.model_dim//self.num_head))
    if self.annot : print(f'reshaping the input for multihead : shape = {x.shape}')

    # swapping the dimensions [b,s,h,d] -> [b,h,s,d]
    x = x.permute(0,2,1,3)
    if self.annot : print(f'swapping the dimensions [b,s,h,d] -> [b,h,s,d] : shape = {x.shape}')

    # splitting into q k v
    q,k,v = x.chunk(3, dim = -1)
    if self.annot : print(f'shape of each q k v matrix is : shape = {q.shape}')

    # passing through attention
    v, attention = scaled_dot_product(q,k,v, mask = mask) # Changed self.mask to mask
    if self.annot : print(f'passed through attention. value matrix : shape = {v.shape}')

    # reshaping the value matrix
    v = v.reshape(batch, seq_len,-1)
    if self.annot : print(f'shape of V after reshaping : shape = {v.shape}')

    # passing through final layer, linear layer
    out = self.linear(v)
    if self.annot : print(f'shape of V after passing through linear linear layer : shape = {v.shape}')

    return out