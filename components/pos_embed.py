import torch as t

class PosEmbed():
  '''
  for postional embeding we need two information from the input

  1. sequence length
  2. embedding dimension = dimension of each embedded word

  '''
  def __init__(self,seq_len = 12, embed_dim = 64, annot = False):
    super().__init__()

    self.seq_len = seq_len
    self.embed_dim = embed_dim
    self.annot = annot

  def forward(self):

    # i , for this we need embed_dim
    i_2 = 2 * t.arange(0,self.embed_dim ,1)
    if self.annot : print(f"2ishape : {i_2.shape}")

    # denominator
    denom = t.pow(1000,(i_2/self.embed_dim))

    # now pos, for this we need sequence length
    even_pos = t.arange(0,self.seq_len ,2).reshape(-1,1)  # ensuring one dimensioni 1 so that division can be carried out
    odd_pos = t.arange(1,self.seq_len ,2).reshape(-1,1)
    if self.annot : print(f'even positions shape : {even_pos.shape}\nodd positions shape : {odd_pos.shape}')

    # applying sin cos to even and odd pos respectively
    even_pe = t.sin(even_pos/denom)
    odd_pe = t.cos(odd_pos/denom)
    if self.annot : print(f'even position size : {even_pe.shape}\nodd position size : {odd_pe.shape}')

    # input.shape = positional_emb.shape
    # input.shape = [seq_len, embed_dim]
    # now stacking even_pe and odd_pe so that positions will be 0,1,2,3,4...... so on
    pos_embed = t.stack([even_pe, odd_pe], dim= 1).view(-1,self.embed_dim)
    if self.annot : print(f'positional embedding shape after stacking : {pos_embed.shape}')

    return pos_embed