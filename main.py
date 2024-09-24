from components.transformer import Transformer
import torch as t

'''
d_model = 512
num_heads = 8
drop_prob = 0.1

ffn_hidden = 1028

'''
# receiving inputs from the user
d_model = int(input("Please enter the dimension of the model input :\n "))
num_head = int(input("Please enter the number of heads of the transformer. Make sure model input dimension is divisible by number of heads :\n"))

assert d_model%num_head == 0, "Please enter number of head such that model input dimension will be divisible by it"

drop_prob = float(input("Please enter the dropout layer probability. It should be between 1 to 0:\n"))
assert drop_prob < 1, "Please enter a decimal number between 1 to 0"
ff_hidden = int(input("Please enter the number of hidden layers of the transformer :\n"))

num_layer = int(input("Please enter number of layers of encoder-decoder :\n"))

batch_size = int(input("Please enter the batch size :\n"))
max_sequence_length = int(input("Please enter the maximum sequenece length :\n"))

annot = input("Please type if you want to see the log of the transformer (yes or no):\n")
annot = annot.lower()

if annot == 'yes':
    annot = True
else:
    annot = False


# instancing transformer
transformer = Transformer(d_model=d_model,
                          hidden_dim=ff_hidden,
                          dropout=drop_prob,
                          num_head=num_head,
                          num_layers= num_layer,
                          annot=annot)



# x, y and mask for the transformer
x = t.randn( (batch_size, max_sequence_length, d_model) )
y = t.randn( (batch_size, max_sequence_length, d_model) ) 
mask = t.full([max_sequence_length, max_sequence_length] , float('-inf'))
mask = t.triu(mask, diagonal=1)

# passing x,y, mask through transformer
out = transformer(x,y,mask)