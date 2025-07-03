import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embd = 384 # size of embedding vector
dropout = 0.2 # dropout probability

n_head=6 # number of heads in the multi-head attention """"Ensure that n_embd is divisible by n_head""""
n_block=6 # number of layers in the transformer


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#Encoder Decoder 
stoi={}
itos={}
i=0
for i in range(len(chars)):
    stoi[chars[i]]=i
i=0
for i in range(len(chars)):
    itos[i]=chars[i]
def encode(inp):
    out=[]
    i=0
    for i in range(len(inp)):
        out.append(stoi[inp[i]])
    return out
    
def decode(inp):
    return ''.join(itos[i] for i in inp)



# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear( n_embd,head_size, bias=False)
        self.quary=nn.Linear( n_embd,head_size ,bias=False)
        self.value=nn.Linear( n_embd,head_size, bias=False)


        #its a register becuase it will not be updated or its not a parameter of the model
        #its used such that a elemets only depends on the previous elements and not the future elements
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout=nn.Dropout(dropout) # dropout is used to prevent overfitting
    def forward(self, x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.quary(x) 
        wei=q@k.transpose(-2,-1) *(k.shape[-1]**-0.5)  # (B,T,C)@(B,C,T)=(B,T,T) the reason for *head_size**-0.5 is for normalization
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei=F.softmax(wei, dim=-1) 
    
        wei=self.dropout(wei) # dropout is used to prevent overfitting

        v=self.value(x)
        output=wei@v 

        return output
class Block(nn.Module):
    def __init__(self,n_embd,n_heads):
        super().__init__()
        self.head_size=n_embd//n_heads
        self.Mult_attention = MultiHeadAttention(n_heads, self.head_size)
        self.ff=FeedForward(n_embd)
        self.layer_norm1=nn.LayerNorm(n_embd) # layer norm is used to normalize the output of the layer based on the row
        self.layer_norm2=nn.LayerNorm(n_embd) # layer norm is used to normalize the output of the layer based on the row

    def forward(self,x):
        layer_norm1=self.layer_norm1(x) 
        Mult_attention=x+self.Mult_attention(layer_norm1) #"x+" is for  adding the residual connection
        layer_norm2=self.layer_norm2(Mult_attention) 
        x=x+self.ff(layer_norm2)            #"x+" is for adding the residual connection
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,n_embd*4), #*4 for adding more depth to the model
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.linear=nn.Linear(n_embd*4,n_embd) #*4 to n_embd making sure the output is of same size as input

    def forward(self,x):
        x=self.net(x)           #will exicute the linear layer and the relu activation function
        x=self.linear(x)
        return x  
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)]) # 4 heads
        self.linear=nn.Linear(n_embd,n_embd)    
        self.dropout=nn.Dropout(dropout) # dropout is used to prevent overfitting        

    def forward(self, x):
        out = [head(x) for head in self.heads] # (B,T,head_size) for each head
        out = torch.cat(out, dim=-1) # (B,T,head_size*4) which is (B,T,n_embd)
        out = self.linear(out)
        out = self.dropout(out)
        return out
    
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
    
        #self.MultiHeadAttention=MultiHeadAttention(4,n_embd//4) # 4 heads each of size n_embd//4 so after concat it will be n_embd
        #self.FeedForward=FeedForward(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size) # output layer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_block)])
        self.LayerNorm=nn.LayerNorm(n_embd) # layer norm is used to normalize the output of the layer based on the row

    def forward(self, idx, targets=None):
        B,T=idx.shape # B=batch size, T=block size
        # idx and targets are both (B,T) tensor of integers
        Token_emb = self.token_embedding_table(idx) # (B,T,C(n_embd))
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C(n_embd))
        x=Token_emb+pos_emb #(B,T,C(n_embd)+T,C(n_embd)) the 2nd one will be brodcasted to (B,T,C(n_embd))
        #x=self.MultiHeadAttention(x) # (B,T,C(n_embd))
        #x=self.FeedForward(x) # (B,T,C(n_embd))
        x=self.blocks(x) # (B,T,C(n_embd))
        x=self.LayerNorm(x) # (B,T,C(n_embd)) layer norm is used to normalize the output of the layer based on the row
        logits = self.lm_head(x) # (B,T,V(vocab_size))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')


    logits, loss = model(xb, yb)           #This is the forward pass, it will return the logits and loss
    optimizer.zero_grad(set_to_none=True)  
    loss.backward()                        #This is the backward pass, it will calculate the gradients of the loss w.r.t. the parameters of the model
    optimizer.step()                       #This will update the parameters of the model using the gradients



torch.save(model.state_dict(), 'model.pth')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))



#--------------Implemeting "Attention Is All You Need"------------------

#initialialy it was a bigram model only but its loss was high
#then there was a concept of adding conection between the current element and the previous elements
#that was implemented using the attention mechanism + the positional embedding
#but still the loss was high     (loss arround 2.5)
#then i added the multi head attention to the model and the loss was reduced (loss was around 2.2)
#then we are adding a "feed forward layer" to the model and the loss was reduced just giving time for the elements to learn 
#then we are block (run head + feed forward multiple times) the loss was not reduced much but due to its depth its difficult to train
#then we are adding the residual connection to the model(loss was around 2.09)
#then we are adding the layer norm(normalized the input for every row mean 0 variance 1 ) to the model (loss was around 1.9)
#then we are adding the dropout to the model to reduce overfitting as we increased the parameters

#layerNorm is like if the input is b,t,c 
#then for undertanding it will make it b,tc then for each row it will make the mean 0 and variance 1 (normalize it)
#then it will make it b,t,c again


#------------------------------------------
#The current code it is just the decoder block as it is just generating the next token based on the previous tokens 