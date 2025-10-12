#!/usr/bin/env python
# coding: utf-8

# # 2 Attention Mechanisms

# # 2.1 A Simple Self-Attention Mechanism Without Trainable Weights

# In[1]:


import torch

inputs= torch.tensor(
    [[0.43,0.20,0.90],
     [0.56,0.84,0.56],
     [0.22,0.85,0.34],
     [0.77,0.25,0.36],
     [0.05,0.80,0.79],
     [0.77,0.46,0.12]]
)   


# In[2]:


input_query= inputs[1]


# In[3]:


input_1= inputs[0]


# In[4]:


res =0
i=0
res = torch.dot(inputs[i],input_query)
print(res)   


# In[5]:


query=inputs[1]

atten_scores_2 = torch.empty(inputs.shape[0])

for i,x_i in enumerate(inputs):
    atten_scores_2[i]=torch.dot(x_i,query)

print(atten_scores_2)


# In[6]:


attn_weights_2_tmp= atten_scores_2 / atten_scores_2.sum()

attn_weights_2_tmp


# In[7]:


attn_weights_2_tmp.sum()


# In[8]:


def softmax_navie(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

softmax_navie(atten_scores_2)


# In[9]:


atten_weights_2=torch.softmax(atten_scores_2 ,dim=0)


# In[10]:


query = inputs[1]

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):

    print(f"{atten_weights_2[i]} ----> {inputs[i]}")
    context_vec_2+= atten_weights_2[i]*x_i

print(context_vec_2)


# In[11]:


for i,x_i in enumerate(inputs):
    print(i, inputs[i])  # i_x is the index, i is the tensor row


# # 2.2 Computing attention weights for all input tokens

# In[12]:


attn_scores = torch.empty(6,6)

for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
     attn_scores[i,j]= torch.dot(x_i,x_j)

print(attn_scores)


# In[13]:


attn_weights = torch.softmax(attn_scores,dim=1)


# In[14]:


attn_weights


# # 2.3 Implementing self-attention with trainable weights

# # 2.3.1 Computing the attention weights step by step

# In[15]:


x_2 = inputs[1]

d_in= inputs.shape[1]

d_out=2


# In[16]:


torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in ,d_out))
W_key = torch.nn.Parameter(torch.rand(d_in,d_out))
W_value = torch.nn.Parameter(torch.rand(d_in,d_out))


# In[17]:


query_2 = x_2 @ W_query

query_2


# In[18]:


keys= inputs @W_key
value=inputs @W_value

keys.shape


# In[19]:


keys_2 =keys[1]

attn_score_22=torch.dot(query_2 ,keys_2)


# In[20]:


attn_score_22


# In[21]:


attn_scores_2= query_2 @ keys.T

attn_scores_2


# In[22]:


d_k=keys.shape[1]

attn_weights_2=torch.softmax(attn_scores_2/d_k**0.5,dim=-1)
attn_weights_2


# In[23]:


context_vec_2 = attn_weights_2 @ value

context_vec_2


# # 2.4 Implementing a compact SelfAttention Class

# In[24]:


import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self,d_in,d_out):
     super().__init__()
     self.W_query = torch.nn.Parameter(torch.rand(d_in ,d_out))
     self.W_key = torch.nn.Parameter(torch.rand(d_in,d_out))
     self.W_value = torch.nn.Parameter(torch.rand(d_in,d_out))

    def forward(self,x):
        queries=inputs @ W_query
        keys=inputs @W_key
        values=inputs @W_value

        attn_scores=queries @ keys.T
        attn_weighs= torch.softmax(attn_scores / d_k**0.5,dim=-1)
        context_vec = attn_weights @ values


        return context_vec

torch.manual_seed(123)
sa_v1 =SelfAttention_v1(d_in,d_out)
sa_v1(inputs)


# In[25]:


import torch.nn as nn

class SelfAttention_v2(nn.Module):

    def __init__(self,d_in,d_out,qkv_bias=False):
     super().__init__()
     self.W_query = torch.nn.Linear(d_in ,d_out,bias=qkv_bias)
     self.W_key = torch.nn.Linear(d_in,d_out,bias=qkv_bias)
     self.W_value = torch.nn.Linear(d_in,d_out,bias=qkv_bias)

    def forward(self,x):
        queries=self.W_query(inputs)
        keys=self.W_key(inputs)
        values=self.W_value(inputs)

        attn_scores=queries @ keys.T
        attn_weighs= torch.softmax(attn_scores / d_k**0.5,dim=-1)
        context_vec = attn_weights @ values


        return context_vec

torch.manual_seed(123)
sa_v2 =SelfAttention_v2(d_in,d_out)
sa_v2(inputs)


# # 2.5 Hiding future words with casual attention 

# # Applying a casual attention mask

# In[26]:


queries=sa_v2.W_query(inputs)
keys=sa_v2.W_key(inputs)
values=sa_v2.W_value(inputs)

attn_scores=queries @ keys.T
attn_weighs= torch.softmax(attn_scores / d_k**0.5,dim=-1)


# In[27]:


context_length = attn_scores.shape[0]

mask_simple =torch.tril(torch.ones(context_length, context_length))

print(mask_simple)


# In[28]:


masked_simple =attn_weights * mask_simple
masked_simple


# In[29]:


row_sums =masked_simple.sum(dim=-1 ,keepdim=True)
masked_simple_norm =masked_simple / row_sums
print(masked_simple_norm)


# In[30]:


mask = torch.triu(torch.ones(context_length,context_length,context_length),diagonal=1)
masked= attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)


# In[31]:


attn_weights=torch.softmax(masked /keys.shape[-1]**0.5,dim=-1)
print(attn_weights)


# # Masking additional attention weights with dropout

# In[32]:


torch.manual_seed(123)

layer=torch.nn.Dropout(0.5)


# In[33]:


drop_rate=0.3

1/(1-drop_rate)


# In[34]:


layer(attn_weights)


# # Implementing a compact casual self_attention class

# In[35]:


batch = torch.stack((inputs,inputs),dim=0)


# In[36]:


import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], float('-inf')
        )

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec

torch.manual_seed(123)

batch = torch.randn(2, 6, 3)  # [batch_size=2, seq_len=6, d_in=3]
d_in = 3
d_out = 4
context_length = batch.shape[1]
dropout = 0.0

ca = CausalAttention(d_in, d_out, context_length, dropout)
out = ca(batch)


# # 2.6 Extending single-head attention to multi-head attention

# # 2.6.1 Stacking multiple single-head attention layers

# In[37]:


import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False, num_heads=2):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # Concatenate outputs from each head along feature dimension
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)

batch = torch.randn(2, 6, 3)  # [batch_size, seq_len, d_in]
context_length = batch.shape[1]
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, qkv_bias=False, num_heads=2)
mha(batch)


# # 2.6.2 Implementing multi-head attention with weight splits

# In[38]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, hea
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


# In[39]:


import torch

torch.manual_seed(123)

batch_size = 2
max_length = 6        # sequence length
output_dim = 3        # embedding dimension
d_in = output_dim
d_out = d_in

# random batch of input embeddings
input_embeddings = torch.randn(batch_size, max_length, d_in)

mha = MultiHeadAttentionWrapper(d_in, d_out, max_length, dropout=0.0, qkv_bias=False, num_heads=2)
context_vecs = mha(input_embeddings)

print("context_vecs.shape:", context_vecs.shape)

