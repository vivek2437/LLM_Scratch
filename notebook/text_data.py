#!/usr/bin/env python
# coding: utf-8

# # 1.1 Tokenize The Text Data

# In[1]:


import os 
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url=("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt")

    file_path="teh-verdict.txt"

    urllib.request.urlretrieve(url,file_path)


# In[2]:


with open("teh-verdict.txt", "r" ,encoding="utf-8") as f:
    raw_text=f.read()


# In[3]:


raw_text


# In[4]:


len(raw_text)


# In[5]:


import re

text= "Hello , world. This is a test."

result=re.split(r'(\s)',text)


# In[6]:


print(result)


# In[7]:


result=[item for item in result if item.strip()]

print(result)


# In[8]:


text="hello,world.is this is a test?"

result=re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)

result=[item.strip() for item in result if item.strip()]


preprocessed=result


# In[9]:


len(preprocessed)


# # 1.2 Converting Token into token IDs

# In[10]:


all_words=sorted(set(preprocessed))

len(all_words)


# In[11]:


vocab={ token:integer for integer,token in enumerate(all_words)}

vocab


# In[12]:


# import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed if s in self.str_to_int]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# In[13]:


tokenizer=SimpleTokenizerV1(vocab)


# In[14]:


text=""""It's the last he painted,you know," Mrs.Gisburn said with pardonable pribe."""


# In[15]:


ids=tokenizer.encode(text)

print(ids)


# In[16]:


tokenizer.decode(ids)


# In[17]:


tokenizer.decode(tokenizer.encode(text))


# # 1.3 Adding Special Context tokens

# In[18]:


text = "Hello, do like tea, is this a test?"

tokenizer.encode(text)


# In[19]:


all_tokens=sorted(list(set(preprocessed)))

all_tokens.extend(["<endoftext|>","|unk|>"])

vocab={token:integer for integer,token in enumerate(all_tokens)}


# In[20]:


len(vocab.items())


# In[21]:


for i,item in enumerate(list(vocab.items())[-5:]):
    print(item)


# In[22]:


# import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed=[item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed if s in self.str_to_int]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# In[23]:


tokenizer=SimpleTokenizerV2(vocab)


# In[24]:


tokenizer.encode(text)


# In[25]:


tokenizer.decode(tokenizer.encode(text))


# # 1.4 Byte Pair encoding

# In[26]:


import tiktoken


# In[27]:


tokenizer=tiktoken.get_encoding('gpt2')


# In[28]:


tokenizer.encode("hello world")


# In[29]:


tokenizer.decode(tokenizer.encode("hello world"))


# In[30]:


text="Hello,Do you like tea? In the sunlit terraces,of some unknown place are surround in that planet"

tokenizer.encode(text)


# # 1.5 Data Sampling With A Sliding Window

# In[31]:


with open("teh-verdict.txt","r", encoding="utf-8") as f:
    raw_text= f.read()

enc_text= tokenizer.encode(raw_text)

print(len(enc_text))


# In[32]:


enc_text


# In[33]:


enc_sample= enc_text[50:]


# In[34]:


context_size=4

x=enc_sample[:context_size]
y=enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y: {y}")


# In[35]:


for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


# In[36]:


import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# In[40]:


def create_dataloader_V1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):

    #Initialize the tokenizer
    tokenizer=tiktoken.get_encoding("gpt2")

    #Create Dataset
    dataset=GPTDatasetV1(txt,tokenizer,max_length,stride)

    #Create Dataloader
    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# In[38]:


with open("teh-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()


# In[42]:


dataloader=create_dataloader_V1(
    raw_text,batch_size=1,max_length=4,stride=4,shuffle=False
)

data_iter=iter(dataloader)

first_batch=next(data_iter)

print(first_batch)


# In[43]:


second_batch=next(data_iter)

print(second_batch)


# In[44]:


dataloader=create_dataloader_V1(
    raw_text,batch_size=8,max_length=4,stride=4,shuffle=False
)

data_iter=iter(dataloader)

inputs,targets=next(data_iter)

print("Inputs:\n",inputs)

print("\nTargets:\n",targets)


# # 1.6 Creating Token Embeddings

# In[62]:


inputs_ids=torch.tensor([4,5,4,3])


# In[67]:


vocab_size=100
output_dim=10

torch.manual_seed(124)
embedding_layer= torch.nn.Embedding(tokenizer.n_vocab,output_dim)


# In[68]:


print(embedding_layer.weight)


# In[69]:


embedding_layer(torch.tensor([3]))


# In[70]:


embedding_layer(inputs_ids)


# # 1.7 Encoding Word Positions

# In[75]:


vocab_size=50257
output_dim=256

token_embedding_layer=torch.nn.Embedding(vocab_size,output_dim)


# In[72]:


max_length=50

dataloader= create_dataloader_V1( raw_text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False)

data_iter=iter(dataloader)

inputs,targets=next(data_iter)


# In[73]:


print("Token IDs:\n",inputs)

print("\nInputs Shape:\n",inputs.shape)


# In[77]:


token_embeddings=token_embedding_layer(inputs)

token_embeddings.shape


# In[78]:


context_length=max_length

pos_embedding_layer=torch.nn.Embedding(context_length,output_dim)


# In[82]:


pos_embeddings=pos_embedding_layer(torch.arange(max_length))

print(pos_embeddings.shape)


# In[ ]:





# In[83]:


input_embeddings=token_embeddings + pos_embeddings

print(input_embeddings.shape)

