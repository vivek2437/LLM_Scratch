# LLM_Scratch

A from-scratch implementation and exploration of Large Language Models (LLMs), designed for learning, experimentation, and extension. This repository aims to demystify the inner workings of LLMs by providing transparent code, modular components, and clear explanations.

---

## Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Details: Notebooks Folder](#details-notebooks-folder)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vivek2437/LLM_Scratch.git
   cd LLM_Scratch
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**  
   There is no requirements.txt at the root. You may need to check the import statements in the notebooks (mainly PyTorch, numpy, matplotlib, regex, urllib, etc.).  
   Example:
   ```bash
   pip install torch numpy matplotlib tiktoken
   ```

---

## Repository Structure

```
LLM_Scratch/
├── notebook/
│   ├── step1.ipynb
│   ├── step2.ipynb
│   ├── step3.ipynb
│   └── step4.ipynb
├── teh-verdict.txt
└── README.md
```

- **notebook/**: Contains Jupyter notebooks for each major conceptual step in building an LLM from scratch.
- **teh-verdict.txt**: Sample text data used for tokenization and language modeling.

---

## Details: Notebooks Folder

### notebook/step1.ipynb — Data Loading & Tokenization
- Downloads and reads `teh-verdict.txt` if not present, from an external URL.
- Loads the file’s contents into a string.
- Tokenizes the text using regex, splits and filters, then prepares a vocabulary (unique tokens) and maps each token to a unique integer ID.
- Sets up the dataset for training and experiment steps.
- **Key Concepts:** Data fetching, text preprocessing, vocabulary building.

### notebook/step2.ipynb — Attention Mechanisms
- Introduces the concept of attention in neural networks.
- Implements a simple self-attention mechanism (no trainable weights) using PyTorch tensors, computes dot products between input vectors and a query, and prints attention scores.
- Progresses to softmax normalization, attention weights, and context vectors.
- Builds toward more complex attention modules (including trainable weights, masking for causality, and modularization).
- **Key Concepts:** Self-attention, dot-product, softmax, context vectors, masking.

### notebook/step3.ipynb — From Attention to GPT
- Implements a GPT-like architecture from scratch.
- Defines a model config (vocab size, embedding size, context length, number of heads/layers, dropout, etc.).
- Implements dummy transformer blocks and layer normalization in PyTorch.
- Assembles token embeddings, positional embeddings, dropout, transformer block stack, normalization, and output head.
- Provides a forward pass for model input through all layers.
- **Key Concepts:** Transformer block structure, layer normalization, residual connections, full model assembly, text generation, model parameterization.

### notebook/step4.ipynb — Advanced/Experimental (Expand as Needed)
- This notebook is intended for continuing the LLM project, adding advanced features, experiments, training/inference workflows, or research ideas.

---

## Usage

- **Run notebooks:**
  ```bash
  jupyter notebook
  ```
  Open and explore the notebooks in the `notebook/` directory.

- **Data:**  
  The code will fetch `teh-verdict.txt` automatically if not present.

- **Dependencies:**  
  Most notebooks require: `torch`, `numpy`, `tiktoken`, `matplotlib`, and Python standard libraries.

---

## Example: Tokenizing Text Data

```python
import os, urllib.request
if not os.path.exists("teh-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    urllib.request.urlretrieve(url, "teh-verdict.txt")

with open("teh-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import re
result = re.split(r'([,.:;?_!"()\'--|\s])', raw_text)
result = [item.strip() for item in result if item.strip()]
vocab = sorted(set(result))
word_to_id = {word: idx for idx, word in enumerate(vocab)}
```

---

## Example: Simple Self-Attention

```python
import torch
inputs = torch.tensor([
    [0.43,0.20,0.90],
    [0.56,0.84,0.56],
    [0.22,0.85,0.34],
    [0.77,0.25,0.36],
    [0.05,0.80,0.79],
    [0.77,0.46,0.12]
])
query = inputs[1]
atten_scores = torch.tensor([torch.dot(x_i, query) for x_i in inputs])
attn_weights = torch.softmax(atten_scores, dim=0)
context_vec = torch.sum(attn_weights[:, None] * inputs, dim=0)
```

---

## Example: Minimal GPT Model

```python
import torch, torch.nn as nn
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[nn.Identity() for _ in range(cfg["n_layers"])]
        )
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.out_head(x)
        return x
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## License

This repository is licensed under the MIT License.

---

## Contact

For questions, suggestions, or collaboration, please contact [vivek2437](https://github.com/vivek2437).
