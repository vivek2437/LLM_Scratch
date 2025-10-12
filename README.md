# LLM_Scratch

A from-scratch implementation and exploration of Large Language Models (LLMs), designed for learning, experimentation, and extension. This repository aims to demystify the inner workings of LLMs by providing transparent code, modular components, and clear explanations.

---

## Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Notebook Contents](#notebook-contents)
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
│   ├── attention.ipynb
│   ├── attention.py
│   ├── llm_gpt.ipynb
│   ├── llm_gpt.py
│   ├── text_data.ipynb
│   └── text_data.py
└── teh-verdict.txt
```

- **notebook/**: Contains Jupyter notebooks and supporting scripts for LLM components and experiments.
- **teh-verdict.txt**: Sample text data used for tokenization and language modeling.

---

## Notebook Contents

### notebook/text_data.ipynb
- **Section 1.1**: Downloads and loads `teh-verdict.txt` if not present.
- Tokenizes the text using regex (splitting, filtering).
- Builds vocabulary (unique tokens).
- Maps each token to a unique integer ID and explores vocabulary statistics.

### notebook/attention.ipynb
- **Section 2.x**: 
    - Simple self-attention without trainable weights.
    - Dot-product attention and softmax normalization.
    - Trainable self-attention step-by-step implementation.
    - Compact self-attention as a PyTorch class.
    - Causal (autoregressive) masking and dropout in attention.
    - Finalizes with a reusable SelfAttention module.

### notebook/llm_gpt.ipynb
- **Section 3.x**:
    - Defines a GPT/transformer config.
    - Implements dummy transformer blocks, layer normalization, GELU activations.
    - Adds feed-forward layers.
    - Demonstrates residual (shortcut) connections.
    - Builds a full GPT model class.
    - Shows text generation, decoding, and parameter/memory sizing.
    - Generates text using the model.

---

## Usage

- **Run any notebook (example with Jupyter):**
  ```bash
  jupyter notebook
  ```
  Open and explore notebooks in the `notebook/` directory.

- **Data:**  
  The code will automatically fetch `teh-verdict.txt` if not present.

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

## Example: Simple Self-Attention (from attention.ipynb)

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

## Example: Minimal GPT Model (from llm_gpt.ipynb)

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
