# LLM_Scratch

A from-scratch implementation and exploration of Large Language Models (LLMs), designed for learning, experimentation, and extension. This repository aims to demystify the inner workings of LLMs by building key components step-by-step and showing how they come together into a working GPT-style model.

---

## Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Details: Notebooks Folder](#details-notebooks-folder)
  - [Step 4 — Pretraining, Evaluation, and Decoding (Full Details)](#step-4--pretraining-evaluation-and-decoding-full-details)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vivek2437/LLM_Scratch.git
   cd LLM_Scratch
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   There is no requirements.txt at the root. Check notebook imports to install needed packages.
   Common set for Step 4:
   ```bash
   pip install torch numpy matplotlib tiktoken requests tensorflow
   ```
   Notes:
   - tensorflow is optional; it appears in some environments for experiments but is not required for the core training loop.

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

- notebook/: Jupyter notebooks for each major conceptual step in building an LLM from scratch.
- teh-verdict.txt: Sample text data used in earlier steps. In Step 4, the notebook automatically downloads "the-verdict.txt" (public domain) if not present.

---

## Details: Notebooks Folder

### notebook/step1.ipynb — Data Loading & Tokenization
- Downloads and reads `teh-verdict.txt` if not present, from an external URL.
- Loads the file’s contents into a string.
- Tokenizes the text using regex, splits and filters, then prepares a vocabulary (unique tokens) and maps each token to a unique integer ID.
- Sets up the dataset for training and experiment steps.
- Key Concepts: Data fetching, text preprocessing, vocabulary building.

### notebook/step2.ipynb — Attention Mechanisms
- Introduces the concept of attention in neural networks.
- Implements a simple self-attention mechanism (no trainable weights) using PyTorch tensors, computes dot products between input vectors and a query, and prints attention scores.
- Progresses to softmax normalization, attention weights, and context vectors.
- Builds toward more complex attention modules (including trainable weights, masking for causality, and modularization).
- Key Concepts: Self-attention, dot-product, softmax, context vectors, masking.

### notebook/step3.ipynb — From Attention to GPT
- Implements a GPT-like architecture from scratch.
- Defines a model config (vocab size, embedding size, context length, number of heads/layers, dropout, etc.).
- Implements transformer blocks and layer normalization in PyTorch.
- Assembles token embeddings, positional embeddings, dropout, transformer block stack, normalization, and output head.
- Provides a forward pass for model input through all layers and basic text generation utilities.
- Key Concepts: Transformer block structure, layer normalization, residual connections, full model assembly, text generation, model parameterization.

### notebook/step4.ipynb — Pretraining, Evaluation, and Decoding

This notebook extends the GPT implementation with data loading, training, evaluation, and sampling strategies. Open it here: https://github.com/vivek2437/LLM_Scratch/blob/main/notebook/step4.ipynb

#### 4.1 Evaluating generative text models
- Using GPT to generate text:
  - Utilities: `text_to_token_ids`, `token_ids_to_text`, and `generate_text_simple` (from `llm_gpt`).
  - Tokenizer: GPT-2 BPE via `tiktoken`.
  - Example prompt: "Every effort moves you".
- Calculating text generation loss: cross-entropy and perplexity:
  - Computes logits via forward pass of `GPTModel`.
  - Applies `torch.nn.functional.cross_entropy` on flattened logits/targets.
  - Computes perplexity as `torch.exp(loss)`.
- Calculating training and validation set losses:
  - Downloads dataset: "the-verdict.txt" (public domain) if missing.
  - Creates train/validation split (default ratio 80/20).
  - Builds dataloaders with `create_dataloader_V1` (from `text_data`) using context-length windows.
  - Reports shapes, token counts, and initial train/val loss (e.g., ~10.99 on untrained model), providing a sanity check.

#### 4.2 Training an LLM
- Model config used (GPT-2 small like):
  - `vocab_size=50257`, `context_length=256`, `emb_dim=768`, `n_heads=12`, `n_layers=12`, `drop_rate=0.1`, `qkv_bias=False`.
- Core functions:
  - `calc_loss_batch`, `calc_loss_loader` for computing cross-entropy over batches/loops.
  - `train_model_simple` for the training loop with periodic evaluation.
  - `evaluate_model` to compute train/val losses over a fixed number of batches.
  - `generate_and_print_sample` to print model samples after each epoch.
- Optimization:
  - `AdamW` with `lr=4e-4`, `weight_decay=0.1` (tunable).
  - Tracks tokens seen and steps for plotting.
- Outputs:
  - Per-epoch/step loss logs for train/val.
  - Periodic text samples showing improved fluency.
  - A loss plot saved to `loss-plot.pdf` (epochs vs loss with a twin x-axis for tokens seen).

#### 4.3 Decoding strategies to control randomness
- Greedy and multinomial sampling illustrated on model logits.
- Temperature scaling:
  - Demonstrates how scaling logits before softmax changes token selection randomness.
  - Includes a toy vocabulary example and frequency counts from repeated sampling to visualize the effect of temperature.

#### Running Step 4
- Launch Jupyter and open the notebook:
  ```bash
  jupyter notebook notebook/step4.ipynb
  # or
  jupyter lab notebook/step4.ipynb
  ```
- The notebook prints package versions, constructs the model, downloads the dataset if needed, trains for several epochs, prints sample generations, and saves `loss-plot.pdf`.

Notes
- The example uses a short context length (256) and a small dataset for speed; this is for educational purposes. Expect limited long-range coherence compared to large-scale pretraining.
- The notebook imports `llm_gpt` and `text_data`. Ensure these modules are available in your environment (e.g., placed in the repo or Python path).

---

## Usage

- Run notebooks:
  ```bash
  jupyter notebook
  ```
  Open and explore the notebooks in the `notebook/` directory.

- Data:
  The Step 4 notebook will fetch `the-verdict.txt` automatically if not present.

- Dependencies:
  Most notebooks require: `torch`, `numpy`, `tiktoken`, `matplotlib`, and Python standard libraries. Step 4 additionally uses `requests` to fetch text.

---

## Example: Tokenizing Text Data

```python
import os, urllib.request
if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    urllib.request.urlretrieve(url, "the-verdict.txt")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
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

## Example: Minimal GPT Model (toy)

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