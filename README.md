# GPT from Scratch

A clean PyTorch implementation of a Generative Pre-trained Transformer (GPT) built from scratch, following Andrej Karpathy's tutorial *"Let's build GPT: from scratch, in code, spelled out"*.

## 📌 Overview

This repository contains a step-by-step implementation of a decoder-only Transformer architecture trained on character-level text (Tiny Shakespeare dataset) to generate Shakespeare-like text. 

### Key Components Implemented:
* **Token & Positional Embeddings:** Mapping characters and their positions into dense vector representations.
* **Self-Attention & Multi-Head Attention:** Scaled dot-product attention with causal masking (tril) so tokens only attend to previous context.
* **Transformer Blocks:** Residual connections, Layer Normalization, and Feed-Forward Networks (MLP).
* **Autoregressive Generation:** Sampling new tokens sequentially using softmax probabilities.

## 🛠 Tech Stack
* **Python 3.8+**
* **PyTorch**

## 📂 Project Structure

```text
GPT_from_scratch/
├── input.txt          # Training dataset (e.g., Tiny Shakespeare)
├── bigram.py              # Baseline Bigram language model
├── v2.py                 # Full Transformer (GPT) architecture & training loop
├── code.ipynb           #Some records for development
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com
cd GPT_from_scratch
```

### 2. Install dependencies
```bash
pip install torch
```

### 3. Run training and generation
```bash
python gpt.py
```

## ⚙️ Model Hyperparameters
* **Batch Size:** `64`
* **Block Size (Context Length):** `256`
* **Embedding Dimension (`n_embd`):** `384`
* **Attention Heads (`n_head`):** `6`
* **Transformer Layers (`n_layer`):** `6`
* **Dropout:** `0.2`

## 🙏 Acknowledgements
* Based on the lecture and code by [Andrej Karpathy](https://github.com).
* Inspired by the original Transformer paper: *"Attention Is All You Need"* (Vaswani et al., 2017).
