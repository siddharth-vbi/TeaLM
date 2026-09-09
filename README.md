# TeaLM

A small transformer language model trained on tea recipes.  
Uses **BPE tokenization**, **RoPE positional embeddings**, and **causal self-attention**.

## Setup

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Usage

All scripts are run from the **project root**.

### 1 — Train the tokenizer
```bash
python scripts/train_tokenizer.py
```

### 2 — Train the model
```bash
python scripts/train.py
```

### 3 — Generate text
```bash
python scripts/generate.py
python scripts/generate.py --prompt "How to make green tea" --tokens 150
python scripts/generate.py --temperature 0.5 --top_k 10
```

## Configuration

All hyperparameters live in [`tealm/config.py`](tealm/config.py). Change values there — every script picks them up automatically.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_EMBED` | 64 | Embedding dimension |
| `N_HEAD` | 2 | Attention heads |
| `N_LAYER` | 1 | Transformer layers |
| `BLOCK_SIZE` | 64 | Context window (tokens) |
| `DROPOUT` | 0.3 | Dropout rate |
| `BATCH_SIZE` | 4 | Training batch size |
| `MAX_ITERS` | 5000 | Max training steps |
| `LEARNING_RATE` | 3e-4 | AdamW learning rate |
| `PATIENCE` | 3 | Early stopping patience |

## Project Structure

```
TeaLM/
├── tealm/                   # core package
│   ├── config.py            # all hyperparameters
│   ├── model.py             # TeaLM transformer (RoPE + causal attention)
│   ├── dataset.py           # data loading and batch sampler
│   └── tokenizer.py         # BPE tokenizer (train + load)
├── scripts/
│   ├── train_tokenizer.py   # step 1: train the tokenizer
│   ├── train.py             # step 2: train the model
│   └── generate.py          # step 3: generate text
├── data/
│   ├── tea.txt              # training corpus
│   └── bpe_tokenizer.json   # saved tokenizer
├── checkpoints/             # saved model weights (.pt)
└── requirements.txt
```

## Dataset Size Guide

| Tokens | Recommended Config |
|--------|-------------------|
| < 10K | `N_EMBED=32, N_HEAD=1, N_LAYER=1` |
| 50K–200K | `N_EMBED=64, N_HEAD=2, N_LAYER=2` ← current |
| 500K+ | `N_EMBED=128, N_HEAD=4, N_LAYER=4` |
