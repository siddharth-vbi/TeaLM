import torch
from tealm import config
from tealm.tokenizer import load_tokenizer


# ============================================================
# Tokenizer & data loading
# ============================================================

tokenizer = load_tokenizer()

with open(config.DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text).ids
data   = torch.tensor(tokens, dtype=torch.long)

# 90 / 10 train-val split
_split     = int(0.9 * len(data))
train_data = data[:_split]
val_data   = data[_split:]

print(f"Total tokens    : {len(data):,}")
print(f"Training tokens : {len(train_data):,}")
print(f"Val tokens      : {len(val_data):,}")
print(f"Vocabulary size : {tokenizer.get_vocab_size():,}")


# ============================================================
# Batch sampler
# ============================================================

def get_batch(split):
    """Return a random (x, y) batch from the requested split.

    x  — input token sequences  [BATCH_SIZE, BLOCK_SIZE]
    y  — target token sequences [BATCH_SIZE, BLOCK_SIZE]  (shifted by 1)
    """
    dataset = train_data if split == "train" else val_data
    starts  = torch.randint(0, len(dataset) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([dataset[i     : i + config.BLOCK_SIZE    ] for i in starts])
    y = torch.stack([dataset[i + 1 : i + config.BLOCK_SIZE + 1] for i in starts])
    return x, y
