import torch
from tokenizer import CharTokenizer


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4
BLOCK_SIZE = 32


# ============================================================
# Load text
# ============================================================

with open(
    "data/tea.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


# ============================================================
# Create tokenizer
# ============================================================

tokenizer = CharTokenizer(text)

# Encode complete dataset
data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)


# ============================================================
# Train / validation split
# ============================================================

split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]


print("Total tokens:", len(data))
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Vocabulary size:", tokenizer.vocab_size)


# ============================================================
# Create batch
# ============================================================

def get_batch(split):
    """
    Returns:
        x -> input tokens
        y -> target tokens
    """

    dataset = train_data if split == "train" else val_data

    # Random starting positions
    starts = torch.randint(
        0,
        len(dataset) - BLOCK_SIZE,
        (BATCH_SIZE,)
    )

    # Input sequences
    x = torch.stack([
        dataset[i:i + BLOCK_SIZE]
        for i in starts
    ])

    # Target sequences are shifted by 1
    y = torch.stack([
        dataset[i + 1:i + BLOCK_SIZE + 1]
        for i in starts
    ])

    return x, y


# ============================================================
# Test batch
# ============================================================

if __name__ == "__main__":

    x, y = get_batch("train")

    print("\nInput shape:")
    print(x.shape)

    print("\nTarget shape:")
    print(y.shape)

    print("\nInput tokens:")
    print(x[0])

    print("\nTarget tokens:")
    print(y[0])

    print("\nInput decoded:")
    print(tokenizer.decode(x[0].tolist()))

    print("\nTarget decoded:")
    print(tokenizer.decode(y[0].tolist()))