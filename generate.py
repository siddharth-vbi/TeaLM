import torch

from model import TeaLM
from tokenizer import CharTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "teaLM_best.pt"
PROMPT = "Add tea"
MAX_NEW_TOKENS = 200

TEMPERATURE = 0.8


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ============================================================
# Load dataset text
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


# ============================================================
# Create model
# ============================================================

model = TeaLM(
    vocab_size=tokenizer.vocab_size
).to(device)


# ============================================================
# Load trained weights
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# Encode prompt
# ============================================================

tokens = tokenizer.encode(PROMPT)

x = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device
)


# ============================================================
# Generate
# ============================================================

generated = model.generate(
    x,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE
)


# ============================================================
# Decode
# ============================================================

output = tokenizer.decode(
    generated[0].tolist()
)


# ============================================================
# Print
# ============================================================

print("\nPrompt:")
print(PROMPT)

print("\nGenerated:")
print(output)