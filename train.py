import torch
from model import TeaLM
from tokenizer import CharTokenizer
from dataset import get_batch, tokenizer


# ============================================================
# Configuration
# ============================================================

MAX_ITERS = 5000
EVAL_INTERVAL = 500

LEARNING_RATE = 3e-4

# Early stopping
PATIENCE = 3


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ============================================================
# Create model
# ============================================================

model = TeaLM(
    vocab_size=tokenizer.vocab_size
).to(device)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Estimate loss
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    losses = {}

    for split in ["train", "val"]:

        total_loss = 0.0

        for _ in range(20):

            x, y = get_batch(split)

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            total_loss += loss.item()

        losses[split] = total_loss / 20

    model.train()

    return losses


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")
patience_counter = 0


for iteration in range(MAX_ITERS):

    # Get batch
    x, y = get_batch("train")

    x = x.to(device)
    y = y.to(device)

    # Forward pass
    logits, loss = model(x, y)

    # Clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Print progress
    if iteration % EVAL_INTERVAL == 0:

        losses = estimate_loss()
        
        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"step {iteration}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}"
        )
        
        # ====================================================
        # Early stopping
        # ====================================================

        if val_loss < best_val_loss:

            # Validation loss improved
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save(
                model.state_dict(),
                "teaLM_best.pt"
            )

            print("  → validation improved, model saved")

        else:

            # Validation loss did not improve
            patience_counter += 1

            print(
                f"  → no improvement "
                f"({patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print("Early stopping triggered!")

                break



# ============================================================
# Save model
# ============================================================

torch.save(
    model.state_dict(),
    "teaLM.pt"
)

print("Model saved to teaLM.pt")