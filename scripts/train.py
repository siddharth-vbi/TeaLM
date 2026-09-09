"""Train TeaLM and save the best checkpoint.

Usage:
    python scripts/train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tealm import config
from tealm.model import TeaLM
from tealm.dataset import get_batch, tokenizer

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Setup
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device     : {device}")

model = TeaLM(vocab_size=tokenizer.get_vocab_size()).to(device)
print(f"Parameters : {model.num_parameters():,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.MAX_ITERS
)


# ============================================================
# Loss estimation
# ============================================================

@torch.no_grad()
def estimate_loss(n_batches=20):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        total = 0.0
        for _ in range(n_batches):
            x, y = get_batch(split)
            _, loss = model(x.to(device), y.to(device))
            total += loss.item()
        losses[split] = total / n_batches
    model.train()
    return losses


# ============================================================
# Training loop
# ============================================================

best_val_loss    = float("inf")
patience_counter = 0

for iteration in range(config.MAX_ITERS):
    x, y = get_batch("train")
    _, loss = model(x.to(device), y.to(device))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if iteration % config.EVAL_INTERVAL == 0:
        losses = estimate_loss()
        lr     = scheduler.get_last_lr()[0]
        print(
            f"step {iteration:>4d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f} | "
            f"lr {lr:.6f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss    = losses["val"]
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_CHECKPOINT)
            print(f"  → best model saved  ({config.BEST_CHECKPOINT})")
        else:
            patience_counter += 1
            print(f"  → no improvement ({patience_counter}/{config.PATIENCE})")
            if patience_counter >= config.PATIENCE:
                print("\nEarly stopping triggered.")
                break

# ============================================================
# Save final model
# ============================================================

torch.save(model.state_dict(), config.LAST_CHECKPOINT)
print(f"\nFinal model saved → {config.LAST_CHECKPOINT}")
