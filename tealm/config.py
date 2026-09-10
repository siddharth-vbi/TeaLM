# ============================================================
# All hyperparameters in one place.
# Change values here — model, dataset, and scripts pick them up automatically.
# ============================================================

# ----- Model ------------------------------------------------
BLOCK_SIZE = 64      # context window (tokens)
N_EMBED    = 64      # embedding dimension
N_HEAD     = 2    # attention heads  (N_EMBED must be divisible by N_HEAD)
N_LAYER    = 1       # transformer blocks
DROPOUT    = 0.1     # dropout rate

# ----- Training ---------------------------------------------
BATCH_SIZE     = 4
MAX_ITERS      = 10000
EVAL_INTERVAL  = 500
LEARNING_RATE  = 3e-4
WEIGHT_DECAY   = 0.1
PATIENCE       = 3    # early-stopping patience (eval steps)

# ----- Paths ------------------------------------------------
DATA_PATH       = "data/tea.txt"
TOKENIZER_PATH  = "data/bpe_tokenizer.json"
CHECKPOINT_DIR  = "checkpoints"
BEST_CHECKPOINT = "checkpoints/tealm_best.pt"
LAST_CHECKPOINT = "checkpoints/tealm_last.pt"
