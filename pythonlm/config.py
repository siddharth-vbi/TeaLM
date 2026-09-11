# ============================================================
# All hyperparameters in one place.
# Change values here — model, dataset, and scripts pick them up automatically.
# ============================================================

# ----- Model ------------------------------------------------
VOCAB_SIZE = 8000     # tokenizer vocabulary size
BLOCK_SIZE = 64       # context window (tokens)
N_EMBED    = 128      # embedding dimension
N_HEAD     = 4    # attention heads  (N_EMBED must be divisible by N_HEAD)
N_LAYER    = 2       # transformer blocks
DROPOUT    = 0.1     # dropout rate

# ----- Training ---------------------------------------------
BATCH_SIZE     = 8
MAX_ITERS      = 10000
EVAL_INTERVAL  = 500
LEARNING_RATE  = 3e-4
WEIGHT_DECAY   = 0.1
PATIENCE       = 3    # early-stopping patience (eval steps)

# ----- Paths ------------------------------------------------
DATA_PATH       = "data/processed/python_train.txt"
TOKENIZER_PATH  = "data/bpe_tokenizer.json"
CHECKPOINT_DIR  = "checkpoints"
BEST_CHECKPOINT = "checkpoints/pythonlm_best.pt"
LAST_CHECKPOINT = "checkpoints/pythonlm_last.pt"
