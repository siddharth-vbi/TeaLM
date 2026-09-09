from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tealm import config


def train_tokenizer():
    """Train a character-level BPE tokenizer on the dataset and save it.

    No pre-tokenizer is used — the BPE model learns directly from raw
    characters, so spaces are just another character in the vocabulary.
    This means tokenizer.decode() concatenates tokens with no separator
    and naturally reproduces the original spacing.
    """
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    # No pre_tokenizer → BPE sees every character including spaces

    trainer = BpeTrainer(
        vocab_size=1000,
        min_frequency=2,
        special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"],
    )

    tokenizer.train([config.DATA_PATH], trainer)
    tokenizer.save(config.TOKENIZER_PATH)
    print(f"Tokenizer saved → {config.TOKENIZER_PATH}")
    print(f"Vocabulary size : {tokenizer.get_vocab_size()}")


def load_tokenizer():
    """Load a previously trained tokenizer from disk."""
    return Tokenizer.from_file(config.TOKENIZER_PATH)


def decode(tokenizer, ids):
    """Decode a list of token IDs back to a string.

    HuggingFace's built-in tokenizer.decode() inserts spaces between every
    token.  For character-level BPE (no pre-tokenizer), spaces are already
    baked into the token strings — so we join with no separator instead.
    """
    special = {"<UNK>", "<PAD>", "<BOS>", "<EOS>"}
    return "".join(
        t for t in (tokenizer.id_to_token(i) for i in ids)
        if t is not None and t not in special
    )


