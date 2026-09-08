import json


class CharTokenizer:
    def __init__(self, text=None):
        self.stoi = {}
        self.itos = {}

        if text is not None:
            self.build_vocab(text)

    # --------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------
    def build_vocab(self, text):
        # Get every unique character
        chars = sorted(list(set(text)))

        # Add special token
        chars = ["<UNK>"] + chars

        # character -> integer
        self.stoi = {
            ch: i
            for i, ch in enumerate(chars)
        }

        # integer -> character
        self.itos = {
            i: ch
            for i, ch in enumerate(chars)
        }

    # --------------------------------------------------
    # Vocabulary size
    # --------------------------------------------------
    @property
    def vocab_size(self):
        return len(self.stoi)

    # --------------------------------------------------
    # Encode text -> token IDs
    # --------------------------------------------------
    def encode(self, text):
        unk_id = self.stoi["<UNK>"]

        return [
            self.stoi.get(char, unk_id)
            for char in text
        ]

    # --------------------------------------------------
    # Decode token IDs -> text
    # --------------------------------------------------
    def decode(self, tokens):
        return "".join(
            self.itos[token]
            for token in tokens
        )

    # --------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------
    def save(self, path):
        data = {
            "stoi": self.stoi,
            "itos": {
                str(k): v
                for k, v in self.itos.items()
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------
    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()

        tokenizer.stoi = data["stoi"]

        tokenizer.itos = {
            int(k): v
            for k, v in data["itos"].items()
        }

        return tokenizer


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    # Load TeaLM dataset
    with open(
        "data/tea.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # Create tokenizer
    tokenizer = CharTokenizer(text)

    print("Vocabulary size:", tokenizer.vocab_size)

    print("\nVocabulary:")
    print(tokenizer.stoi)

    # Test encoding
    sample = "Tea is made"

    encoded = tokenizer.encode(sample)

    print("\nOriginal:")
    print(sample)

    print("\nEncoded:")
    print(encoded)

    # Test decoding
    decoded = tokenizer.decode(encoded)

    print("\nDecoded:")
    print(decoded)

    # Verify
    assert sample == decoded

    print("\nTokenizer test: PASSED")

    # Save tokenizer
    tokenizer.save("data/tokenizer.json")

    print("\nTokenizer saved to:")
    print("data/tokenizer.json")