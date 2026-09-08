import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

BLOCK_SIZE = 32
N_EMBED = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0


# ============================================================
# Causal Self-Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()

        assert n_embed % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embed // n_head

        # Create Q, K, V in one projection
        self.qkv = nn.Linear(
            n_embed,
            3 * n_embed
        )

        # Final projection
        self.proj = nn.Linear(
            n_embed,
            n_embed
        )

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )
            ).view(
                1,
                1,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        # ----------------------------------------------------
        # Create Q, K, V
        # ----------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # ----------------------------------------------------
        # Split into attention heads
        #
        # [B, T, C]
        #       ↓
        # [B, n_head, T, head_dim]
        # ----------------------------------------------------

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Attention scores
        #
        # QK^T / sqrt(d)
        # ----------------------------------------------------

        scores = q @ k.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------------------
        # Attention × V
        # ----------------------------------------------------

        out = attention_weights @ v

        # ----------------------------------------------------
        # Combine heads
        #
        # [B, n_head, T, head_dim]
        #       ↓
        # [B, T, C]
        # ----------------------------------------------------

        out = out.transpose(1, 2).contiguous()

        out = out.view(
            B,
            T,
            C
        )

        # Output projection
        out = self.proj(out)

        return out


# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                n_embed,
                4 * n_embed
            ),

            nn.GELU(),

            nn.Linear(
                4 * n_embed,
                n_embed
            )
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embed)

        self.attention = CausalSelfAttention(
            n_embed,
            n_head
        )

        self.ln2 = nn.LayerNorm(n_embed)

        self.ffn = FeedForward(
            n_embed
        )

    def forward(self, x):

        # Pre-LayerNorm + Attention + Residual
        x = x + self.attention(
            self.ln1(x)
        )

        # Pre-LayerNorm + FFN + Residual
        x = x + self.ffn(
            self.ln2(x)
        )

        return x


# ============================================================
# TeaLM
# ============================================================

class TeaLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size=BLOCK_SIZE,
        n_embed=N_EMBED,
        n_head=N_HEAD,
        n_layer=N_LAYER
    ):
        super().__init__()

        self.block_size = block_size

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embed
        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            block_size,
            n_embed
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embed,
                    n_head
                )
                for _ in range(n_layer)
            ]
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.ln_f = nn.LayerNorm(
            n_embed
        )

        # ----------------------------------------------------
        # Language-model head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            n_embed,
            vocab_size,
            bias=False
        )

        # ----------------------------------------------------
        # Initialize weights
        # ----------------------------------------------------

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        assert T <= self.block_size

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_emb = self.token_embedding(idx)

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            T,
            device=idx.device
        )

        position_emb = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        x = token_emb + position_emb

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.blocks(x)

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.ln_f(x)

        # ----------------------------------------------------
        # Convert to vocabulary logits
        # ----------------------------------------------------

        logits = self.lm_head(x)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits_flat = logits.view(
                B * T,
                C
            )

            targets_flat = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    # ========================================================
    # Generate tokens
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0
    ):

        for _ in range(max_new_tokens):

            # Keep only the last block_size tokens
            idx_cond = idx[
                :, -self.block_size:
            ]

            # Get predictions
            logits, _ = self(
                idx_cond
            )

            # Last token prediction
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Convert to probabilities
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append token
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx


# ============================================================
# Test Model
# ============================================================

if __name__ == "__main__":

    from tokenizer import CharTokenizer

    # Load Tea text
    with open(
        "data/tea.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # Create tokenizer
    tokenizer = CharTokenizer(text)

    # Create model
    model = TeaLM(
        vocab_size=tokenizer.vocab_size
    )

    # Print model
    print(model)

    # Number of parameters
    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\nParameters:",
        parameters
    )

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    sample = "Tea is"

    tokens = tokenizer.encode(sample)

    x = torch.tensor(
        [tokens],
        dtype=torch.long
    )

    print("\nInput:")
    print(sample)

    print("\nInput shape:")
    print(x.shape)

    # Forward pass
    logits, loss = model(x)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nLoss:")
    print(loss)

    # --------------------------------------------------------
    # Test generation
    # --------------------------------------------------------

    generated = model.generate(
        x,
        max_new_tokens=50
    )

    generated_text = tokenizer.decode(
        generated[0].tolist()
    )

    print("\nGenerated:")
    print(generated_text)