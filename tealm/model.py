import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tealm import config


# ============================================================
# Rotary Position Embedding (RoPE)
# ============================================================

def apply_rope(x):
    """Apply rotary position embeddings to query or key tensors.

    Args:
        x: [B, heads, T, head_dim]
    Returns:
        Rotated tensor of same shape.
    """
    B, H, T, D = x.shape
    half = D // 2
    device = x.device

    freq = 1.0 / (
        10000 ** (torch.arange(half, device=device, dtype=torch.float32) / half)
    )
    angles = torch.arange(T, device=device, dtype=torch.float32)[:, None] * freq[None, :]

    cos = torch.cos(angles)[None, None, :, :]   # [1, 1, T, half]
    sin = torch.sin(angles)[None, None, :, :]

    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ============================================================
# Causal Self-Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        assert n_embed % n_head == 0

        self.n_head  = n_head
        self.head_dim = n_embed // n_head

        self.qkv          = nn.Linear(n_embed, 3 * n_embed)
        self.proj         = nn.Linear(n_embed, n_embed)
        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular, registered as a non-parameter buffer
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
              .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE)
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape to [B, heads, T, head_dim]
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Apply RoPE to queries and keys
        q, k = apply_rope(q), apply_rope(k)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))

        # Weighted sum of values → reshape back to [B, T, C]
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.ln1       = nn.LayerNorm(n_embed)
        self.attention = CausalSelfAttention(n_embed, n_head, dropout)
        self.ln2       = nn.LayerNorm(n_embed)
        self.ffn       = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))  # attention  + residual
        x = x + self.ffn(self.ln2(x))        # FFN        + residual
        return x


# ============================================================
# TeaLM
# ============================================================

class TeaLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, config.N_EMBED)

        self.blocks = nn.Sequential(
            *[TransformerBlock(config.N_EMBED, config.N_HEAD, config.DROPOUT)
              for _ in range(config.N_LAYER)]
        )

        self.ln_f    = nn.LayerNorm(config.N_EMBED)
        self.lm_head = nn.Linear(config.N_EMBED, vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    # ----------------------------------------------------------
    # Weight initialization
    # ----------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > config.BLOCK_SIZE:
            raise ValueError(
                f"Sequence length {T} exceeds BLOCK_SIZE {config.BLOCK_SIZE}"
            )

        x = self.token_embedding(idx)   # [B, T, N_EMBED]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)        # [B, T, vocab_size]

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    # ----------------------------------------------------------
    # Autoregressive generation
    # ----------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
