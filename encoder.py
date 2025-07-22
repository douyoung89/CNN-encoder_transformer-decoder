import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Helper function for weight initialization
def init_weights(module):
    """
    Initializes weights of linear and embedding layers.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class SelfAttention(nn.Module):
    """
    A standard (non-causal) self-attention module.
    Each token can attend to all other tokens in the sequence.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimension (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Self-attention (dot product attention)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        # No masking applied here, as this is a non-causal self-attention for an encoder.
        # All tokens can attend to all other tokens.

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class EncoderBlock(nn.Module):
    """
    A single encoder block consisting of a self-attention layer and a feed-forward network.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config) # Use the non-causal SelfAttention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(), # Using GELU as per TrAISformer's Block
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Encoder(nn.Module):
    """
    The main Encoder model for processing input sequences.
    It takes only the density feature, embeds it, adds positional information,
    and processes it through a stack of EncoderBlocks.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input embedding for only the density feature
        self.density_emb = nn.Embedding(config.density_size, config.n_embd)

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # Encoder blocks
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])

        # Apply weight initialization
        self.apply(init_weights)

        # Report number of parameters
        print(f"Number of parameters in Encoder: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def forward(self, density): # Only density as input
        """
        Forward pass for the Encoder.

        Args:
            density (torch.Tensor): Density feature sequence (batch_size, seq_len)

        Returns:
            torch.Tensor: The encoded representation of the input sequence (batch_size, seq_len, n_embd)
        """
        device = density.device
        b, t = density.size() # Batch size, sequence length
        assert t <= self.config.max_seqlen+1, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.max_seqlen}"

        # Embeddings for density feature
        token_embeddings = self.density_emb(density) # Only density embedding

        # Positional embeddings
        position_embeddings = self.pos_emb[:, :t, :] # Each position maps to a (learnable) embedding vector
        x = self.drop(token_embeddings + position_embeddings)

        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x)

        return x