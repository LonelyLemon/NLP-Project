import torch
import torch.nn as nn
from .feed_forward import FeedForward
from .multihead_attention import MultiHeadAttention
from .swiglu import SwiGLU
from typing import Literal


class Encoder(nn.Module):
    def __init__(
        self, 
        model_dim: int, 
        num_heads: int, 
        ff_hidden_dim=2048, 
        dropout=0.1,
        use_rope: bool = False,
        use_swig: bool = False
    ):
        super().__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads, dropout, use_rope)
        if use_swig:
            self.ffn = SwiGLU(model_dim=model_dim, hidden_dim=ff_hidden_dim)
        else:
            self.ffn = FeedForward(model_dim, ff_hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # [B, L, D]
        x_norm = self.norm1(x)
        attn_out = self.mha(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_out)

        # [B, L, D]
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return ffn_out