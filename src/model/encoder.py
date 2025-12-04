import torch
import torch.nn as nn
from .feed_forward import FeedForward
from .multihead_attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(
        self, 
        model_dim: int, 
        num_heads: int, 
        ff_hidden_dim=2048, 
        dropout=0.1
    ):
        super().__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads, dropout)
        self.ffn = FeedForward(model_dim, ff_hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # [B, L, D]
        attn_out = self.mha(x, x, x, mask)
        attn_out = self.norm1(x + self.dropout(attn_out))

        # [B, L, D]
        ffn_out = self.ffn(attn_out)
        ffn_out = self.norm2(x + self.dropout(ffn_out))

        return ffn_out