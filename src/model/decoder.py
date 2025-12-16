import torch
import torch.nn as nn
from .multihead_attention import MultiHeadAttention
from .feed_forward import FeedForward

class Decoder(nn.Module):
    def __init__(
        self, 
        model_dim, 
        num_heads, 
        ff_hidden_dim=2048, 
        dropout=0.1,
        use_rope: bool = False
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout, use_rope)
        self.cross_attn = MultiHeadAttention(model_dim, num_heads, dropout, use_rope)
        self.ffn = FeedForward(model_dim, ff_hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [B, T, D]
            enc_output: [B, L, D]
            src_mask: [B, 1, 1, L]
            tgt_mask: [B, 1, T, T]
        Returns:
            [B, T, D]
        """
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x2 = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(x2, enc_output, enc_output, src_mask)
        x3 = self.norm2(x2 + self.dropout(cross_attn_out))

        ffn_out = self.ffn(x3)
        x4 = self.norm3(x3 + self.dropout(ffn_out))

        return x4
