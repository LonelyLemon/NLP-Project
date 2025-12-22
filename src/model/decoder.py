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
        self.cross_attn = MultiHeadAttention(model_dim, num_heads, dropout, False)
        self.ffn = FeedForward(model_dim, ff_hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x_norm = self.norm1(x)
        self_attn_out = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout(self_attn_out)

        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attn(x_norm, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_out)

        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x
