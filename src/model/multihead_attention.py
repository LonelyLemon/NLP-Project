import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        model_dim: int, 
        num_heads: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "embedding_dim phải chia hết cho num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.Q_linear = nn.Linear(model_dim, model_dim)
        self.K_linear = nn.Linear(model_dim, model_dim)
        self.V_linear = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [B, L, D]
            k: [B, L, D]
            v: [B, L, D]
        Returns:
            context: [B, L, D]
        """
        B, L, D = q.size()

        # [B, L, D]
        Q = self.Q_linear(q)
        K = self.K_linear(k)
        V = self.V_linear(v)

        def split_heads(x):
            # [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
            return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # [B, num_heads, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # [B, H, L, head_dim]
        context = torch.matmul(attn_weights, V)

        # [B, L, D]
        context = context.transpose(1,2).contiguous().view(B, L, D)
        out = self.out_proj(context)
        return out
