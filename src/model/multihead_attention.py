import torch
import torch.nn as nn
import math
from .rope import RoPE

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        model_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "embedding_dim phải chia hết cho num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.use_rope = use_rope

        if use_rope:
            self.rope = RoPE(self.head_dim)

        self.Q_linear = nn.Linear(model_dim, model_dim)
        self.K_linear = nn.Linear(model_dim, model_dim)
        self.V_linear = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [B, L_q, D]
            k: [B, L_k, D]
            v: [B, L_v, D]
        Returns:
            context: [B, L_q, D]
        """
        B = q.size(0)
        L_q = q.size(1)
        L_k = k.size(1)
        L_v = v.size(1)

        # Linear projections: [B, L, D]
        Q = self.Q_linear(q)
        K = self.K_linear(k)
        V = self.V_linear(v)

        # Split heads: [B, L, D] -> [B, num_heads, L, head_dim]
        Q = Q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            B, H, L, D = Q.shape
            Q = self.rope(Q.reshape(B * H, L, D)).reshape(B, H, L, D)
            K = self.rope(K.reshape(B * H, L, D)).reshape(B, H ,L, D)

        # [B, num_heads, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # [B, H, L_q, head_dim]
        context = torch.matmul(attn_weights, V)

        # [B, L_q, D]
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.model_dim)
        out = self.out_proj(context)
        return out
