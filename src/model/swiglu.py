import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()
        h = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(model_dim, h, bias=False)  # gate
        self.w2 = nn.Linear(model_dim, h, bias=False)  # data
        self.w3 = nn.Linear(h, model_dim, bias=False)  # output

    def forward(self, x: torch.Tensor):
        gate = F.silu(self.w1(x))
        data = self.w2(x)
        return self.w3(gate * data)
