import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.register_buffer(
            'inv_freq',
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        )

    def forward(self, x):
        B, L, D = x.shape
        device = x.device
        positions = torch.arange(L, device=device).float()
        theta = torch.einsum('n,d->nd', positions, self.inv_freq)
        cos = theta.cos()[None, :, :]
        sin = theta.sin()[None, :, :]
        x_reshaped = x.view(B, L, D//2, 2)
        x_even = x_reshaped[...,0]
        x_odd  = x_reshaped[...,1]
        cos = cos.expand(B, -1, -1)
        sin = sin.expand(B, -1, -1)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return x_rot


if __name__ == "__main__":
    B, L, D = 2, 5, 8
    x = torch.randn(B, L, D)
    print("Input x[0]:\n", x[0])

    rope = RoPE(D)
    x_rot = rope(x)
    print("Rotated x[0]:\n", x_rot[0])
