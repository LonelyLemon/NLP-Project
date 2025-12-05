import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(
            self, 
            model_dim: int, 
            hidden_dim: int = 2048, 
            dropout: float = 0.1
        ):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        hidden = self.activation(self.linear1(x))
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)
        return output
