import torch
import torch.nn as nn
from .position_encoding import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder
import math

class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        tgt_vocab_size, 
        model_dim=512, 
        num_heads=8, 
        num_enc_layers=6, 
        num_dec_layers=6, 
        ff_hidden_dim=2048, 
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()
        self.model_dim = model_dim
        
        self.src_embedding = nn.Embedding(src_vocab_size, model_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            Encoder(model_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_enc_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            Decoder(model_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_dec_layers)
        ])
        
        self.output_proj = nn.Linear(model_dim, tgt_vocab_size)
    
    def encode(self, src, src_mask=None):
        """
        Args:
            src: [B, S]
            src_mask: [B, 1, 1, S]
        Returns: [B, S, D]
        """
        x = self.src_embedding(src) * math.sqrt(self.model_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            tgt: [B, T]
            enc_output: [B, L, D]
            src_mask: [B, 1, 1, S]
            tgt_mask: [B, 1, T, T]
        Returns: [B, T, D]
        """
        x = self.tgt_embedding(tgt) * math.sqrt(self.model_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: [B, L]
            tgt: [B, T]
        Returns: [B, T, tgt_vocab_size]
        """

        # [B, S, D]
        enc_output = self.encode(src, src_mask) 

        # [B, T, D]
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)

        # [B, T, V]
        logits = self.output_proj(dec_output) 
        return logits