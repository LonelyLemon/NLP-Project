import torch
import torch.nn as nn


def create_padding_mask(seq, pad_idx):
    # [B, L] -> [B, 1, 1, L]
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()


def create_causal_mask(seq_len, device='cpu'):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0).float()


def create_masks(src, tgt, src_pad_idx, trg_pad_idx, device='cpu'):
    src_mask = create_padding_mask(src, src_pad_idx)
    tgt_padding_mask = create_padding_mask(tgt, trg_pad_idx)  # [B, 1, 1, T]
    tgt_len = tgt.size(1)
    tgt_causal_mask = create_causal_mask(tgt_len, device)  # [1, 1, T, T]
    tgt_padding_mask = tgt_padding_mask.expand(-1, -1, tgt_len, -1)
    tgt_mask = tgt_padding_mask * tgt_causal_mask  # [B, 1, T, T]
    
    return src_mask, tgt_mask


def save_checkpoint(model, optimizer, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded: {filepath}")
    return checkpoint

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
