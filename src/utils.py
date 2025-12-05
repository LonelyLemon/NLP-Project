import torch
import torch.nn as nn


def create_padding_mask(seq, pad_idx):
    """
    Tạo mask để ignore padding tokens trong attention.
    
    Args:
        seq: [B, L] - Input sequence
        pad_idx: int - Index của padding token
        
    Returns:
        mask: [B, 1, 1, L] - Mask cho attention (1 = ignore, 0 = attend)
    """
    # [B, L] -> [B, 1, 1, L]
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(seq_len, device='cpu'):
    """
    Tạo causal mask để prevent attention to future tokens (cho decoder).
    
    Args:
        seq_len: int - Độ dài sequence
        device: str - Device để tạo tensor
        
    Returns:
        mask: [1, 1, seq_len, seq_len] - Upper triangular mask
    """
    # Tạo upper triangular matrix (trên đường chéo = 1)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    # [L, L] -> [1, 1, L, L]
    return mask.unsqueeze(0).unsqueeze(0)


def create_masks(src, tgt, pad_idx, device='cpu'):
    """
    Tạo tất cả masks cần thiết cho Transformer.
    
    Args:
        src: [B, S] - Source sequence
        tgt: [B, T] - Target sequence
        pad_idx: int - Padding token index
        device: str - Device
        
    Returns:
        src_mask: [B, 1, 1, S] - Padding mask cho encoder
        tgt_mask: [B, 1, T, T] - Combined padding + causal mask cho decoder
    """
    # Source padding mask
    src_mask = create_padding_mask(src, pad_idx)
    
    # Target padding mask
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)  # [B, 1, 1, T]
    
    # Target causal mask
    tgt_len = tgt.size(1)
    tgt_causal_mask = create_causal_mask(tgt_len, device)  # [1, 1, T, T]
    
    # Combine: padding mask OR causal mask
    tgt_mask = tgt_padding_mask | tgt_causal_mask  # [B, 1, T, T]
    
    return src_mask, tgt_mask


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filepath):
    """
    Lưu model checkpoint.
    
    Args:
        model: nn.Module - Model cần lưu
        optimizer: Optimizer
        epoch: int - Current epoch
        train_loss: float - Training loss
        val_loss: float - Validation loss
        filepath: str - Đường dẫn lưu file
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        filepath: str - Đường dẫn file checkpoint
        model: nn.Module - Model để load weights
        optimizer: Optimizer (optional) - Optimizer để load state
        device: str - Device
        
    Returns:
        epoch: int - Epoch đã train
        train_loss: float
        val_loss: float
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint loaded: {filepath}")
    print(f"   Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def count_parameters(model):
    """
    Đếm số lượng parameters của model.
    
    Args:
        model: nn.Module
        
    Returns:
        total: int - Tổng số parameters
        trainable: int - Số parameters có thể train
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
