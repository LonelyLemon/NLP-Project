import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from pathlib import Path
from src.utils import create_masks, save_checkpoint, count_parameters


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        src_pad_idx,
        trg_pad_idx,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch, warmup_scheduler=None):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (raw_src, src, raw_tgt, tgt) in enumerate(pbar):
            src = src.to(self.device)  # [B, S]
            tgt = tgt.to(self.device)  # [B, T]
            
            # Decoder input: bỏ token cuối (<eos>)
            tgt_input = tgt[:, :-1]  # [B, T-1]
            
            # Target output: bỏ token đầu (<sos>)
            tgt_output = tgt[:, 1:]  # [B, T-1]
            
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input, self.src_pad_idx, self.trg_pad_idx, self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(src, tgt_input, src_mask, tgt_mask)  # [B, T-1, V]
            
            # Calculate loss
            # Reshape: [B, T-1, V] -> [B * T-1, V]
            #          [B, T-1] -> [B * T-1]
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = self.criterion(logits, tgt_output)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ Warning: NaN/Inf loss detected! Skipping batch.")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update warmup scheduler (per batch)
            if warmup_scheduler is not None:
                warmup_scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar with current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """
        Validate model.
        
        Returns:
            avg_loss: float - Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]  ')
        for raw_src, src, raw_tgt, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = create_masks(src, tgt_input, self.src_pad_idx, self.trg_pad_idx, self.device)
            
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = self.criterion(logits, tgt_output)
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs, warmup_scheduler=None, plateau_scheduler=None, patience=5):
        print("\n" + "="*60)
        print("🚀 BẮT ĐẦU TRAINING")
        print("="*60)
        
        total, trainable = count_parameters(self.model)
        print(f"📊 Model Parameters:")
        print(f"   Total: {total:,}")
        print(f"   Trainable: {trainable:,}")
        print("="*60 + "\n")
        
        epochs_no_improve = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch, warmup_scheduler)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            
            # Print summary
            print(f"\n📈 Epoch {epoch}/{num_epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   LR:         {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_no_improve = 0
                
                checkpoint_path = self.checkpoint_dir / 'best_model.pt'
                save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    train_loss, 
                    val_loss, 
                    checkpoint_path
                )
                print(f"   ✨ New best model! (Val Loss: {val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"   ⏳ No improvement for {epochs_no_improve} epoch(s)")
            
            # Learning rate scheduling (plateau scheduler after warmup)
            if plateau_scheduler is not None:
                plateau_scheduler.step(val_loss)
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs.")
                break
            
            # Save checkpoint mỗi 5 epochs
            if epoch % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_loss,
                    val_loss,
                    checkpoint_path
                )
            
            print("-" * 60 + "\n")
        
        # Save training history
        self.save_history()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED!")
        print(f"📊 Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")
    
    def save_history(self):
        """Lưu training history vào JSON file."""
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        print(f"📝 Training history saved: {history_path}")


def create_optimizer(model, learning_rate=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4):
    """
    Tạo Adam optimizer với hyperparameters chuẩn cho Transformer.
    
    Args:
        model: nn.Module
        learning_rate: float
        betas: tuple - Adam beta parameters
        eps: float - Epsilon for numerical stability
        weight_decay: float - L2 regularization
        
    Returns:
        optimizer: torch.optim.AdamW
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def create_scheduler(optimizer, mode='plateau', factor=0.5, patience=3, min_lr=1e-6):
    """
    Tạo learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        mode: str - 'plateau' hoặc 'step'
        factor: float - Factor giảm learning rate
        patience: int - Số epochs chờ trước khi giảm LR
        min_lr: float - Minimum learning rate
        
    Returns:
        scheduler: Learning rate scheduler
    """
    if mode == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
    elif mode == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=factor
        )
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")


class WarmupScheduler:
    """
    Learning rate scheduler with warmup.
    Implements the schedule from "Attention is All You Need" paper.
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        """
        Args:
            optimizer: Optimizer
            d_model: int - Model dimension
            warmup_steps: int - Number of warmup steps
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self._update_lr()
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        self._update_lr()
    
    def _update_lr(self):
        """Calculate and update learning rate."""
        step = max(self.current_step, 1)  # Avoid division by zero
        lr = (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
