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
        checkpoint_path='best_model.pt',
        log_path='history.json'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path

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
        for raw_src, src, raw_tgt, tgt in pbar:
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            src_mask, tgt_mask = create_masks(src, tgt_input, self.src_pad_idx, self.trg_pad_idx, self.device)

            self.optimizer.zero_grad()
            logits = self.model(src, tgt_input, src_mask, tgt_mask)  # [B, T-1, V]
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = self.criterion(logits, tgt_output)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("\nWarning: NaN or Inf loss detected! Skipping batch.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if warmup_scheduler is not None:
                warmup_scheduler.step()
            
            total_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
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
        print("Start training")
        total, trainable = count_parameters(self.model)
        print("Model parameters - Total:", total, "Trainable:", trainable)
        
        epochs_no_improve = 0
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch, warmup_scheduler)
            val_loss = self.validate(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(
                    self.model, 
                    self.optimizer,
                    self.checkpoint_path
                )
                print("New best model saved")
            else:
                epochs_no_improve += 1
                print("No improvement for", epochs_no_improve, "epoch(s)")

            if plateau_scheduler is not None:
                plateau_scheduler.step(val_loss)
            

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

        self.save_history()
        print("Training completed. Best Val Loss:", self.best_val_loss)
    
    def save_history(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        print("Training history saved at", self.log_path)


def create_optimizer(model, learning_rate=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def create_scheduler(optimizer, mode='plateau', factor=0.5, patience=3, min_lr=1e-6):
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
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self._update_lr()
    
    def step(self):
        self.current_step += 1
        self._update_lr()
    
    def _update_lr(self):
        step = max(self.current_step, 1)
        lr = (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
