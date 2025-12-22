import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from pathlib import Path
from src.utils import create_masks, save_checkpoint, count_parameters, create_padding_mask, create_causal_mask
from sacrebleu.metrics import BLEU

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        src_vocab,
        trg_vocab,
        max_tgt_len,
        checkpoint_path='best_model.pt',
        log_path='history.json'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.src_vocab = src_vocab
        self.tgt_vocab = trg_vocab
        # self.src_pad_idx = src_pad_idx
        # self.trg_pad_idx = trg_pad_idx
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.max_tgt_len = max_tgt_len
        self.bleu_metric = BLEU(tokenize='13a', lowercase=True, smooth_method='exp')

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_bleu': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_bleu = -float('inf')
        
    def train_epoch(self, epoch, warmup_scheduler=None):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for raw_src, src, raw_tgt, tgt in pbar:
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            src_mask, tgt_mask = create_masks(
                src, 
                tgt_input, 
                self.src_vocab.pad_idx, 
                self.tgt_vocab.pad_idx, 
                self.device
            )

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
    def greedy_decode_batch(self, src, max_len=100):
        self.model.eval()
        B = src.size(0)
        device = src.device
        pad_idx = self.src_vocab.pad_idx
        sos_idx = self.tgt_vocab.sos_idx
        eos_idx = self.tgt_vocab.eos_idx

        src_mask = create_padding_mask(src, pad_idx).to(device)
        enc_output = self.model.encode(src, src_mask)
        ys = torch.full(
            (B, 1),
            sos_idx,
            dtype=torch.long,
            device=device
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            tgt_mask = create_causal_mask(ys.size(1), device)  # [1, T, T] hoặc [T, T]

            dec_output = self.model.decode(
                ys, enc_output, src_mask, tgt_mask
            )  # [B, T, D]

            logits = self.model.output_proj(dec_output[:, -1, :])  # [B, V]
            next_tokens = logits.argmax(dim=-1)                     # [B]

            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)  # [B, T+1]

            finished |= (next_tokens == eos_idx)
            if finished.all():
                break

        return ys.tolist()

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_hypotheses = []
        all_references = []
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]  ')
        for raw_src, src, raw_tgt, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = create_masks(
                src, 
                tgt_input, 
                self.src_vocab.pad_idx, 
                self.tgt_vocab.pad_idx, 
                self.device
            )
            
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = self.criterion(logits, tgt_output)
            total_loss += loss.item()

            decoded_batch = self.greedy_decode_batch(src, max_len=self.max_tgt_len)
            for pred_ids, ref_sentence in zip(decoded_batch, raw_tgt):
                hyp = self.tgt_vocab.decode(pred_ids, raw_src)
                all_hypotheses.append(hyp)
                all_references.append(ref_sentence)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        bleu = self.bleu_metric.corpus_score(
            all_hypotheses,
            [all_references]
        ).score
        return avg_loss, bleu
    
    def train(self, num_epochs, warmup_scheduler=None, plateau_scheduler=None, patience=5):
        print("Start training")
        total, trainable = count_parameters(self.model)
        print("Model parameters - Total:", total, "Trainable:", trainable)
        
        epochs_no_improve = 0
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch, warmup_scheduler)
            val_loss, val_bleu = self.validate(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_bleu'].append(val_bleu)
            self.history['learning_rates'].append(current_lr)
            
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"BLEU: {val_bleu:.2f} | "
                f"LR: {current_lr:.2e}"
            )
            if val_bleu > self.best_val_bleu:
                self.best_val_bleu = val_bleu
                epochs_no_improve = 0

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.checkpoint_path
                )
                print(f"New best model saved (BLEU = {val_bleu:.2f})")
            else:
                epochs_no_improve += 1
                print(f"No BLEU improvement for {epochs_no_improve} epoch(s)")

            if plateau_scheduler is not None:
                plateau_scheduler.step(val_loss)
            

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

        self.save_history()
        print("Training completed. Best BLEU:", self.best_val_bleu)
    
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
