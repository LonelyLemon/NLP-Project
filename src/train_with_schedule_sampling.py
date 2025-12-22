from src.train import Trainer
from tqdm import tqdm
from src.utils import create_masks, create_padding_mask, create_causal_mask
import torch

class ScheduledSamplingTrainer(Trainer):
    def __init__(
        self,
        *args,
        tf_start=1.0,
        tf_end=0.2,
        tf_decay_steps=50000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tf_start = tf_start
        self.tf_end = tf_end
        self.tf_decay_steps = tf_decay_steps
        self.global_step = 0
    
    def teacher_forcing_ratio(self):
        if self.global_step >= self.tf_decay_steps:
            return self.tf_end
        return self.tf_start - (self.tf_start - self.tf_end) * (self.global_step / self.tf_decay_steps)

    def train_epoch(self, epoch, warmup_scheduler=None):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for raw_src, src, raw_tgt, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            B, TL = tgt.size()
            
            src_mask = create_padding_mask(src, self.src_pad_idx).to(self.device)
            enc_output = self.model.encode(src, src_mask)

            input_step = tgt[:, 0].unsqueeze(1)  # [B,1]
            logits_seq = []

            for t in range(1, TL):
                tf_ratio = self.teacher_forcing_ratio()
                tgt_mask = create_causal_mask(input_step.size(1), self.device)
                dec_output = self.model.decode(input_step, enc_output, src_mask, tgt_mask) # [B, T, D] 
                logits_step = self.model.output_proj(dec_output[:, -1, :]) # [B, V]
                logits_seq.append(logits_step.unsqueeze(1)) # [B, 1, V]

                use_teacher = torch.rand(B, device=self.device) < tf_ratio
                next_input = torch.where(
                    use_teacher.unsqueeze(1),
                    tgt[:, t].unsqueeze(1),
                    logits_step.argmax(dim=-1, keepdim=True)
                )
                input_step = torch.cat([input_step, next_input], dim=1)
            logits = torch.cat(logits_seq, dim=1)

            tgt_output = tgt[:, 1:].reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

            loss = self.criterion(logits, tgt_output)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN/Inf loss detected! Skipping batch.")
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.global_step += 1
            total_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}', 'tf_ratio': f'{tf_ratio:.2f}'})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0

        for raw_src, src, raw_tgt, tgt in val_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            B, TL = tgt.size()
            
            src_mask = create_padding_mask(src, self.src_pad_idx).to(self.device)
            enc_output = self.model.encode(src, src_mask)

            input_step = tgt[:, 0].unsqueeze(1)
            logits_seq = []

            for t in range(1, TL):
                tgt_mask = create_causal_mask(input_step.size(1), self.device)
                dec_output = self.model.decode(input_step, enc_output, src_mask, tgt_mask)
                logits_step = self.model.output_proj(dec_output[:, -1, :])
                logits_seq.append(logits_step.unsqueeze(1))

                next_input = logits_step.argmax(dim=-1, keepdim=True)
                input_step = torch.cat([input_step, next_input], dim=1)

            logits = torch.cat(logits_seq, dim=1)
            tgt_output = tgt[:, 1:].reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))
            loss = self.criterion(logits, tgt_output)
            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, num_epochs, plateau_scheduler=None, patience=5):
        return super().train(
            num_epochs=num_epochs, 
            warmup_scheduler=None, 
            plateau_scheduler=plateau_scheduler, 
            patience=patience
        )