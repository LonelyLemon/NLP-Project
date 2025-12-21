import torch
import torch.nn.functional as F
from src.utils import create_padding_mask, create_causal_mask
from src.data_processor import preprocess_text


class GreedySearchDecoder:    
    def __init__(self, model, max_len=100, use_subword=False):
        self.model = model
        self.max_len = max_len
        self.use_subword = use_subword
    
    @torch.no_grad()
    def decode(self, src, src_vocab, tgt_vocab, device):
        self.model.eval()
        
        src = src.to(device)
        pad_idx = src_vocab.pad_idx
        sos_idx = tgt_vocab.sos_idx
        eos_idx = tgt_vocab.eos_idx
        
        src_mask = create_padding_mask(src, pad_idx).to(device)
        enc_output = self.model.encode(src, src_mask)  # [1, S, D]

        decoded_tokens = [sos_idx]
        
        for _ in range(self.max_len):
            tgt = torch.LongTensor([decoded_tokens]).to(device)  # [1, T]
            tgt_mask = create_causal_mask(len(decoded_tokens), device)
            dec_output = self.model.decode(tgt, enc_output, src_mask, tgt_mask)  # [1, T, D]
            logits = self.model.output_proj(dec_output[:, -1, :])  # [1, V]
            next_token = logits.argmax(dim=-1).item()
            decoded_tokens.append(next_token)
            if next_token == eos_idx:
                break
        
        return decoded_tokens
    
    def translate(self, src_sentence, src_vocab, tgt_vocab, device):
        sent = preprocess_text(src_sentence)
        src_ids = (
            [src_vocab.sos_idx]
            + src_vocab.numericalize(sent)
            + [src_vocab.eos_idx]
        )
        src_tensor = torch.LongTensor([src_ids]).to(device)  # [1, S]

        decoded_ids = self.decode(src_tensor, src_vocab, tgt_vocab, device)
        
        return tgt_vocab.decode(decoded_ids, src_sentence)


class BeamSearchDecoder:
    def __init__(self, model, beam_size=5, max_len=100, length_penalty=0.6, use_subword=False):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.use_subword = use_subword
    
    @torch.no_grad()
    def decode(self, src, src_vocab, tgt_vocab, device):
        self.model.eval()
        
        src = src.to(device)
        pad_idx = src_vocab.pad_idx
        sos_idx = tgt_vocab.sos_idx
        eos_idx = tgt_vocab.eos_idx

        src_mask = create_padding_mask(src, pad_idx).to(device)
        enc_output = self.model.encode(src, src_mask)  # [1, S, D]

        beams = [(0.0, [sos_idx])]
        completed_beams = []
        
        for step in range(self.max_len):
            candidates = []
            
            for score, tokens in beams:
                if tokens[-1] == eos_idx:
                    completed_beams.append((score, tokens))
                    continue
                
                # Tạo target tensor
                tgt = torch.LongTensor([tokens]).to(device)  # [1, T]
                tgt_mask = create_causal_mask(len(tokens), device)
                
                # Decode
                dec_output = self.model.decode(tgt, enc_output, src_mask, tgt_mask)
                logits = self.model.output_proj(dec_output[:, -1, :])  # [1, V]
                
                # Get log probabilities
                log_probs = F.log_softmax(logits, dim=-1)  # [1, V]
                
                # Get top-k tokens
                topk_log_probs, topk_indices = log_probs.topk(self.beam_size, dim=-1)
                
                # Create new candidates
                for i in range(self.beam_size):
                    token_id = topk_indices[0, i].item()
                    token_score = topk_log_probs[0, i].item()
                    
                    new_score = score + token_score
                    new_tokens = tokens + [token_id]
                    
                    candidates.append((new_score, new_tokens))
            
            # Không còn candidates nào
            if not candidates:
                break

            candidates.sort(reverse=True, key=lambda x: x[0])
            beams = candidates[:self.beam_size]
            
            # Chọn top-k beams theo score
            # Apply length normalization: score / (len ** alpha)
            # candidates_normalized = [
            #     (score / (len(tokens) ** self.length_penalty), score, tokens)
            #     for score, tokens in candidates
            # ]
            # candidates_normalized.sort(reverse=True, key=lambda x: x[0])
            
            # # Keep top beam_size
            # beams = [(score, tokens) for _, score, tokens in candidates_normalized[:self.beam_size]]
            
            # Early stopping: nếu đã có đủ completed beams
            if len(completed_beams) >= self.beam_size:
                break
        
        # Add remaining beams to completed
        completed_beams.extend(beams)
        
        # Chọn best beam (normalize by length)
        # if completed_beams:
        #     best = max(completed_beams, key=lambda x: x[0] / (len(x[1]) ** self.length_penalty))
        #     return best[1]
        # else:
        #     # Fallback: return beam đầu tiên
        #     return beams[0][1] if beams else [sos_idx, eos_idx]

        if completed_beams:
            best = max(
                completed_beams,
                key=lambda x: x[0] / (len(x[1]) ** self.length_penalty)
            )
            return best[1]
        else:
            return beams[0][1] if beams else [sos_idx, eos_idx]
    
    def translate(self, src_sentence, src_vocab, tgt_vocab, device):
        sent = preprocess_text(src_sentence)
        src_ids = (
            [src_vocab.sos_idx]
            + src_vocab.numericalize(sent)
            + [src_vocab.eos_idx]
        )
        src_tensor = torch.LongTensor([src_ids]).to(device)

        decoded_ids = self.decode(src_tensor, src_vocab, tgt_vocab, device)

        return tgt_vocab.decode(decoded_ids, src_sentence)
