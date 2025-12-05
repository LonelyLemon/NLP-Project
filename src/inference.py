import torch
import torch.nn.functional as F
from .utils import create_padding_mask, create_causal_mask


class GreedySearchDecoder:
    """
    Greedy Search: Chọn token có xác suất cao nhất ở mỗi bước.
    Nhanh nhưng quality thấp hơn Beam Search.
    """
    
    def __init__(self, model, max_len=100):
        """
        Args:
            model: Trained Transformer model
            max_len: Maximum length của generated sequence
        """
        self.model = model
        self.max_len = max_len
    
    @torch.no_grad()
    def decode(self, src, src_vocab, tgt_vocab, device):
        """
        Greedy decode một câu source.
        
        Args:
            src: [1, S] - Source tensor (batch_size=1)
            src_vocab: Vocabulary object cho source
            tgt_vocab: Vocabulary object cho target
            device: torch.device
            
        Returns:
            decoded_tokens: list of int - Token IDs
        """
        self.model.eval()
        
        src = src.to(device)
        pad_idx = src_vocab.stoi['<pad>']
        sos_idx = tgt_vocab.stoi['<sos>']
        eos_idx = tgt_vocab.stoi['<eos>']
        
        # Encode source
        src_mask = create_padding_mask(src, pad_idx).to(device)
        enc_output = self.model.encode(src, src_mask)  # [1, S, D]
        
        # Start với <sos> token
        decoded_tokens = [sos_idx]
        
        for _ in range(self.max_len):
            # Tạo target tensor từ tokens đã decode
            tgt = torch.LongTensor([decoded_tokens]).to(device)  # [1, T]
            
            # Create target mask
            tgt_mask = create_causal_mask(len(decoded_tokens), device)
            
            # Decode
            dec_output = self.model.decode(tgt, enc_output, src_mask, tgt_mask)  # [1, T, D]
            
            # Get logits cho token cuối cùng
            logits = self.model.output_proj(dec_output[:, -1, :])  # [1, V]
            
            # Greedy: chọn token có prob cao nhất
            next_token = logits.argmax(dim=-1).item()
            
            decoded_tokens.append(next_token)
            
            # Stop nếu gặp <eos>
            if next_token == eos_idx:
                break
        
        return decoded_tokens
    
    def translate(self, src_sentence, src_vocab, tgt_vocab, device):
        """
        Translate một câu từ source sang target language.
        
        Args:
            src_sentence: str - Câu source (đã tokenized)
            src_vocab: Vocabulary cho source
            tgt_vocab: Vocabulary cho target
            device: torch.device
            
        Returns:
            translation: str - Câu đã dịch
        """
        # Tokenize và convert sang tensor
        src_tokens = src_sentence.split()
        src_indices = [src_vocab.stoi.get(token, src_vocab.stoi['<unk>']) for token in src_tokens]
        src_tensor = torch.LongTensor([src_indices]).to(device)  # [1, S]
        
        # Decode
        decoded_indices = self.decode(src_tensor, src_vocab, tgt_vocab, device)
        
        # Convert indices to words
        decoded_words = [tgt_vocab.itos[idx] for idx in decoded_indices]
        
        # Remove <sos>, <eos>, <pad>
        decoded_words = [w for w in decoded_words if w not in ['<sos>', '<eos>', '<pad>']]
        
        return ' '.join(decoded_words)


class BeamSearchDecoder:
    """
    Beam Search: Maintain top-k hypotheses để tìm translation tốt hơn.
    Chậm hơn nhưng quality cao hơn Greedy Search.
    """
    
    def __init__(self, model, beam_size=5, max_len=100, length_penalty=0.6):
        """
        Args:
            model: Trained Transformer model
            beam_size: Số lượng beams (hypotheses) để maintain
            max_len: Maximum length của generated sequence
            length_penalty: Alpha parameter cho length normalization
                           (0.0 = no penalty, 1.0 = full penalty)
        """
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
    
    @torch.no_grad()
    def decode(self, src, src_vocab, tgt_vocab, device):
        """
        Beam search decode.
        
        Args:
            src: [1, S] - Source tensor
            src_vocab: Vocabulary cho source
            tgt_vocab: Vocabulary cho target
            device: torch.device
            
        Returns:
            best_sequence: list of int - Best decoded token IDs
        """
        self.model.eval()
        
        src = src.to(device)
        pad_idx = src_vocab.stoi['<pad>']
        sos_idx = tgt_vocab.stoi['<sos>']
        eos_idx = tgt_vocab.stoi['<eos>']
        
        # Encode source
        src_mask = create_padding_mask(src, pad_idx).to(device)
        enc_output = self.model.encode(src, src_mask)  # [1, S, D]
        
        # Initialize beams
        # Each beam: (score, tokens)
        beams = [(0.0, [sos_idx])]
        completed_beams = []
        
        for step in range(self.max_len):
            candidates = []
            
            for score, tokens in beams:
                # Nếu beam đã kết thúc, add vào completed
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
            
            # Chọn top-k beams theo score
            # Apply length normalization: score / (len ** alpha)
            candidates_normalized = [
                (score / (len(tokens) ** self.length_penalty), score, tokens)
                for score, tokens in candidates
            ]
            candidates_normalized.sort(reverse=True, key=lambda x: x[0])
            
            # Keep top beam_size
            beams = [(score, tokens) for _, score, tokens in candidates_normalized[:self.beam_size]]
            
            # Early stopping: nếu đã có đủ completed beams
            if len(completed_beams) >= self.beam_size:
                break
        
        # Add remaining beams to completed
        completed_beams.extend(beams)
        
        # Chọn best beam (normalize by length)
        if completed_beams:
            best = max(completed_beams, key=lambda x: x[0] / (len(x[1]) ** self.length_penalty))
            return best[1]
        else:
            # Fallback: return beam đầu tiên
            return beams[0][1] if beams else [sos_idx, eos_idx]
    
    def translate(self, src_sentence, src_vocab, tgt_vocab, device):
        """
        Translate một câu sử dụng beam search.
        
        Args:
            src_sentence: str - Câu source
            src_vocab: Vocabulary cho source
            tgt_vocab: Vocabulary cho target
            device: torch.device
            
        Returns:
            translation: str - Câu đã dịch
        """
        # Tokenize
        src_tokens = src_sentence.split()
        src_indices = [src_vocab.stoi.get(token, src_vocab.stoi['<unk>']) for token in src_tokens]
        src_tensor = torch.LongTensor([src_indices]).to(device)
        
        # Decode
        decoded_indices = self.decode(src_tensor, src_vocab, tgt_vocab, device)
        
        # Convert to words
        decoded_words = [tgt_vocab.itos[idx] for idx in decoded_indices]
        decoded_words = [w for w in decoded_words if w not in ['<sos>', '<eos>', '<pad>']]
        
        return ' '.join(decoded_words)
