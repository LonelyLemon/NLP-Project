import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import numpy as np


class Evaluator:
    """
    Evaluator để đánh giá model translation quality với BLEU và ROUGE-L scores.
    """
    
    def __init__(self, model, test_loader, src_vocab, tgt_vocab, device, use_subword: bool = False):
        """
        Args:
            model: Trained Transformer model
            test_loader: DataLoader cho test set
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            device: torch.device
        """
        self.model = model
        self.test_loader = test_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.use_subword = use_subword
        
        self.bleu_metric = BLEU(tokenize='none' if use_subword else '13a')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    def indices_to_sentence(self, indices, vocab, remove_special=True):
        """
        Convert token indices sang sentence string.
        
        Args:
            indices: list or tensor - Token indices
            vocab: Vocabulary object
            remove_special: bool - Remove special tokens (<sos>, <eos>, <pad>)
            
        Returns:
            sentence: str
        """
        if torch.is_tensor(indices):
            indices = indices.tolist()
        if self.use_subword:
            return vocab.sp.decode_ids(indices)
        
        words = [vocab.itos[idx] for idx in indices]
        words = [w for w in words if w not in ['<sos>', '<eos>', '<pad>']]
        return ' '.join(words)
    
    def evaluate_with_decoder(self, decoder, desc="Evaluation"):
        """
        Evaluate model sử dụng decoder cụ thể (Greedy hoặc Beam Search).
        
        Args:
            decoder: GreedySearchDecoder hoặc BeamSearchDecoder instance
            desc: str - Description cho progress bar
            
        Returns:
            results: dict - Contains BLEU, ROUGE-L scores và examples
        """
        self.model.eval()
        
        references = []  # Ground truth translations
        hypotheses = []  # Model predictions
        rouge_scores = []
        
        # Sample translations để show
        examples = []
        num_examples = 5
        
        print(f"\n{'='*60}")
        print(f"📊 {desc}")
        print(f"{'='*60}\n")
        
        with torch.no_grad():
            for batch_idx, (src_batch, tgt_batch) in enumerate(tqdm(self.test_loader, desc=desc)):
                batch_size = src_batch.size(0)
                
                for i in range(batch_size):
                    src = src_batch[i:i+1]  # [1, S]
                    tgt = tgt_batch[i]  # [T]
                    
                    # Decode
                    pred_indices = decoder.decode(src, self.src_vocab, self.tgt_vocab, self.device)
                    
                    # Convert to sentences
                    pred_sentence = self.indices_to_sentence(pred_indices, self.tgt_vocab)
                    ref_sentence = self.indices_to_sentence(tgt, self.tgt_vocab)
                    src_sentence = self.indices_to_sentence(src[0], self.src_vocab)
                    
                    # Collect for metrics
                    hypotheses.append(pred_sentence)
                    references.append(ref_sentence)
                    
                    # Calculate ROUGE-L for this pair
                    rouge_result = self.rouge_scorer.score(ref_sentence, pred_sentence)
                    rouge_scores.append(rouge_result['rougeL'].fmeasure)
                    
                    # Save examples
                    if len(examples) < num_examples:
                        examples.append({
                            'source': src_sentence,
                            'reference': ref_sentence,
                            'prediction': pred_sentence
                        })
        
        # Calculate BLEU
        # sacrebleu expects list of references for each hypothesis
        bleu_score = self.bleu_metric.corpus_score(hypotheses, [references])
        
        # Average ROUGE-L
        avg_rouge = np.mean(rouge_scores)
        
        results = {
            'bleu': bleu_score.score,
            'rouge_l': avg_rouge,
            'num_samples': len(hypotheses),
            'examples': examples
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"📈 Results:")
        print(f"   BLEU Score:   {bleu_score.score:.2f}")
        print(f"   ROUGE-L F1:   {avg_rouge:.4f}")
        print(f"   Samples:      {len(hypotheses)}")
        print(f"{'='*60}\n")
        
        # Print examples
        print(f"{'='*60}")
        print(f"📝 Translation Examples:")
        print(f"{'='*60}")
        for idx, ex in enumerate(examples, 1):
            print(f"\nExample {idx}:")
            print(f"  Source:     {ex['source']}")
            print(f"  Reference:  {ex['reference']}")
            print(f"  Prediction: {ex['prediction']}")
        print(f"\n{'='*60}\n")
        
        return results
    
    def compare_decoders(self, greedy_decoder, beam_decoder):
        """
        So sánh Greedy Search vs Beam Search.
        
        Args:
            greedy_decoder: GreedySearchDecoder instance
            beam_decoder: BeamSearchDecoder instance
            
        Returns:
            comparison: dict - Results từ cả 2 decoders
        """
        print("\n" + "="*60)
        print("🔍 COMPARING DECODING STRATEGIES")
        print("="*60)
        
        greedy_results = self.evaluate_with_decoder(greedy_decoder, "Greedy Search")
        beam_results = self.evaluate_with_decoder(beam_decoder, f"Beam Search (k={beam_decoder.beam_size})")
        
        # Summary comparison
        print("\n" + "="*60)
        print("📊 COMPARISON SUMMARY")
        print("="*60)
        print(f"\n{'Method':<20} {'BLEU':<10} {'ROUGE-L':<10}")
        print("-" * 40)
        print(f"{'Greedy Search':<20} {greedy_results['bleu']:<10.2f} {greedy_results['rouge_l']:<10.4f}")
        print(f"{'Beam Search':<20} {beam_results['bleu']:<10.2f} {beam_results['rouge_l']:<10.4f}")
        print("-" * 40)
        
        improvement_bleu = beam_results['bleu'] - greedy_results['bleu']
        improvement_rouge = beam_results['rouge_l'] - greedy_results['rouge_l']
        
        print(f"{'Improvement':<20} {improvement_bleu:<10.2f} {improvement_rouge:<10.4f}")
        print("="*60 + "\n")
        
        return {
            'greedy': greedy_results,
            'beam': beam_results,
            'improvement': {
                'bleu': improvement_bleu,
                'rouge_l': improvement_rouge
            }
        }


def calculate_bleu_score(references, hypotheses):
    """
    Helper function để tính BLEU score.
    
    Args:
        references: list of str - Ground truth translations
        hypotheses: list of str - Model predictions
        
    Returns:
        bleu_score: float
    """
    bleu = BLEU()
    score = bleu.corpus_score(hypotheses, [references])
    return score.score


def calculate_rouge_score(references, hypotheses):
    """
    Helper function để tính ROUGE-L score.
    
    Args:
        references: list of str - Ground truth translations
        hypotheses: list of str - Model predictions
        
    Returns:
        avg_rouge_l: float - Average ROUGE-L F1 score
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = []
    
    for ref, hyp in zip(references, hypotheses):
        result = scorer.score(ref, hyp)
        scores.append(result['rougeL'].fmeasure)
    
    return np.mean(scores)
