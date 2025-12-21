import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import numpy as np


class Evaluator:
    def __init__(self, model, test_loader, src_vocab, tgt_vocab, device, use_subword: bool = False):
        self.model = model
        self.test_loader = test_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.use_subword = use_subword
        
        self.bleu_metric = BLEU(tokenize='intl', lowercase=True, smooth_method='exp')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    
    def evaluate_with_decoder(self, decoder, desc="Evaluation"):
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
            for batch_idx, (raw_src_batch, src_batch, raw_tgt_batch, tgt_batch) in enumerate(tqdm(self.test_loader, desc=desc)):
                batch_size = src_batch.size(0)
                
                for i in range(batch_size):
                    raw_src = raw_src_batch[i]
                    src = src_batch[i:i+1]  # [1, S]
                    raw_tgt = raw_tgt_batch[i]
                    tgt = tgt_batch[i]  # [T]
                    
                    # Decode
                    pred_indices = decoder.decode(src, self.src_vocab, self.tgt_vocab, self.device)
                    
                    # Convert to sentences
                    pred_sentence = self.tgt_vocab.decode(pred_indices, raw_src)
                    ref_sentence = raw_tgt
                    src_sentence = raw_src
                    
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
