import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import numpy as np
from src.inference import GreedySearchDecoder


class Evaluator:
    def __init__(self, model, test_loader, src_vocab, tgt_vocab, device):
        self.model = model
        self.test_loader = test_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.bleu_metric = BLEU(tokenize='13a', lowercase=True, smooth_method='exp')

    
    def evaluate(self, decoder):
        self.model.eval()
        references, hypotheses = [], []
        
        with torch.no_grad():
            if isinstance(decoder, GreedySearchDecoder):
                log_rel_path = 'log_result_greedy.txt'
            else:
                log_rel_path = 'log_result_beam.txt'
            with open(log_rel_path, 'w', encoding='utf-8') as f:
                for raw_src_batch, src_batch, raw_tgt_batch, tgt_batch in tqdm(self.test_loader):
                    batch_size = src_batch.size(0)
                    for i in range(batch_size):
                        raw_src = raw_src_batch[i]
                        src = src_batch[i:i+1].to(self.device)
                        ref_sentence = raw_tgt_batch[i]
                        pred_indices = decoder.decode(src, self.src_vocab, self.tgt_vocab, self.device)
                        pred_sentence = self.tgt_vocab.decode(pred_indices, raw_src)

                        hypotheses.append(pred_sentence)
                        references.append(ref_sentence)
                        f.write(f'pred: {pred_sentence}\n')
                        f.write(f'ref: {ref_sentence}\n')
                        f.write('-' * 50)
                        f.write('\n')
                        
        bleu_score = self.bleu_metric.corpus_score(hypotheses, [references])
        print(f"BLEU Score: {bleu_score.score:.2f}")
        return bleu_score.score
    
    def compare_decoders(self, greedy_decoder, beam_decoder):
        greedy_bleu = self.evaluate(greedy_decoder)
        beam_bleu = self.evaluate(beam_decoder)

        print("Comparison of decoding strategies:")
        print(f"Greedy BLEU: {greedy_bleu:.2f}")
        print(f"Beam BLEU:   {beam_bleu:.2f}")
        print(f"Improvement: {beam_bleu - greedy_bleu:.2f}")

        return {
            'greedy_bleu': greedy_bleu,
            'beam_bleu': beam_bleu,
            'improvement': beam_bleu - greedy_bleu
        }
