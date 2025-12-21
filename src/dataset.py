import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re
import sentencepiece as spm
from src.data_processor import postprocess_text


class Vocabulary:
    def __init__(self, freq_threshold=2):
        UNK_TOKEN = '<unk>'
        PAD_TOKEN = '<pad>'
        SOS_TOKEN = '<sos>'
        EOS_TOKEN = '<eos>'
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.freq_threshold = freq_threshold

        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [tok.lower() for tok in re.findall(r"\w+|[^\w\s]", text, re.UNICODE)]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4 

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi[token] if token in self.stoi else self.unk_idx
            for token in tokenized_text
        ]
    
    def decode(self, indices, raw_src):
        tokens = []
        for idx in indices:
            if idx in [self.pad_idx, self.sos_idx, self.eos_idx]:
                continue
            if idx == self.unk_idx:
                tokens.append("<unk>")
            else:
                tokens.append(self.itos.get(idx, "<unk>"))

        seq = " ".join(tokens)
        return postprocess_text(raw_input=raw_src, pred=seq)
        

class SubwordVocabulary:
    def __init__(self, spm_model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)

        self.pad_idx = self.sp.pad_id()
        self.sos_idx = self.sp.bos_id()
        self.eos_idx = self.sp.eos_id()
        self.unk_idx = self.sp.unk_id()
    
    def __len__(self):
        return self.sp.get_piece_size()

    def numericalize(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, indices, raw_src):
        seq = self.sp.decode_ids(indices)
        return postprocess_text(raw_input=raw_src, pred=seq)


class BilingualDataset(Dataset):
    def __init__(self, dataset, src_vocab, trg_vocab, max_src_len, max_trg_len, src_lang='en', trg_lang='vi'):
        self.dataset = dataset
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw_src_text = self.dataset[index][f'raw_{self.src_lang}']
        src_text = self.dataset[index][self.src_lang]
        raw_trg_text = self.dataset[index][f'raw_{self.trg_lang}']
        trg_text = self.dataset[index][self.trg_lang]

        src_numericalized = [self.src_vocab.sos_idx]
        src_numericalized += self.src_vocab.numericalize(src_text)[:self.max_src_len - 2]
        src_numericalized.append(self.src_vocab.eos_idx)

        trg_numericalized = [self.trg_vocab.sos_idx]
        trg_numericalized += self.trg_vocab.numericalize(trg_text)[:self.max_trg_len - 2]
        trg_numericalized.append(self.trg_vocab.eos_idx)

        return raw_src_text, torch.tensor(src_numericalized), raw_trg_text, torch.tensor(trg_numericalized)

class SpmBilingualDataset(Dataset):
    def __init__(self, dataset, src_vocab, trg_vocab, max_src_len, max_trg_len, src_lang='en', trg_lang='vi'):
        self.dataset = dataset
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        raw_src_text = self.dataset[idx][f'raw_{self.src_lang}']
        src_ids = (
            [self.src_vocab.sos_idx]
            + self.src_vocab.numericalize(self.dataset[idx][self.src_lang])[:self.max_src_len - 2]
            + [self.src_vocab.eos_idx]
        )

        raw_trg_text = self.dataset[idx][f'raw_{self.trg_lang}']
        trg_ids = (
            [self.trg_vocab.sos_idx]
            + self.trg_vocab.numericalize(self.dataset[idx][self.trg_lang])[:self.max_trg_len - 2]
            + [self.trg_vocab.eos_idx]
        )

        return raw_src_text, torch.tensor(src_ids), raw_trg_text, torch.tensor(trg_ids)

class Collate:
    def __init__(self, src_pad_idx, trg_pad_idx, max_src_len=None, max_trg_len=None):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __call__(self, batch):
        raw_src = [item[0] for item in batch]
        src = [item[1] for item in batch]
        raw_trg = [item[2] for item in batch]
        trg = [item[3] for item in batch]
        if self.max_src_len is not None:
            src = [s[:self.max_src_len] for s in src]
        if self.max_trg_len is not None:
            trg = [t[:self.max_trg_len] for t in trg]

        src = pad_sequence(src, batch_first=True, padding_value=self.src_pad_idx)
        trg = pad_sequence(trg, batch_first=True, padding_value=self.trg_pad_idx)

        return raw_src, src, raw_trg, trg