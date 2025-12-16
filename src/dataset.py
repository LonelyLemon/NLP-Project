import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re
import sentencepiece as spm

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>' # Start of Sentence
EOS_TOKEN = '<eos>' # End of Sentence

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.freq_threshold = freq_threshold

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
            self.stoi[token] if token in self.stoi else self.stoi[UNK_TOKEN]
            for token in tokenized_text
        ]

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


class BilingualDataset(Dataset):
    def __init__(self, dataset, src_vocab, trg_vocab, src_lang='en', trg_lang='vi'):
        self.dataset = dataset
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_lang = src_lang
        self.trg_lang = trg_lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_text = self.dataset[index][self.src_lang]
        trg_text = self.dataset[index][self.trg_lang]

        src_numericalized = [self.src_vocab.stoi[SOS_TOKEN]]
        src_numericalized += self.src_vocab.numericalize(src_text)
        src_numericalized.append(self.src_vocab.stoi[EOS_TOKEN])

        trg_numericalized = [self.trg_vocab.stoi[SOS_TOKEN]]
        trg_numericalized += self.trg_vocab.numericalize(trg_text)
        trg_numericalized.append(self.trg_vocab.stoi[EOS_TOKEN])

        return torch.tensor(src_numericalized), torch.tensor(trg_numericalized)

class SpmBilingualDataset(Dataset):
    def __init__(self, dataset, src_vocab, trg_vocab, src_lang='en', trg_lang='vi'):
        self.dataset = dataset
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_lang = src_lang
        self.trg_lang = trg_lang

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        src_ids = (
            [self.src_vocab.sos_idx]
            + self.src_vocab.encode(self.dataset[idx][self.src_lang])
            + [self.src_vocab.eos_idx]
        )

        trg_ids = (
            [self.trg_vocab.sos_idx]
            + self.trg_vocab.encode(self.dataset[idx][self.trg_lang])
            + [self.trg_vocab.eos_idx]
        )

        return torch.tensor(src_ids), torch.tensor(trg_ids)

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src = [item[0] for item in batch]
        trg = [item[1] for item in batch]

        src = pad_sequence(src, batch_first=True, padding_value=self.pad_idx)
        trg = pad_sequence(trg, batch_first=True, padding_value=self.pad_idx)

        return src, trg