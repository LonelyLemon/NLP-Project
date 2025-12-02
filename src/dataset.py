import torch
import re

from torch.utils.data import Dataset, DataLoader
from collections import Counter


# Token Define
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
        # Tokenizer
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
        src_text = self.dataset[index]['translation'][self.src_lang]
        trg_text = self.dataset[index]['translation'][self.trg_lang]

        # Text -> List of indices
        src_numericalized = [self.src_vocab.stoi[SOS_TOKEN]]
        src_numericalized += self.src_vocab.numericalize(src_text)
        src_numericalized.append(self.src_vocab.stoi[EOS_TOKEN])

        trg_numericalized = [self.trg_vocab.stoi[SOS_TOKEN]]
        trg_numericalized += self.trg_vocab.numericalize(trg_text)
        trg_numericalized.append(self.trg_vocab.stoi[EOS_TOKEN])

        return torch.tensor(src_numericalized), torch.tensor(trg_numericalized)