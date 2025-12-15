import os
import gc
from datasets import load_dataset
import sentencepiece as spm

DATASET = "thainq107/iwslt2015-en-vi"
SPLIT = "train"

VOCAB_SIZE = 16000
MODEL_TYPE = "unigram"

ds = load_dataset(DATASET, split=SPLIT)
with open("train.en", "w", encoding="utf-8") as f_en, \
     open("train.vi", "w", encoding="utf-8") as f_vi:
    for x in ds:
        f_en.write(x["en"].strip() + "\n")
        f_vi.write(x["vi"].strip() + "\n")

print("Dump done")

del ds
gc.collect()

print("Training SentencePiece...")

spm.SentencePieceTrainer.train(
    input="train.en",
    model_prefix="spm_en",
    vocab_size=VOCAB_SIZE,
    model_type=MODEL_TYPE,
    character_coverage=1.0,
    pad_id=0, bos_id=1, eos_id=2, unk_id=3
)

spm.SentencePieceTrainer.train(
    input="train.vi",
    model_prefix="spm_vi",
    vocab_size=VOCAB_SIZE,
    model_type=MODEL_TYPE,
    character_coverage=0.9995,
    pad_id=0, bos_id=1, eos_id=2, unk_id=3
)

print("SentencePiece trained")

os.remove("train.en")
os.remove("train.vi")