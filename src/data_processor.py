import html
import re
import unicodedata
from typing import Literal
from datasets import Dataset

def preprocess_text(text: str) -> str:
    text = html.unescape(text)
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(dataset: Dataset, ignore: Literal['vi', 'en', None] = None):
    def _run(example):
        example['raw_en'] = example['en']
        example['raw_vi'] = example['vi']
        if ignore != 'vi':
            example['vi'] = preprocess_text(example['vi'])
        if ignore != 'en':
            example['en'] = preprocess_text(example['en'])
        return example
    
    return dataset.map(_run)

def invert_html_sign(text: str) -> str:
    html_map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&apos;',
    }
    for sign, value in html_map.items():
        text = text.replace(sign, value)
    return text

def postprocess_text(raw_input: str, pred: str) -> str:
    pred = pred.strip()
    pred = invert_html_sign(pred)
    entities = re.findall(r"&[a-z]+;", pred)
    for i, ent in enumerate(entities):
        pred = pred.replace(ent, f"@@{i}@@")
    pred = re.sub(r"([,.\?!'\":;])", r' \1 ', pred)
    for i, ent in enumerate(entities):
        pred = pred.replace(f"@@{i}@@", ent)
    
    raw_input_words = raw_input.split()
    start_id = 0
    while start_id < len(raw_input_words):
        if not re.fullmatch(r"[A-Za-zÀ-ỹ0-9]+", raw_input_words[start_id]):
            start_id += 1
        else:
            break
    upper_words = []
    for i, word in enumerate(raw_input_words):
        if i <= start_id:
            continue
        if word[0].isupper() and raw_input_words[i - 1][0] not in ['.', '?', '!']:
            upper_words.append(word)
    
    for word in upper_words:
        pred_words = pred.split()
        for i in range(len(pred_words)):
            if pred_words[i] == word.lower():
                pred_words[i] = word
        pred = ' '.join(pred_words)

    pred_words = pred.split()
    all_words = []
    start_id = 0
    while start_id < len(pred_words):
        if not re.fullmatch(r"[A-Za-zÀ-ỹ0-9]+", pred_words[start_id]):
            start_id += 1
        else:
            break
    for i, word in enumerate(pred_words):
        if i < start_id:
            all_words.append(word)
        elif i == start_id or pred_words[i - 1] in ['.', '?', '!']:
            all_words.append(word[0].upper() + word[1:])
        else:
            all_words.append(word)
    return ' '.join(all_words)

if __name__ == '__main__':
    raw_text = 'And he said , &quot; Well , recently I pitched a sustainability project to a client , and turned and he said to me , &apos; I know it &apos;s going to cost less , I know it &apos;s going to sell more , but we &apos;re not pioneers , because pioneers have arrows in their backs . &apos; &quot; I think we &apos;ve got a roomful of pioneers , and I hope there are far more pioneers out there , because we need to solve these problems .'

    convert = preprocess_text(raw_text)

    inconvert = postprocess_text(raw_text, convert)

    print(raw_text)
    print('-' * 50)
    print(convert)
    print('-' * 50)
    print(inconvert)