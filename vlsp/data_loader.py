from datasets import Dataset
import os

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def create_dataset(en_path, vi_path):
    en_lines = read_text_file(en_path)
    vi_lines = read_text_file(vi_path)
    
    assert len(en_lines) == len(vi_lines), "Số dòng file En và Vi không khớp!"
    
    data = []
    
    for en, vi in zip(en_lines, vi_lines):
        if not en or not vi: 
            continue
        
        data.append({
            "messages": [
                {"role": "system", "content": "Bạn là một trợ lý dịch thuật y tế chuyên nghiệp."},
                {"role": "user", "content": f"Dịch câu sau sang tiếng Việt: {en}"},
                {"role": "assistant", "content": vi}
            ]
        })
        
        data.append({
            "messages": [
                {"role": "system", "content": "You are a professional medical translation assistant."},
                {"role": "user", "content": f"Translate the following sentence to English: {vi}"},
                {"role": "assistant", "content": en}
            ]
        })
        
    return Dataset.from_list(data)

def get_formatted_dataset(config):
    print(f"Loading data from {config.TRAIN_EN} and {config.TRAIN_VI}...")
    dataset = create_dataset(config.TRAIN_EN, config.TRAIN_VI)
    
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    return dataset