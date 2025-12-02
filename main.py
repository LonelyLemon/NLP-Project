# main.py
import torch
from datasets import load_dataset
from src.dataset import Vocabulary, BilingualDataset, Collate, PAD_TOKEN

def main():
    print(">>> 1. Đang tải dữ liệu IWSLT2015 (En-Vi)...")
    dataset = load_dataset("mt_eng_vietnamese", "iwslt2015-en-vi", split='train[:1%]')
    print(f"Số lượng mẫu: {len(dataset)}")

    print("\n>>> 2. Đang xây dựng Vocabulary...")
    src_sentences = [x['translation']['en'] for x in dataset]
    trg_sentences = [x['translation']['vi'] for x in dataset]

    src_vocab = Vocabulary(freq_threshold=1)
    src_vocab.build_vocabulary(src_sentences)
    
    trg_vocab = Vocabulary(freq_threshold=1)
    trg_vocab.build_vocabulary(trg_sentences)

    print(f"Kích thước Vocab Tiếng Anh: {len(src_vocab)}")
    print(f"Kích thước Vocab Tiếng Việt: {len(trg_vocab)}")

    print("\n>>> 3. Tạo DataLoader...")
    train_dataset = BilingualDataset(dataset, src_vocab, trg_vocab)
    
    pad_idx = src_vocab.stoi[PAD_TOKEN]
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=Collate(pad_idx=pad_idx)
    )

    # Kiểm tra 1 batch
    src_batch, trg_batch = next(iter(train_loader))
    print("\n>>> 4. Kiểm tra Batch Output:")
    print(f"Source Batch Shape: {src_batch.shape}")
    print(f"Target Batch Shape: {trg_batch.shape}")
    print("\nVí dụ Tensor câu đầu tiên (Source):")
    print(src_batch[0])

if __name__ == "__main__":
    main()