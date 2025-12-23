# NLP Project: Neural Machine Translation (2025)

Repository này chứa mã nguồn cho **Bài tập lớn cuối kỳ môn Xử lý Ngôn ngữ Tự nhiên (NLP) năm 2025**. Dự án tập trung vào hai nhiệm vụ chính:

1.  **Task 1 (70%):** Xây dựng mô hình Transformer từ đầu (Implement from Scratch) cho bài toán dịch máy (Machine Translation).
2.  **Task 2 (30%):** Tham gia VLSP Shared Task - Dịch máy lĩnh vực Y tế (Medical Domain) sử dụng kỹ thuật Fine-tuning LLM (Qwen + LoRA).

---

## 📂 Cấu trúc Dự án

Dự án được tổ chức thành các thư mục chức năng riêng biệt để tách bạch giữa mã nguồn tự xây dựng (Task 1) và mã nguồn sử dụng thư viện có sẵn (Task 2).

```text
├── data/                       # Chứa dữ liệu huấn luyện và kiểm thử
│   ├── task1/                  # Dữ liệu cho Task 1
│   └── ...                     # Các dữ liệu sử dụng đến
│
├── src/                        # Mã nguồn xử lý cốt lõi
│   ├── model/                  
│   │   ├── transformer.py      # Class Transformer chính
│   │   ├── encoder.py          # Encoder Block & Layer
│   │   ├── decoder.py          # Decoder Block & Layer
│   │   ├── multihead_attention.py          # Multi-Head Attention Mechanism
│   │   ├── rope.py             # Rotary Positional Embeddings
│   │   └── swiglu.py           # SwiGLU Activation
│   ├── data_processor.py       # Xử lý dữ liệu, Tokenization (SentencePiece/BPE)
│   ├── dataset.py              # Custom Pytorch Dataset
│   ├── train.py                # Training Loop
│   ├── evaluate.py             # Tính toán BLEU Score
│   └── inference.py            # Beam Search & Greedy Decoding
│
├── vlsp/                       # Mã nguồn VLSP Finetuning
│   ├── train.py                # Script train Qwen với LoRA (QLoRA)
│   ├── data_loader.py          # Load dữ liệu VLSP
│   ├── config.py               # File cấu hình Hyperparameters
│   └── inference.py            # Script chạy dịch thử nghiệm
│
├── notebooks/                  # Chạy thực nghiệm và visualize kết quả
│   ├── task1_pos_ffn.ipynb     # [Task 1] Transformer cơ bản
│   ├── task1_rope_swiglu.ipynb # [Task 1] Transformer nâng cao (RoPE + SwiGLU)
│   ├── finetune_vlsp vi2en.ipynb # [Task 2 - Vie ---> En] Notebook train Task 2 (Việt -> Anh)
│   └── finetune_vlsp_en2vi.ipynb # [Task 2 - En ---> Vie]Notebook train Task 2 (Anh -> Việt)
│
├── ...   
├── main.py                     
└── pyproject.toml              # Quản lý thư viện (Dependencies)
