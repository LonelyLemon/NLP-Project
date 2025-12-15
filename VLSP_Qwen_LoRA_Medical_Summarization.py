import torch
import sacrebleu
import google.generativeai as genai
import json
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# -------------------------------
# CẤU HÌNH API KEY GEMINI
# -------------------------------
GEMINI_API_KEY = "DÁN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"  # <-- Thay bằng API key thật của bạn

if GEMINI_API_KEY == "DÁN_API_KEY_CỦA_BẠN_VÀO_ĐÂY":
    print("⚠️ CẢNH BÁO: Bạn chưa điền API Key! Phần đánh giá bằng Gemini sẽ bị lỗi.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Giả sử các biến sau đã được định nghĩa trước đó trong notebook/script chính:
# - trainer
# - tokenizer
# - eval_ds
# - OUTPUT_DIR (kiểu pathlib.Path)

# --- 1. HUẤN LUYỆN (TRAINING) ---
print("🚀 Đang bắt đầu huấn luyện model... (Vui lòng đợi)")
trainer.train()

# Lưu model
final_path = OUTPUT_DIR / "final_model"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"✅ Đã train xong! Model lưu tại: {final_path}")

# --- 2. CHUẨN BỊ ĐÁNH GIÁ (INFERENCE) ---
model_to_eval = trainer.model
model_to_eval.eval()

def generate_summary(text, model, tokenizer):
    prompt = f"<|im_start|>system\nBạn là một trợ lý y khoa tiếng Việt.<|im_end|>\n<|im_start|>user\nTóm tắt đoạn văn y khoa sau bằng tiếng Việt:\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resp.split("assistant\n")[-1].strip() if "assistant\n" in resp else resp

def evaluate_with_gemini(source, reference, prediction):
    if not GEMINI_API_KEY or "DÁN_API" in GEMINI_API_KEY:
        return {"score": 0, "reason": "No API Key"}
    
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Bạn là chuyên gia y khoa. Chấm điểm tóm tắt (thang 1-10) dựa trên độ chính xác thông tin:
    1. Gốc: "{source}"
    2. Chuẩn: "{reference}"
    3. Máy tạo: "{prediction}"
    Trả về JSON duy nhất: {{"score": <số>, "reason": "<ngắn gọn>"}}
    """
    try:
        res = model_gemini.generate_content(prompt)
        cleaned_text = res.text.replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_text)
    except Exception as e:
        return {"score": 0, "reason": f"API Error: {str(e)}"}

# --- 3. CHẠY TEST ---
print("\n🔎 Đang đánh giá kết quả...")

# Lấy 3 mẫu test nhanh (để chạy nhanh hơn, bạn có thể tăng số lượng nếu muốn)
test_samples = eval_ds.select(range(3))

results = []
preds_bleu, refs_bleu = [], []

for sample in tqdm(test_samples):
    src, ref = sample['text'], sample['summary']
    pred = generate_summary(src, model_to_eval, tokenizer)
    gemini_res = evaluate_with_gemini(src, ref, pred)
    time.sleep(1.5)  # Delay để tránh vượt rate limit của Gemini API
    
    preds_bleu.append(pred)
    refs_bleu.append([ref])
    results.append({
        "Src": src[:50] + "...",
        "Ref": ref,
        "Pred": pred,
        "Score": gemini_res['score'],
        "Reason": gemini_res['reason']
    })

# --- 4. KẾT QUẢ ---
bleu = sacrebleu.corpus_bleu(preds_bleu, refs_bleu)
avg_gemini = sum(r['Score'] for r in results) / len(results) if results else 0

print(f"\n📊 KẾT QUẢ CUỐI CÙNG:")
print(f"- BLEU Score: {bleu.score:.2f}")
print(f"- Gemini Score: {avg_gemini:.2f}/10")

# In mẫu đầu tiên để xem chi tiết
if results:
    print(f"\n[Mẫu thử 1]")
    print(f"Ref: {results[0]['Ref']}")
    print(f"Pred: {results[0]['Pred']}")
    print(f"Gemini chấm: {results[0]['Score']} điểm ({results[0]['Reason']})")