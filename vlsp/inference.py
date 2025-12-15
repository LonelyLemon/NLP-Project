import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vlsp.config import Config
import sacrebleu
from tqdm import tqdm

def generate_translation(model, tokenizer, text, direction="en2vi"):
    if direction == "en2vi":
        prompt = f"system\nBạn là một trợ lý dịch thuật y tế chuyên nghiệp.\nuser\nDịch câu sau sang tiếng Việt: {text}\nassistant\n"
    else:
        prompt = f"system\nYou are a professional medical translation assistant.\nuser\nTranslate the following sentence to English: {text}\nassistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            num_beams=4,
            early_stopping=True
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = decoded.split("assistant\n")[-1].strip()
    return result

def evaluate():
    # Load Base Model & Adapter
    print("Loading fine-tuned model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.NEW_MODEL_NAME)
    
    # Merge adapter
    model = PeftModel.from_pretrained(base_model, Config.NEW_MODEL_NAME)
    model.eval()

    # Load Test Data
    with open(Config.TEST_EN, 'r', encoding='utf-8') as f:
        sources = [line.strip() for line in f.readlines()]
    with open(Config.TEST_VI, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f.readlines()]

    predictions = []
    print("Translating...")
    
    for src in tqdm(sources):
        pred = generate_translation(model, tokenizer, src, direction="en2vi")
        predictions.append(pred)

    # Lưu kết quả
    with open("prediction.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))

    # Tính BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"BLEU Score: {bleu.score}")

if __name__ == "__main__":
    evaluate()