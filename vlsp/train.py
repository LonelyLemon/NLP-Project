import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from vlsp.config import Config
from vlsp.data_loader import get_formatted_dataset

def train():
    # 1. Load Data
    dataset = get_formatted_dataset(Config)
    print(f"Train size: {len(dataset['train'])}, Val size: {len(dataset['test'])}")

    # 2. Config Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=Config.USE_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Config LoRA
    peft_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACC_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=0.001,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=Config.MAX_SEQ_LENGTH,
    )

    # 8. Start Training
    print("Starting training...")
    trainer.train()
    
    # 9. Save Model
    print("Saving adapter...")
    trainer.model.save_pretrained(Config.NEW_MODEL_NAME)
    tokenizer.save_pretrained(Config.NEW_MODEL_NAME)
    print("Done!")

if __name__ == "__main__":
    train()