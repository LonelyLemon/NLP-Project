import os

class Config:
    # Model
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    NEW_MODEL_NAME = "qwen-medical-vlsp-lora"
    
    TRAIN_EN = "data/train.en.txt" 
    TRAIN_VI = "data/train.vi.txt"
    TEST_EN = "data/public_test.en.txt"
    TEST_VI = "data/public_test.vi.txt" 
    
    OUTPUT_DIR = "./results"
    
    # Hyperparameters
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    BATCH_SIZE = 4
    GRAD_ACC_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_SEQ_LENGTH = 4096
    
    # System
    USE_4BIT = True