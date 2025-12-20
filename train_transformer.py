"""
Main training script cho Transformer Machine Translation
IWSLT2015 English-Vietnamese Dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.dataset import Vocabulary, BilingualDataset, Collate, PAD_TOKEN, SubwordVocabulary, SpmBilingualDataset
from src.model.transformer import Transformer
from src.train import Trainer, create_optimizer, create_scheduler, WarmupScheduler
from src.inference import GreedySearchDecoder, BeamSearchDecoder
from src.evaluate import Evaluator
from src.visualization import generate_all_plots
from src.data_processor import preprocess_dataset
import os
import sentencepiece as spm

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")

    CONFIG = {
        'src': 'en',
        'trg': 'vi',
        'use_subword': True,
        'use_rope': True,
        'vocab_size_en': 12000,
        'vocab_size_vi': 7000,
        'vocab_model_type': 'unigram',
        'num_workers': 4,

        'model_dim': 384,
        'num_heads': 6,
        'num_enc_layers': 4,
        'num_dec_layers': 4,
        'ff_hidden_dim': 1536,
        'dropout': 0.1,
        'max_len_en': 150,
        'max_len_vi': 180,
        
        'batch_size': 8,
        'num_epochs': 20,
        'learning_rate': 5e-5,
        'weight_decay': 1e-5,
        'warmup_steps': 4000,
        'patience': 5,  # Early stopping

        'freq_threshold': 2,
        

        'beam_size': 5,
        'length_penalty': 0.6,
    }

    for key, value in CONFIG.items():
        if key in config:
            CONFIG[key] = config[key]
    
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION")
    print("="*60)
    for key, value in CONFIG.items():
        print(f"  {key:<20}: {value}")
    print("="*60 + "\n")
    
    # ==================== 1. LOAD DATA ====================
    
    print("\n" + "="*60)
    print("📥 LOADING IWSLT2015 DATASET")
    print("="*60)
    dataset = load_dataset('thainq107/iwslt2015-en-vi')
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    train_dataset = preprocess_dataset(train_dataset)
    val_dataset = preprocess_dataset(val_dataset)
    test_dataset = preprocess_dataset(test_dataset)
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")
    print(f"  Test samples:  {len(test_dataset):,}")
    print("="*60 + "\n")
    
    # ==================== 2. BUILD VOCABULARY ====================
    
    print("\n" + "="*60)
    print("📚 BUILDING VOCABULARY")
    print("="*60)

    with open("temp_train.en", "w", encoding="utf-8") as f_en, \
     open("temp_train.vi", "w", encoding="utf-8") as f_vi:
        for x in train_dataset:
            f_en.write(x["en"].strip() + "\n")
            f_vi.write(x["vi"].strip() + "\n")
    
    spm.SentencePieceTrainer.train(
        input="temp_train.en",
        model_prefix="spm_en",
        vocab_size=CONFIG['vocab_size_en'],
        model_type=CONFIG['vocab_model_type'],
        character_coverage=1.0,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )

    spm.SentencePieceTrainer.train(
        input="temp_train.vi",
        model_prefix="spm_vi",
        vocab_size=CONFIG['vocab_size_vi'],
        model_type=CONFIG['vocab_model_type'],
        character_coverage=0.9995,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )
    os.remove("temp_train.en")
    os.remove("temp_train.vi")
    
    src_sentences = [x[CONFIG['src']] for x in train_dataset]
    trg_sentences = [x[CONFIG['trg']] for x in train_dataset]
    
    if CONFIG['use_subword']:
        src_vocab = SubwordVocabulary(f"spm_{CONFIG['src']}.model")
        trg_vocab = SubwordVocabulary(f"spm_{CONFIG['trg']}.model")
    else:
        src_vocab = Vocabulary(freq_threshold=CONFIG['freq_threshold'])
        src_vocab.build_vocabulary(src_sentences)
        
        trg_vocab = Vocabulary(freq_threshold=CONFIG['freq_threshold'])
        trg_vocab.build_vocabulary(trg_sentences)
    
    
    # ==================== 3. CREATE DATALOADERS ====================
    
    print("\n" + "="*60)
    print("🔄 CREATING DATALOADERS")
    print("="*60)
    
    if CONFIG['use_subword']:
        train_data = SpmBilingualDataset(
            train_dataset, 
            src_vocab=src_vocab, 
            trg_vocab=trg_vocab, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
            src_lang=CONFIG['src'], 
            trg_lang=CONFIG['trg']
        )
        val_data = SpmBilingualDataset(
            val_dataset, 
            src_vocab=src_vocab, 
            trg_vocab=trg_vocab, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
            src_lang=CONFIG['src'], 
            trg_lang=CONFIG['trg']
        )
        test_data = SpmBilingualDataset(
            test_dataset, 
            src_vocab=src_vocab, 
            trg_vocab=trg_vocab, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
            src_lang=CONFIG['src'], 
            trg_lang=CONFIG['trg']
        )
    else:
        train_data = BilingualDataset(
            train_dataset, 
            src_vocab=src_vocab, 
            trg_vocab=trg_vocab, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
            src_lang=CONFIG['src'], 
            trg_lang=CONFIG['trg']
        )
        val_data = BilingualDataset(
            val_dataset, 
            src_vocab=src_vocab, 
            trg_vocab=trg_vocab, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
            src_lang=CONFIG['src'], 
            trg_lang=CONFIG['trg']
        )
        test_data = BilingualDataset(
            test_dataset, 
            src_vocab=src_vocab, 
            trg_vocab=trg_vocab, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
            src_lang=CONFIG['src'], 
            trg_lang=CONFIG['trg']
        )
    
    src_pad_idx = src_vocab.pad_idx
    trg_pad_idx = trg_vocab.pad_idx
    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=Collate(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"]
        ),
        num_workers=CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=Collate(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"]
        ),
        num_workers=CONFIG['num_workers']
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=Collate(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"]
        ),
        num_workers=CONFIG['num_workers']
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print("="*60 + "\n")
    
    # ==================== 4. CREATE MODEL ====================
    
    print("\n" + "="*60)
    print("🏗️  CREATING TRANSFORMER MODEL")
    print("="*60)
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(trg_vocab),
        model_dim=CONFIG['model_dim'],
        num_heads=CONFIG['num_heads'],
        num_enc_layers=CONFIG['num_enc_layers'],
        num_dec_layers=CONFIG['num_dec_layers'],
        ff_hidden_dim=CONFIG['ff_hidden_dim'],
        max_len_src=CONFIG[f"max_len_{CONFIG['src']}"],
        max_len_trg=CONFIG[f"max_len_{CONFIG['trg']}"],
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("="*60 + "\n")
    
    # ==================== 5. SETUP TRAINING ====================
    
    print("\n" + "="*60)
    print("🎯 SETUP TRAINING")
    print("="*60)
    
    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.05)
    
    # Optimizer (learning rate sẽ được điều chỉnh bởi warmup scheduler)
    optimizer = create_optimizer(
        model,
        learning_rate=CONFIG['learning_rate'],  # Base LR, sẽ được warmup scheduler điều chỉnh
        weight_decay=CONFIG['weight_decay']
    )
    
    # Warmup scheduler (theo paper "Attention is All You Need")
    warmup_scheduler = WarmupScheduler(
        optimizer,
        d_model=CONFIG['model_dim'],
        warmup_steps=CONFIG['warmup_steps']
    )
    
    # Learning rate scheduler (sau warmup)
    plateau_scheduler = create_scheduler(
        optimizer,
        mode='plateau',
        factor=0.5,
        patience=3
    )
    
    print(f"  Loss function: CrossEntropyLoss (ignore_index={trg_pad_idx}, label_smoothing=0.05)")
    print(f"  Optimizer: Adam (base_lr=1.0, weight_decay={CONFIG['weight_decay']})")
    print(f"  Warmup Scheduler: {CONFIG['warmup_steps']} steps")
    print(f"  Plateau Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
    print("="*60 + "\n")
    
    # ==================== 6. TRAINING ====================
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        warmup_scheduler=warmup_scheduler,
        plateau_scheduler=plateau_scheduler,
        patience=CONFIG['patience']
    )
    
    # ==================== 7. LOAD BEST MODEL ====================
    
    print("\n" + "="*60)
    print("📦 LOADING BEST MODEL")
    print("="*60)
    
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['val_loss']:.4f}")
    print("="*60 + "\n")
    
    # ==================== 8. INFERENCE & EVALUATION ====================
    
    print("\n" + "="*60)
    print("🔍 INFERENCE & EVALUATION")
    print("="*60 + "\n")
    
    # Create decoders
    greedy_decoder = GreedySearchDecoder(model, max_len=100, use_subword=CONFIG['use_subword'])
    beam_decoder = BeamSearchDecoder(
        model,
        beam_size=CONFIG['beam_size'],
        max_len=CONFIG[f"max_len_{CONFIG['trg']}"],
        length_penalty=CONFIG['length_penalty'],
        use_subword=CONFIG['use_subword']
    )
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, src_vocab, trg_vocab, device, use_subword=CONFIG['use_subword'])
    comparison_results = evaluator.compare_decoders(greedy_decoder, beam_decoder)
    
    # ==================== 9. VISUALIZATION ====================
    
    generate_all_plots(
        history_path='logs/training_history.json',
        comparison_results=comparison_results,
        save_dir='figures'
    )
    
    # ==================== DONE ====================
    
    print("\n" + "="*60)
    print("✅ ALL TASKS COMPLETED!")
    print("="*60)
    print("\n📁 Output files:")
    print("  - checkpoints/best_model.pt")
    print("  - logs/training_history.json")
    print("  - figures/training_curves.png")
    print("  - figures/metrics_comparison.png")
    print("  - figures/loss_histogram.png")
    print("  - figures/summary_table.png")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
