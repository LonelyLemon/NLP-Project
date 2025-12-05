"""
Main training script cho Transformer Machine Translation
IWSLT2015 English-Vietnamese Dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.dataset import Vocabulary, BilingualDataset, Collate, PAD_TOKEN
from src.model.transformer import Transformer
from src.train import Trainer, create_optimizer, create_scheduler
from src.inference import GreedySearchDecoder, BeamSearchDecoder
from src.evaluate import Evaluator
from src.visualization import generate_all_plots


def main():
    # ==================== CONFIGURATION ====================
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Hyperparameters
    CONFIG = {
        # Model
        'model_dim': 512,
        'num_heads': 8,
        'num_enc_layers': 6,
        'num_dec_layers': 6,
        'ff_hidden_dim': 2048,
        'dropout': 0.1,
        'max_len': 5000,
        
        # Training
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'patience': 5,  # Early stopping
        
        # Data
        'freq_threshold': 2,  # Minimum word frequency
        'train_split': 'train[:80%]',
        'val_split': 'train[80%:90%]',
        'test_split': 'train[90%:]',
        
        # Inference
        'beam_size': 5,
        'length_penalty': 0.6,
    }
    
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
    
    train_dataset = load_dataset("thainq107/iwslt2015-en-vi", split=CONFIG['train_split'])
    val_dataset = load_dataset("thainq107/iwslt2015-en-vi", split=CONFIG['val_split'])
    test_dataset = load_dataset("thainq107/iwslt2015-en-vi", split=CONFIG['test_split'])
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")
    print(f"  Test samples:  {len(test_dataset):,}")
    print("="*60 + "\n")
    
    # ==================== 2. BUILD VOCABULARY ====================
    
    print("\n" + "="*60)
    print("📚 BUILDING VOCABULARY")
    print("="*60)
    
    src_sentences = [x['en'] for x in train_dataset]
    trg_sentences = [x['vi'] for x in train_dataset]
    
    src_vocab = Vocabulary(freq_threshold=CONFIG['freq_threshold'])
    src_vocab.build_vocabulary(src_sentences)
    
    trg_vocab = Vocabulary(freq_threshold=CONFIG['freq_threshold'])
    trg_vocab.build_vocabulary(trg_sentences)
    
    print(f"  English vocab size: {len(src_vocab):,}")
    print(f"  Vietnamese vocab size: {len(trg_vocab):,}")
    print("="*60 + "\n")
    
    # ==================== 3. CREATE DATALOADERS ====================
    
    print("\n" + "="*60)
    print("🔄 CREATING DATALOADERS")
    print("="*60)
    
    pad_idx = src_vocab.stoi[PAD_TOKEN]
    
    train_data = BilingualDataset(train_dataset, src_vocab, trg_vocab)
    val_data = BilingualDataset(val_dataset, src_vocab, trg_vocab)
    test_data = BilingualDataset(test_dataset, src_vocab, trg_vocab)
    
    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=Collate(pad_idx=pad_idx),
        num_workers=0  # Windows compatibility
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=Collate(pad_idx=pad_idx),
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=1,  # For inference
        shuffle=False,
        collate_fn=Collate(pad_idx=pad_idx),
        num_workers=0
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
        max_len=CONFIG['max_len'],
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
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    
    # Optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = create_scheduler(
        optimizer,
        mode='plateau',
        factor=0.5,
        patience=3
    )
    
    print(f"  Loss function: CrossEntropyLoss (ignore_index={pad_idx}, label_smoothing=0.1)")
    print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
    print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
    print("="*60 + "\n")
    
    # ==================== 6. TRAINING ====================
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        pad_idx=pad_idx,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        scheduler=scheduler,
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
    greedy_decoder = GreedySearchDecoder(model, max_len=100)
    beam_decoder = BeamSearchDecoder(
        model,
        beam_size=CONFIG['beam_size'],
        max_len=100,
        length_penalty=CONFIG['length_penalty']
    )
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, src_vocab, trg_vocab, device)
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
