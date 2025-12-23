import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_training_curves(history_path, save_dir='figures'):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['learning_rates'], 'g-^', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()


def plot_metrics_comparison(comparison_results, save_dir='figures'):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    methods = ['Greedy', 'Beam']
    bleu_scores = [comparison_results['greedy_bleu'], comparison_results['beam_bleu']]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, bleu_scores, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('BLEU Score')
    ax.set_ylim(0, max(bleu_scores) * 1.2)
    ax.set_title('BLEU Score Comparison')

    for bar, score in zip(bars, bleu_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{score:.2f}',
                ha='center', va='bottom')

    plt.tight_layout()
    save_path = save_dir / 'bleu_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Saved BLEU comparison figure at", save_path)
    plt.show()


def plot_loss_histogram(history_path, save_dir='figures'):
    with open(history_path, 'r') as f:
        history = json.load(f)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(history['train_loss'], bins=20, alpha=0.5, label='Train', color='blue')
    ax.hist(history['val_loss'], bins=20, alpha=0.5, label='Val', color='red')

    ax.set_xlabel('Loss')
    ax.set_ylabel('Frequency')
    ax.set_title('Loss Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'loss_histogram.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Saved loss histogram at", save_path)
    plt.show()


def create_summary_table(comparison_results, history_path, save_dir='figures'):
    with open(history_path, 'r') as f:
        history = json.load(f)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Chuẩn bị dữ liệu summary
    summary = {
        'Training': {
            'Total Epochs': len(history['train_loss']),
            'Final Train Loss': f"{history['train_loss'][-1]:.4f}",
            'Final Val Loss': f"{history['val_loss'][-1]:.4f}",
            'Best Val Loss': f"{min(history['val_loss']):.4f}",
            'Final LR': f"{history['learning_rates'][-1]:.6f}"
        },
        'Greedy': {
            'BLEU Score': f"{comparison_results['greedy_bleu']:.2f}"
        },
        'Beam': {
            'BLEU Score': f"{comparison_results['beam_bleu']:.2f}"
        },
        'Improvement': {
            'BLEU': f"{comparison_results['improvement']:+.2f}"
        }
    }

    # Tạo figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    table_data = []
    for section, metrics in summary.items():
        table_data.append([section, ''])
        for k, v in metrics.items():
            table_data.append([k, str(v)])
        table_data.append(['', ''])

    table = ax.table(cellText=table_data, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    save_path = save_dir / 'summary_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Saved summary table at", save_path)
    plt.show()

def generate_all_plots(history_path, comparison_results, save_dir='figures'):
    plot_training_curves(history_path, save_dir)
    plot_metrics_comparison(comparison_results, save_dir)
    plot_loss_histogram(history_path, save_dir)
    create_summary_table(comparison_results, history_path, save_dir)
