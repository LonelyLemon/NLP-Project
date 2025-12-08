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
    """
    Vẽ training và validation loss curves.
    
    Args:
        history_path: str - Path to training_history.json
        save_dir: str - Directory để lưu figures
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Learning rate plot
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
    """
    Vẽ biểu đồ so sánh BLEU và ROUGE-L giữa Greedy và Beam Search.
    
    Args:
        comparison_results: dict - Results từ Evaluator.compare_decoders()
        save_dir: str - Directory để lưu figures
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Extract data
    methods = ['Greedy Search', 'Beam Search']
    bleu_scores = [
        comparison_results['greedy']['bleu'],
        comparison_results['beam']['bleu']
    ]
    rouge_scores = [
        comparison_results['greedy']['rouge_l'],
        comparison_results['beam']['rouge_l']
    ]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # BLEU comparison
    colors = ['#3498db', '#e74c3c']
    bars1 = ax1.bar(methods, bleu_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('BLEU Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(bleu_scores) * 1.2)
    
    # Add value labels on bars
    for bar, score in zip(bars1, bleu_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ROUGE-L comparison
    bars2 = ax2.bar(methods, rouge_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('ROUGE-L F1 Score', fontsize=12)
    ax2.set_title('ROUGE-L Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(rouge_scores) * 1.2)
    
    for bar, score in zip(bars2, rouge_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = save_dir / 'metrics_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()


def plot_loss_histogram(history_path, save_dir='figures'):
    """
    Vẽ histogram distribution của train/val loss.
    
    Args:
        history_path: str - Path to training_history.json
        save_dir: str - Directory để lưu figures
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(history['train_loss'], bins=20, alpha=0.5, label='Train Loss', color='blue', edgecolor='black')
    ax.hist(history['val_loss'], bins=20, alpha=0.5, label='Val Loss', color='red', edgecolor='black')
    
    ax.set_xlabel('Loss Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Loss Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = save_dir / 'loss_histogram.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()


def create_summary_table(comparison_results, history_path, save_dir='figures'):
    """
    Tạo summary table với tất cả metrics.
    
    Args:
        comparison_results: dict - Results từ comparison
        history_path: str - Path to training history
        save_dir: str - Directory để lưu
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Prepare data
    summary = {
        'Training Summary': {
            'Total Epochs': len(history['train_loss']),
            'Final Train Loss': f"{history['train_loss'][-1]:.4f}",
            'Final Val Loss': f"{history['val_loss'][-1]:.4f}",
            'Best Val Loss': f"{min(history['val_loss']):.4f}",
            'Final LR': f"{history['learning_rates'][-1]:.6f}"
        },
        'Greedy Search': {
            'BLEU Score': f"{comparison_results['greedy']['bleu']:.2f}",
            'ROUGE-L F1': f"{comparison_results['greedy']['rouge_l']:.4f}",
            'Samples': comparison_results['greedy']['num_samples']
        },
        'Beam Search': {
            'BLEU Score': f"{comparison_results['beam']['bleu']:.2f}",
            'ROUGE-L F1': f"{comparison_results['beam']['rouge_l']:.4f}",
            'Samples': comparison_results['beam']['num_samples']
        },
        'Improvement (Beam vs Greedy)': {
            'BLEU': f"{comparison_results['improvement']['bleu']:+.2f}",
            'ROUGE-L': f"{comparison_results['improvement']['rouge_l']:+.4f}"
        }
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for section, metrics in summary.items():
        table_data.append([section, '', ''])
        table_data.append(['─' * 30, '─' * 20, '─' * 10])
        for key, value in metrics.items():
            table_data.append(['  ' + key, str(value), ''])
        table_data.append(['', '', ''])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header rows
    for i, row in enumerate(table_data):
        if row[1] == '':
            for j in range(3):
                cell = table[(i, j)]
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
    
    plt.title('Training & Evaluation Summary', fontsize=16, fontweight='bold', pad=20)
    
    save_path = save_dir / 'summary_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()


def generate_all_plots(history_path, comparison_results, save_dir='figures'):
    """
    Generate tất cả plots cùng lúc.
    
    Args:
        history_path: str - Path to training_history.json
        comparison_results: dict - Results từ evaluation
        save_dir: str - Directory để lưu figures
    """
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    plot_training_curves(history_path, save_dir)
    plot_metrics_comparison(comparison_results, save_dir)
    plot_loss_histogram(history_path, save_dir)
    create_summary_table(comparison_results, history_path, save_dir)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print(f"📁 Saved to: {save_dir}/")
    print("="*60 + "\n")
