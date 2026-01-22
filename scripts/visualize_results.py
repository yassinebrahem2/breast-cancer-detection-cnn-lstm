"""
Visualize training results and generate publication-ready figures
Generates: Training curves, Confusion Matrix, ROC Curve
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Results')
    parser.add_argument('--exp-dir', type=str, default='experiments/latest',
                       help='Experiment directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: exp_dir/results)')
    
    return parser.parse_args()


def plot_training_history(history_file, output_dir):
    """Plot training and validation metrics over epochs"""
    print("Generating training history plots...")
    
    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History - CNN-LSTM Model', fontsize=16, fontweight='bold')
    
    # Accuracy
    axes[0, 0].plot(history['train_acc'], label='Train', linewidth=2, color='#2ecc71')
    axes[0, 0].plot(history['val_acc'], label='Validation', linewidth=2, color='#e74c3c')
    axes[0, 0].set_title('Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['train_loss'], label='Train', linewidth=2, color='#2ecc71')
    axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2, color='#e74c3c')
    axes[0, 1].set_title('Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # AUC
    if 'train_auc' in history:
        axes[1, 0].plot(history['train_auc'], label='Train', linewidth=2, color='#2ecc71')
        axes[1, 0].plot(history['val_auc'], label='Validation', linewidth=2, color='#e74c3c')
        axes[1, 0].set_title('AUC', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], linewidth=2, color='#3498db')
        axes[1, 1].set_title('Learning Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_confusion_matrix(results_dir, output_dir):
    """Plot confusion matrix"""
    print("Generating confusion matrix...")
    
    y_true = np.load(os.path.join(results_dir, 'y_true.npy'))
    y_pred_proba = np.load(os.path.join(results_dir, 'y_pred_proba.npy'))
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['IDC-Negative', 'IDC-Positive'],
                yticklabels=['IDC-Negative', 'IDC-Positive'],
                cbar_kws={'label': 'Count'})
    
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_roc_curve(results_dir, output_dir):
    """Plot ROC curve"""
    print("Generating ROC curve...")
    
    y_true = np.load(os.path.join(results_dir, 'y_true.npy'))
    y_pred_proba = np.load(os.path.join(results_dir, 'y_pred_proba.npy'))
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2.5,
            label=f'CNN-LSTM (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier (AUC = 0.5)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def main():
    args = parse_args()
    
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("VISUALIZING RESULTS")
    print("="*70)
    print(f"Experiment: {exp_dir}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Plot training history
    history_file = exp_dir / 'logs' / 'training_history.pkl'
    if history_file.exists():
        plot_training_history(history_file, output_dir)
    else:
        print(f"⚠️  Training history not found: {history_file}")
    
    # Plot confusion matrix and ROC curve
    results_dir = exp_dir / 'results'
    if (results_dir / 'y_true.npy').exists():
        plot_confusion_matrix(results_dir, output_dir)
        plot_roc_curve(results_dir, output_dir)
    else:
        print(f"⚠️  Evaluation results not found. Run evaluate.py first.")
    
    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")


if __name__ == '__main__':
    main()
