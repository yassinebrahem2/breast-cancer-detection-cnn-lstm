"""
Complete the analysis after early stopping training
Runs the exact same test/results/plotting code from train.py
"""

import os
import sys
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.dataset import create_dataloaders
from scripts.cnn_lstm import CNNLSTM


def parse_training_log(log_file):
    """Parse training.log to extract metrics history"""
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match epoch lines: "Epoch X/Y - Train Loss: 0.xxxx, Train Acc: xx.xx% - Val Loss: 0.xxxx, Val Acc: xx.xx%"
            match = re.search(r'Epoch \d+/\d+ - Train Loss: ([\d.]+), Train Acc: ([\d.]+)% - Val Loss: ([\d.]+), Val Acc: ([\d.]+)%', line)
            if match:
                train_loss, train_acc, val_loss, val_acc = match.groups()
                history['train_loss'].append(float(train_loss))
                history['train_acc'].append(float(train_acc))
                history['val_loss'].append(float(val_loss))
                history['val_acc'].append(float(val_acc))
    
    return history


def test(model, test_loader, device, config):
    """Test the model and compute paper metrics - SAME AS train.py"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Paper metrics
    results = {}
    
    if 'accuracy' in config['evaluation.metrics']:
        results['accuracy'] = accuracy_score(all_labels, all_preds) * 100
    
    if 'sensitivity' in config['evaluation.metrics']:
        results['sensitivity'] = recall_score(all_labels, all_preds, pos_label=1) * 100
    
    if 'specificity' in config['evaluation.metrics']:
        results['specificity'] = recall_score(all_labels, all_preds, pos_label=0) * 100
    
    if 'f_score' in config['evaluation.metrics']:
        results['f_score'] = f1_score(all_labels, all_preds) * 100
    
    if 'auc' in config['evaluation.metrics']:
        results['auc'] = roc_auc_score(all_labels, all_probs)
    
    if config['evaluation.save_predictions']:
        results['predictions'] = all_preds.tolist()
        results['labels'] = all_labels.tolist()
        results['probabilities'] = all_probs.tolist()
    
    return results


def main():
    """Main analysis function - Runs the exact test/results section from train.py"""
    output_dir = 'outputs'
    
    print("="*70)
    print("COMPLETING ANALYSIS AFTER EARLY STOP")
    print("="*70)
    
    # 1. Parse training history from log
    log_file = os.path.join(output_dir, 'training.log')
    if not os.path.exists(log_file):
        print(f"❌ Error: Training log not found at {log_file}")
        return
    
    print(f"\nParsing training history from: {log_file}")
    history = parse_training_log(log_file)
    print(f"✓ Found {len(history['train_loss'])} training epochs")
    
    # 2. Load configuration
    config_file = os.path.join(output_dir, 'train_config.json')
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file not found at {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    # Create Config object to match train.py
    class Config:
        def __init__(self, config):
            self.config = config
        def get(self, key, default=None):
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, default)
                else:
                    return default
            return value
        def __getitem__(self, key):
            return self.get(key)
    
    config = Config(config_dict)
    
    # 3. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 4. Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=config['data.metadata_path'],
        batch_size=config['data.batch_size'],
        num_workers=config['data.num_workers']
    )
    print(f"✓ Test batches: {len(test_loader):,}")
    
    # 5. Load model
    print("\nLoading best model...")
    model_path = os.path.join(output_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    model = CNNLSTM(
        cnn_filters=config['model.cnn_filters'],
        cnn_kernels=[tuple(k) for k in config['model.cnn_kernels']],
        lstm_units=config['model.lstm_units'],
        lstm_dropout=config['model.lstm_dropout'],
        lstm_recurrent_dropout=config['model.lstm_recurrent_dropout'],
        num_classes=config['model.num_classes']
    ).to(device)
    
    # Load best model state
    best_model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(best_model_state)
    print(f"✓ Best model loaded")
    
    # ===== EXACTLY AS train.py - Test Section =====
    print(f"\n{'='*70}")
    print("TESTING")
    print('='*70)
    
    test_metrics = test(model, test_loader, device, config)
    
    if config['logging.verbose']:
        print(f"\nTest Results (Paper Metrics):")
        for metric, value in test_metrics.items():
            if metric not in ['predictions', 'labels', 'probabilities']:
                if isinstance(value, float):
                    suffix = '%' if metric != 'auc' else ''
                    print(f"  {metric.capitalize()}: {value:.2f}{suffix}")
    
    # Save results - EXACTLY AS train.py
    # Note: epoch variable doesn't exist, use history length
    epoch = len(history['train_loss']) - 1
    best_epoch = int(np.argmin(history['val_loss']))
    
    results = {
        'test_metrics': {k: v for k, v in test_metrics.items() 
                        if k not in ['predictions', 'labels', 'probabilities']},
        'history': history,
        'config': {
            'epochs_trained': epoch + 1,
            'best_epoch': best_epoch + 1,
            'device': str(device),
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    if config['logging.verbose']:
        print(f"\n✓ Results saved to: {os.path.join(output_dir, 'results.json')}")
    
    # Save predictions if requested - EXACTLY AS train.py
    if config['evaluation.save_predictions'] and 'predictions' in test_metrics:
        predictions = {
            'predictions': test_metrics['predictions'],
            'labels': test_metrics['labels'],
            'probabilities': test_metrics['probabilities']
        }
        with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions, f)
        print(f"✓ Predictions saved to predictions.json")
    
    # Plot training history - EXACTLY AS train.py
    if config['logging.plot_results']:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Loss', fontsize=11)
        plt.title('Loss over Epochs', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Accuracy (%)', fontsize=11)
        plt.title('Accuracy over Epochs', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
        
        if config['logging.verbose']:
            print(f"✓ Training history saved to: {os.path.join(output_dir, 'training_history.png')}")
    
    # ===== Final Summary =====
    print(f"\n{'='*70}")
    print("✓ TRAINING COMPLETE")
    print('='*70)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - best_model.pth (trained weights)")
    print(f"  - results.json (metrics & history)")
    print(f"  - training.log (training log)")
    print(f"  - train_config.json (configuration used)")
    if config['logging.save_checkpoints']:
        print(f"  - checkpoint_epoch_*.pth (intermediate checkpoints)")
    if config['evaluation.save_predictions']:
        print(f"  - predictions.json (model predictions)")
    if config['logging.plot_results']:
        print(f"  - training_history.png (plots)")


if __name__ == '__main__':
    main()
