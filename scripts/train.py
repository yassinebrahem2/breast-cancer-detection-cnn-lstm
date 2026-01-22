"""
CNN-LSTM Training Script - Configuration File Support
Paper: "Breast cancer classification based on hybrid CNN with LSTM model"

Usage:
    python scripts/train.py --config configs/train.json
    python scripts/train.py --config configs/train.json --output-dir custom_outputs
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score
)

# ===== Path setup =====
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# ====================

# Import from same scripts directory
from scripts.dataset import create_dataloaders
from scripts.cnn_lstm import CNNLSTM



import logging

# Setup logging to both console AND file
def setup_logging(output_dir):
    """Setup logging to file and console"""
    log_file = os.path.join(output_dir, 'training.log')
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger




class Config:
    """Configuration loader from JSON"""
    
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get(self, key, default=None):
        """Get config value by dot notation: 'data.batch_size'"""
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
    
    def __repr__(self):
        return json.dumps(self.config, indent=2)


class EarlyStopping:
    """Early stopping based on validation loss (Paper requirement)"""
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        return False


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def test(model, test_loader, device, config):
    """Test the model and compute paper metrics"""
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


def load_config(config_path):
    """Load configuration from JSON file"""
    return Config(config_path)


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN-LSTM model with configuration file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train.json',
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Override device (cuda/cpu) from config'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint file (e.g., outputs/checkpoint_epoch_014.pth)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n{'='*70}")
    print("LOADING CONFIGURATION")
    print('='*70)
    print(f"Config file: {args.config}")
    
    config = load_config(args.config)
    
    if config:
        print(f"✓ Configuration loaded successfully")
    
    # Override with command line args
    output_dir = args.output_dir or config['logging.output_dir']
    device = args.device or config['training.device']
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, 'training.log')
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("="*70)
    logger.info("TRAINING STARTED")
    logger.info("="*70)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'train_config.json'), 'w') as f:
        json.dump(config.config, f, indent=2)
    
    logger.info("Configuration saved to train_config.json")
    
    # ===== SETUP TENSORBOARD =====
    writer = None
    
    if config['logging.save_tensorboard']:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        
        tensorboard_dir = config['logging.tensorboard_dir']
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Create unique run directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(tensorboard_dir, f'run_{timestamp}')
        
        # Create writer with auto-flush every 30 seconds
        writer = SummaryWriter(run_dir, flush_secs=30)
        logger.info(f"TensorBoard logging to: {run_dir}")
        
        print(f"\n{'='*70}")
        print("TENSORBOARD")
        print('='*70)
        print(f"Directory: {run_dir}")
        print(f"View with: tensorboard --logdir {tensorboard_dir} --reload_interval 5")
        print(f"Then open: http://localhost:6006")
    # ============================
    
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print('='*70)
    
    logger.info("Loading data...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=config['data.metadata_path'],
        batch_size=config['data.batch_size'],
        num_workers=config['data.num_workers']
    )
    
    logger.info(f"Data loaded successfully")
    logger.info(f"  Train batches: {len(train_loader):,}")
    logger.info(f"  Val batches: {len(val_loader):,}")
    logger.info(f"  Test batches: {len(test_loader):,}")
    
    # Compute class weights for imbalanced data (if enabled)
    use_class_weights = config.get('training.use_class_weights', False)
    class_weights = None
    
    if use_class_weights:
        print(f"\nComputing class weights...")
        logger.info("Computing class weights from training data...")
        
        from sklearn.utils.class_weight import compute_class_weight
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy())
        train_labels = np.array(train_labels)
        
        class_weights_np = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        
        logger.info(f"Class distribution in training set:")
        logger.info(f"  Class 0 (Negative): {(train_labels == 0).sum():,} samples")
        logger.info(f"  Class 1 (Positive): {(train_labels == 1).sum():,} samples")
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        
        print(f"Class weights calculated:")
        print(f"  Class 0 (Negative): {class_weights[0]:.4f}")
        print(f"  Class 1 (Positive): {class_weights[1]:.4f}")
    else:
        logger.info("Class weights disabled")
    
    print(f"\n{'='*70}")
    print("CREATING MODEL")
    print('='*70)
    
    logger.info("Creating CNN-LSTM model...")
    
    # Create model
    model = CNNLSTM(
        cnn_filters=config['model.cnn_filters'],
        cnn_kernels=[tuple(k) for k in config['model.cnn_kernels']],
        cnn_dropout=config.get('model.cnn_dropout', 0.5),
        lstm_units=config['model.lstm_units'],
        lstm_num_layers=config.get('model.lstm_num_layers', 2),
        lstm_dropout=config['model.lstm_dropout'],
        lstm_recurrent_dropout=config['model.lstm_recurrent_dropout'],
        dense_units=config.get('model.dense_units', None),
        dense_dropout=config.get('model.dense_dropout', 0.2),
        num_classes=config['model.num_classes']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    logger.info(f"Model created successfully")
    logger.info(f"  Device: {device}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion_name = config['training.criterion']
    if criterion_name == 'cross_entropy':
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Loss function: CrossEntropyLoss (with class weights)")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info(f"Loss function: CrossEntropyLoss")
    elif criterion_name == 'bce':
        # For binary classification with 1 output neuron
        if class_weights is not None:
            pos_weight = class_weights[1] / class_weights[0]
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
            logger.info(f"Loss function: BCEWithLogitsLoss (with pos_weight={pos_weight:.4f})")
        else:
            criterion = nn.BCEWithLogitsLoss()
            logger.info(f"Loss function: BCEWithLogitsLoss")
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")
    
    # Optimizer
    optimizer_name = config['training.optimizer'].lower()
    lr = config['training.learning_rate']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config['regularization.l2_weight_decay']
        )
        logger.info(f"Optimizer: Adam (lr={lr}, weight_decay={config['regularization.l2_weight_decay']})")
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=config['regularization.l2_weight_decay']
        )
        logger.info(f"Optimizer: SGD (lr={lr}, momentum=0.9, weight_decay={config['regularization.l2_weight_decay']})")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # LR Scheduler - ReduceLROnPlateau
    use_scheduler = config.get('training.use_lr_scheduler', True)
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('training.lr_scheduler_factor', 0.5),
            patience=config.get('training.lr_scheduler_patience', 3),
            min_lr=config.get('training.lr_scheduler_min_lr', 1e-6)
        )
        logger.info(f"LR Scheduler: ReduceLROnPlateau (factor={config.get('training.lr_scheduler_factor', 0.5)}, "
                   f"patience={config.get('training.lr_scheduler_patience', 3)}, "
                   f"min_lr={config.get('training.lr_scheduler_min_lr', 1e-6)})")
    else:
        scheduler = None
        logger.info("LR Scheduler: Disabled")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['regularization.early_stopping_patience'],
        min_delta=config['regularization.early_stopping_min_delta']
    )
    logger.info(f"Early stopping: patience={config['regularization.early_stopping_patience']}, "
               f"min_delta={config['regularization.early_stopping_min_delta']}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nLoading checkpoint: {args.resume}")
            logger.info(f"Loading checkpoint: {args.resume}")
            
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            history = checkpoint.get('history', history)
            
            print(f"✓ Resumed from epoch {start_epoch}")
            print(f"  Previous train loss: {checkpoint['train_loss']:.4f}")
            print(f"  Previous val loss: {checkpoint['val_loss']:.4f}")
            logger.info(f"✓ Resumed from epoch {start_epoch}")
            logger.info(f"  Previous train loss: {checkpoint['train_loss']:.4f}")
            logger.info(f"  Previous val loss: {checkpoint['val_loss']:.4f}")
        else:
            print(f"\n⚠️  Checkpoint file not found: {args.resume}")
            print("Starting training from scratch...")
            logger.warning(f"Checkpoint file not found: {args.resume}")
            logger.info("Starting training from scratch...")
    
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print('='*70)
    print(f"Epochs: {config['training.epochs']}")
    print(f"Batch size: {config['data.batch_size']}")
    print(f"Learning rate: {config['training.learning_rate']}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Early stopping patience: {config['regularization.early_stopping_patience']}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Save checkpoints: {config['logging.save_checkpoints']}")
    if config['logging.save_checkpoints']:
        print(f"Checkpoint interval: {config['logging.checkpoint_interval']} epochs")
    print(f"TensorBoard: {config['logging.save_tensorboard']}")
    
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {config['training.epochs']}")
    logger.info(f"  Batch size: {config['data.batch_size']}")
    logger.info(f"  Learning rate: {config['training.learning_rate']}")
    logger.info(f"  Optimizer: {optimizer_name}")
    logger.info(f"  Early stopping patience: {config['regularization.early_stopping_patience']}")
    logger.info(f"  Save checkpoints: {config['logging.save_checkpoints']}")
    if config['logging.save_checkpoints']:
        logger.info(f"  Checkpoint interval: {config['logging.checkpoint_interval']} epochs")
    logger.info(f"  TensorBoard: {config['logging.save_tensorboard']}")
    
    print(f"\n{'='*70}")
    print("TRAINING")
    print('='*70)
    
    logger.info("Starting training loop...")
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # If resuming, set best_val_loss from history
    if start_epoch > 0 and history['val_loss']:
        best_val_loss = min(history['val_loss'])
        logger.info(f"Best validation loss from history: {best_val_loss:.4f}")
    
    for epoch in range(start_epoch, config['training.epochs']):
        if config['logging.verbose']:
            print(f"\nEpoch {epoch+1}/{config['training.epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update learning rate based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if config['logging.verbose']:
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Log to file
        logger.info(f"Epoch {epoch+1}/{config['training.epochs']} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log learning rate changes
        if old_lr != current_lr:
            logger.info(f"→ Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")
            if config['logging.verbose']:
                print(f"  → Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")
        
        # ===== LOG TO TENSORBOARD =====
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
            writer.add_scalar('Accuracy/val', val_acc, epoch + 1)
            writer.add_scalar('Learning_rate', current_lr, epoch + 1)
            writer.flush()  # ✅ Flush data so TensorBoard can read it immediately
        # ==============================
        
        # Save best model
        if val_loss < best_val_loss and config['logging.save_best_model']:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))
            if config['logging.verbose']:
                print("  ✓ Best model saved!")
            logger.info(f"✓ Best model saved (val_loss={val_loss:.4f})")
        
        # ===== SAVE CHECKPOINTS =====
        if config['logging.save_checkpoints']:
            checkpoint_interval = config['logging.checkpoint_interval']
            
            # Save every N epochs
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    output_dir, 
                    f'checkpoint_epoch_{epoch+1:03d}.pth'
                )
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'history': history,
                }
                torch.save(checkpoint, checkpoint_path)
                if config['logging.verbose']:
                    print(f"  ✓ Checkpoint saved: epoch {epoch+1}")
                logger.info(f"✓ Checkpoint saved: checkpoint_epoch_{epoch+1:03d}.pth")
        # ===========================
        
        # Early stopping check
        if early_stopping(val_loss, epoch):
            if config['logging.verbose']:
                print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best epoch: {early_stopping.best_epoch+1}")
            logger.info(f"✓ Early stopping triggered at epoch {epoch+1}")
            logger.info(f"  Best epoch: {early_stopping.best_epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model for testing")
    
    # Test
    print(f"\n{'='*70}")
    print("TESTING")
    print('='*70)
    
    logger.info("Testing model on test set...")
    
    test_metrics = test(model, test_loader, device, config)
    
    if config['logging.verbose']:
        print(f"\nTest Results (Paper Metrics):")
        for metric, value in test_metrics.items():
            if metric not in ['predictions', 'labels', 'probabilities']:
                if isinstance(value, float):
                    suffix = '%' if metric != 'auc' else ''
                    print(f"  {metric.capitalize()}: {value:.2f}{suffix}")
    
    # Log test results
    logger.info("Test Results (Paper Metrics):")
    for metric, value in test_metrics.items():
        if metric not in ['predictions', 'labels', 'probabilities']:
            if isinstance(value, float):
                suffix = '%' if metric != 'auc' else ''
                logger.info(f"  {metric.capitalize()}: {value:.2f}{suffix}")
    
    # ===== LOG TEST METRICS TO TENSORBOARD =====
    if writer:
        for metric, value in test_metrics.items():
            if metric not in ['predictions', 'labels', 'probabilities']:
                if isinstance(value, float):
                    writer.add_scalar(f'Test/{metric}', value)
        writer.flush()  # ✅ Final flush for test metrics
    # ==========================================
    
    # Save results
    results = {
        'test_metrics': {k: v for k, v in test_metrics.items() 
                        if k not in ['predictions', 'labels', 'probabilities']},
        'history': history,
        'config': {
            'epochs_trained': epoch + 1,
            'best_epoch': early_stopping.best_epoch + 1,
            'device': str(device),
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    if config['logging.verbose']:
        print(f"\n✓ Results saved to: {os.path.join(output_dir, 'results.json')}")
    
    logger.info(f"✓ Results saved to results.json")
    
    # Save predictions if requested
    if config['evaluation.save_predictions'] and 'predictions' in test_metrics:
        predictions = {
            'predictions': test_metrics['predictions'],
            'labels': test_metrics['labels'],
            'probabilities': test_metrics['probabilities']
        }
        with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions, f)
        logger.info(f"✓ Predictions saved to predictions.json")
    
    # Plot training history
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
        
        logger.info(f"✓ Training history saved to training_history.png")
    
    # ===== CLOSE TENSORBOARD WRITER =====
    if writer:
        writer.flush()  # ✅ Final flush before closing
        writer.close()
        logger.info("TensorBoard logging closed")
    # ====================================
    
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
    
    if config['logging.save_tensorboard']:
        print(f"\nTensorBoard:")
        print(f"  - Run: tensorboard --logdir {config['logging.tensorboard_dir']} --reload_interval 5")
        print(f"  - View at: http://localhost:6006")
    
    logger.info("="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total epochs trained: {epoch + 1}")
    logger.info(f"Best epoch: {early_stopping.best_epoch + 1}")
    logger.info("Training finished successfully")






if __name__ == '__main__':
    main()
