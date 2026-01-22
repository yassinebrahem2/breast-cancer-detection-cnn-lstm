"""
Evaluate trained CNN-LSTM model on test set
Generates: Confusion Matrix, ROC Curve, Classification Report
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_lstm import CNNLSTM
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CNN-LSTM Model')
    parser.add_argument('--checkpoint', type=str, 
                       default='experiments/latest/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to preprocessed data')
    parser.add_argument('--output-dir', type=str, 
                       default='experiments/latest/results',
                       help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU id (-1 for CPU)')
    
    return parser.parse_args()


def load_test_data(data_dir):
    """Load test data"""
    print(f"Loading test data from {data_dir}...")
    
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"✓ Test: {X_test.shape}")
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2) / 255.0
    y_test_tensor = torch.FloatTensor(y_test)
    
    return X_test_tensor, y_test_tensor


def evaluate_model(model, X_test, y_test, device):
    """Run inference and calculate metrics"""
    model.eval()
    
    print("\nRunning inference...")
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred_proba = model(X_test).cpu().numpy().flatten()
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = y_test.numpy().astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    metrics['specificity'] = specificity
    
    return metrics, cm, y_pred_proba, y_true


def print_results(metrics, cm):
    """Print evaluation results"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f} (Sensitivity)")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    print(f"\nPaper Comparison:")
    print(f"  Paper Accuracy: 99.17%")
    print(f"  Your Accuracy:  {metrics['accuracy']*100:.2f}%")
    
    if metrics['accuracy'] > 0.97:
        print(f"  ✓ Excellent! Close to paper results")
    elif metrics['accuracy'] > 0.95:
        print(f"  ✓ Good performance")
    else:
        print(f"  ⚠️  Consider training longer or tuning hyperparameters")


def main():
    args = parse_args()
    
    print("="*70)
    print("CNN-LSTM MODEL EVALUATION")
    print("="*70)
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = CNNLSTM(input_shape=(3, 224, 224), num_classes=1)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("✓ Model loaded")
    
    # Load test data
    X_test, y_test = load_test_data(args.data_dir)
    
    # Evaluate
    metrics, cm, y_pred_proba, y_true = evaluate_model(model, X_test, y_test, device)
    
    # Print results
    print_results(metrics, cm)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_file = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"TN: {cm[0,0]}  FP: {cm[0,1]}\n")
        f.write(f"FN: {cm[1,0]}  TP: {cm[1,1]}\n")
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Save predictions for visualization
    np.save(os.path.join(args.output_dir, 'y_pred_proba.npy'), y_pred_proba)
    np.save(os.path.join(args.output_dir, 'y_true.npy'), y_true)
    
    print("="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
