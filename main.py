"""
Breast Cancer CNN-LSTM Classification - Main Menu
Complete pipeline for IDC breast histopathology image classification

Paper: Breast cancer classification based on hybrid CNN with LSTM model
Dataset: IDC Breast Histopathology Images (Kaggle)
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header():
    print("\n" + "="*70)
    print("BREAST CANCER CNN-LSTM CLASSIFICATION PIPELINE")
    print("="*70)
    print("Paper: Breast cancer classification based on hybrid CNN with LSTM")
    print("Dataset: IDC Breast Histopathology Images (Kaggle)")
    print("="*70 + "\n")


def print_menu():
    print("\nMAIN MENU:")
    print("-" * 70)
    print("  [1] Download Dataset (Kaggle)")
    print("  [2] Preprocess Data (Disk-based SMOTE)")
    print("  [3] Train Model (CNN-LSTM)")
    print("  [4] Resume Training (from checkpoint)")
    print("  [5] Evaluate Model")
    print("  [6] Visualize Results")
    print("  [7] Run Full Pipeline (1→2→3→4→5→6)")
    print("  [8] Inference on Single Image")
    print("  [0] Exit")
    print("-" * 70)


def check_prerequisites(step):
    """Check if previous steps are complete"""
    checks = {
        2: ("data/raw/IDC_regular_ps50_idx5", "Dataset not downloaded. Run step 1 first."),
        3: ("data/processed/metadata.pkl", "Data not preprocessed. Run step 2 first."),
        5: ("outputs/best_model.pth", "Model not trained. Run step 3 first."),
        6: ("outputs/best_model.pth", "Model not trained. Run step 3 first."),
        8: ("outputs/best_model.pth", "Model not trained. Run step 3 first."),
    }
    
    if step in checks:
        path, message = checks[step]
        if not os.path.exists(path):
            print(f"\n⚠️  {message}")
            return False
    return True


def step_1_download():
    """Download dataset from Kaggle"""
    print("\n" + "="*70)
    print("STEP 1: DOWNLOAD DATASET")
    print("="*70)
    print("\nThis will download ~3.5GB from Kaggle.")
    print("Make sure you have configured Kaggle API credentials.")
    print("(See: https://github.com/Kaggle/kaggle-api#api-credentials)")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    try:
        from scripts.download_data import main as download_main
        sys.argv = ['download_data.py']
        download_main()
        print("\n✓ Download completed successfully!")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        import traceback
        traceback.print_exc()


def step_2_preprocess():
    """Preprocess dataset with disk-based SMOTE"""
    if not check_prerequisites(2):
        return
    
    print("\n" + "="*70)
    print("STEP 2: PREPROCESS DATA (DISK-BASED SMOTE)")
    print("="*70)
    print("\nOptions:")
    print("  [1] With SMOTE (generates synthetic images, ~30-60 min)")
    print("  [2] Without SMOTE (faster, uses class weighting, ~5-10 min)")
    
    choice = input("\nSelect option (1/2): ").strip()
    
    if choice == '1':
        apply_smote = True
        print("\n⚠️  This will generate synthetic images and take 30-60 minutes...")
    elif choice == '2':
        apply_smote = False
        print("\n✓ Using class weighting instead of SMOTE (faster)")
    else:
        print("Invalid option.")
        return
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    try:
        from scripts.preprocess_data import main as preprocess_main
        
        sys.argv = ['preprocess_data.py']
        if apply_smote:
            sys.argv.append('--apply-smote')
        
        preprocess_main()
        print("\n✓ Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()


def step_3_train():
    """Train CNN-LSTM model"""
    if not check_prerequisites(3):
        return
    
    print("\n" + "="*70)
    print("STEP 3: TRAIN CNN-LSTM MODEL")
    print("="*70)
    print("\nTraining time estimates:")
    print("  - With GPU: ~2-3 hours per epoch")
    print("  - With CPU: ~10-15 hours per epoch")
    print("  - Total (with early stopping): ~30-50 hours")
    print("\n⚠️  Recommended: Run in tmux/screen for SSH sessions")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print("\nConfig file: configs/train.json")
    custom = input("Use custom config file? (y/n): ").strip().lower()
    
    config_file = 'configs/train.json'
    if custom == 'y':
        config_file = input("Enter config file path: ").strip()
    
    try:
        from scripts.train import main as train_main
        
        # NEW: Use config file instead of individual args
        sys.argv = [
            'train.py',
            '--config', config_file
        ]
        
        train_main()
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()



def step_4_resume_training():
    """Resume training from checkpoint"""
    print("\n" + "="*70)
    print("STEP 4: RESUME TRAINING")
    print("="*70)
    
    # Find available checkpoints
    checkpoint_dir = 'outputs'
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    
    if not checkpoints:
        print("\n⚠️  No checkpoints found in outputs/")
        print("Please train a model first (option 3).")
        return
    
    # Sort checkpoints by epoch number
    checkpoints.sort()
    
    print("\nAvailable checkpoints:")
    for i, ckpt in enumerate(checkpoints, 1):
        epoch = ckpt.replace('checkpoint_epoch_', '').replace('.pth', '')
        print(f"  [{i}] {ckpt} (Epoch {int(epoch)})")
    
    choice = input(f"\nSelect checkpoint (1-{len(checkpoints)}) or 'latest': ").strip().lower()
    
    if choice == 'latest':
        checkpoint_file = checkpoints[-1]
    elif choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
        checkpoint_file = checkpoints[int(choice) - 1]
    else:
        print("Invalid choice.")
        return
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    epoch_num = checkpoint_file.replace('checkpoint_epoch_', '').replace('.pth', '')
    
    print(f"\nResuming from: {checkpoint_file} (Epoch {int(epoch_num)})")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print("\nConfig file: configs/train.json")
    custom = input("Use custom config file? (y/n): ").strip().lower()
    
    config_file = 'configs/train.json'
    if custom == 'y':
        config_file = input("Enter config file path: ").strip()
    
    try:
        from scripts.train import main as train_main
        
        sys.argv = [
            'train.py',
            '--config', config_file,
            '--resume', checkpoint_path
        ]
        
        train_main()
        print("\n✓ Training resumed and completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


def step_5_evaluate():
    """Evaluate trained model"""
    if not check_prerequisites(5):
        return
    
    print("\n" + "="*70)
    print("STEP 4: EVALUATE MODEL")
    print("="*70)
    
    try:
        # Load and display results from training
        import json
        
        results_file = 'outputs/results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print("\n📊 Test Results (Paper Metrics):")
            print("-" * 70)
            
            metrics = results.get('test_metrics', {})
            print(f"  Accuracy:    {metrics.get('accuracy', 0):.2f}%")
            print(f"  Sensitivity: {metrics.get('sensitivity', 0):.2f}%")
            print(f"  Specificity: {metrics.get('specificity', 0):.2f}%")
            print(f"  F-score:     {metrics.get('f_score', 0):.2f}%")
            print(f"  AUC:         {metrics.get('auc', 0):.4f}")
            
            print("\n✓ Evaluation completed!")
        else:
            print(f"⚠️  Results file not found: {results_file}")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def step_6_visualize():
    """Visualize results"""
    if not check_prerequisites(5):
        return
    
    print("\n" + "="*70)
    print("STEP 5: VISUALIZE RESULTS")
    print("="*70)
    
    try:
        import matplotlib.pyplot as plt
        import json
        
        results_file = 'outputs/results.json'
        if not os.path.exists(results_file):
            print(f"⚠️  Results file not found: {results_file}")
            return
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        history = results.get('history', {})
        
        if not history:
            print("⚠️  No training history found")
            return
        
        # Plot training history
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
        plt.savefig('outputs/training_visualization.png', dpi=150, bbox_inches='tight')
        
        print("\n✓ Visualization saved to: outputs/training_visualization.png")
        plt.show()
        
    except Exception as e:
        print(f"\n✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


def step_7_full_pipeline():
    """Run complete pipeline"""
    print("\n" + "="*70)
    print("FULL PIPELINE: ALL STEPS")
    print("="*70)
    print("\nThis will run:")
    print("  1. Download dataset (~10 min)")
    print("  2. Preprocess with SMOTE (~30-60 min)")
    print("  3. Train model (~30-50 hours with GPU)")
    print("  4. Evaluate model (~2 min)")
    print("  5. Visualize results (~2 min)")
    print("\n⚠️  Total time: ~35-60 hours")
    
    confirm = input("\nProceed with full pipeline? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    steps = [
        ("Download Dataset", step_1_download),
        ("Preprocess Data", step_2_preprocess),
        ("Train Model", step_3_train),
        ("Evaluate Model", step_4_evaluate),
        ("Visualize Results", step_5_visualize)
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"PIPELINE STEP {i}/{len(steps)}: {name.upper()}")
        print('='*70)
        
        try:
            func()
        except Exception as e:
            print(f"\n✗ Step {i} failed: {e}")
            print("Pipeline aborted.")
            return
        
        if i < len(steps):
            input("\nPress Enter to continue to next step...")
    
    print(f"\n{'='*70}")
    print("✓ PIPELINE COMPLETE!")
    print('='*70)


def step_8_inference():
    """Run inference on single image"""
    if not check_prerequisites(7):
        return
    
    print("\n" + "="*70)
    print("STEP 7: SINGLE IMAGE INFERENCE")
    print("="*70)
    
    img_path = input("\nEnter path to image: ").strip()
    
    if not os.path.exists(img_path):
        print(f"✗ Image not found: {img_path}")
        return
    
    try:
        import torch
        import cv2
        import numpy as np
        from src.models.cnn_lstm import CNNLSTM
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CNNLSTM()
        model.load_state_dict(torch.load('outputs/best_model.pth', map_location=device))
        model = model.to(device)
        model.eval()
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (244, 244))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item() * 100
        
        class_names = ['Negative (No Cancer)', 'Positive (Cancer Detected)']
        
        print(f"\n{'='*70}")
        print("PREDICTION RESULT")
        print('='*70)
        print(f"\nImage: {img_path}")
        print(f"Prediction: {class_names[pred_class]}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"\nProbabilities:")
        print(f"  Negative: {probs[0, 0].item()*100:.2f}%")
        print(f"  Positive: {probs[0, 1].item()*100:.2f}%")
        print('='*70)
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main menu loop"""
    print_header()
    
    while True:
        print_menu()
        choice = input("\nSelect option (0-8): ").strip()
        
        if choice == '0':
            print("\n👋 Exiting. Good luck with your research!")
            sys.exit(0)
        elif choice == '1':
            step_1_download()
        elif choice == '2':
            step_2_preprocess()
        elif choice == '3':
            step_3_train()
        elif choice == '4':
            step_4_resume_training()
        elif choice == '5':
            step_5_evaluate()
        elif choice == '6':
            step_6_visualize()
        elif choice == '7':
            step_7_full_pipeline()
        elif choice == '8':
            step_8_inference()
        else:
            print("\n✗ Invalid option. Please select 0-8.")
        
        input("\nPress Enter to return to main menu...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Exiting...")
        sys.exit(0)

