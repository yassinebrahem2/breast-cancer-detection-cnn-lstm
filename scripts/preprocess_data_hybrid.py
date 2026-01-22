"""
Preprocess IDC Dataset - HYBRID SMOTE + Undersampling (Paper's Actual Approach)
Paper: Uses combination of SMOTE oversampling + Random undersampling
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImblearnPipeline
from tqdm import tqdm
import pickle


def get_image_paths(data_path):
    """Collect all image paths and labels"""
    print("Scanning dataset for image paths...")
    
    image_paths = []
    labels = []
    
    patient_dirs = sorted([d for d in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, d))])
    
    for patient_dir in tqdm(patient_dirs, desc="Scanning"):
        patient_path = os.path.join(data_path, patient_dir)
        
        for class_dir in os.listdir(patient_path):
            class_path = os.path.join(patient_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            label = int(class_dir)
            
            for img_file in os.listdir(class_path):
                if img_file.endswith('.png'):
                    image_paths.append(os.path.join(class_path, img_file))
                    labels.append(label)
    
    print(f"✓ Found {len(image_paths):,} images")
    print(f"  - Negative (class 0): {labels.count(0):,}")
    print(f"  - Positive (class 1): {labels.count(1):,}")
    
    return image_paths, labels


def create_splits(image_paths, labels, test_size=0.2, val_size=0.15):
    """Create stratified train/val/test splits"""
    print(f"\nCreating stratified splits...")
    
    labels_array = np.array(labels)
    
    # Train+Val / Test (80/20)
    train_val_idx, test_idx = train_test_split(
        range(len(image_paths)),
        test_size=test_size,
        random_state=42,
        stratify=labels_array
    )
    
    # Train / Val (85/15)
    train_val_labels = labels_array[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=42,
        stratify=train_val_labels
    )
    
    print(f"\n✓ Splits created:")
    print(f"  Train: {len(train_idx):,}")
    print(f"    - Negative: {(labels_array[train_idx]==0).sum():,}")
    print(f"    - Positive: {(labels_array[train_idx]==1).sum():,}")
    print(f"  Val:   {len(val_idx):,}")
    print(f"  Test:  {len(test_idx):,}")
    
    return train_idx, val_idx, test_idx


def apply_hybrid_smote(image_paths, labels, train_idx, output_dir, img_size=244, 
                       smote_strategy=0.7, undersample_strategy=0.85):
    """
    Apply HYBRID SMOTE + Random Undersampling (Paper's approach)
    
    Strategy:
    1. SMOTE: Oversample minority to 70% of majority
    2. RandomUnderSampler: Reduce majority to 85% ratio
    
    Result: More balanced but not perfectly 50/50
    """
    print(f"\n{'='*70}")
    print("APPLYING HYBRID SMOTE + UNDERSAMPLING (PAPER'S APPROACH)")
    print('='*70)
    
    train_paths = np.array(image_paths)[train_idx]
    train_labels = np.array([labels[i] for i in train_idx])
    
    neg_count = (train_labels == 0).sum()
    pos_count = (train_labels == 1).sum()
    
    print(f"\nOriginal training distribution:")
    print(f"  Negative (class 0): {neg_count:,} ({neg_count/(neg_count+pos_count)*100:.1f}%)")
    print(f"  Positive (class 1): {pos_count:,} ({pos_count/(neg_count+pos_count)*100:.1f}%)")
    
    # Load ALL training images first
    print(f"\nLoading {len(train_paths):,} training images...")
    X_train = []
    y_train = []
    valid_paths = []
    
    for path, label in tqdm(zip(train_paths, train_labels), total=len(train_paths), desc="Loading"):
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            X_train.append(img)
            y_train.append(label)
            valid_paths.append(path)
        except Exception as e:
            continue
    
    X_train = np.array(X_train, dtype=np.uint8)
    y_train = np.array(y_train)
    
    print(f"✓ Loaded {len(X_train):,} images")
    
    # Flatten images for SMOTE
    print(f"\nFlattening images for SMOTE...")
    X_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Create hybrid pipeline
    print(f"\nApplying HYBRID resampling...")
    print(f"  Step 1: SMOTE (minority → {smote_strategy*100:.0f}% of majority)")
    print(f"  Step 2: Undersampling (final ratio: {undersample_strategy*100:.0f}%)")
    
    pipeline = ImblearnPipeline([
        ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
        ('undersample', RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42))
    ])
    
    X_flat_resampled, y_resampled = pipeline.fit_resample(X_flat, y_train)
    
    # Reshape back to images
    X_resampled = X_flat_resampled.reshape(-1, img_size, img_size, 3).astype(np.uint8)
    
    print(f"\n✓ Resampling complete:")
    print(f"  Final training size: {len(X_resampled):,}")
    print(f"  Class 0: {(y_resampled == 0).sum():,} ({(y_resampled == 0).mean()*100:.1f}%)")
    print(f"  Class 1: {(y_resampled == 1).sum():,} ({(y_resampled == 1).mean()*100:.1f}%)")
    
    # Save resampled images
    print(f"\nSaving resampled images...")
    synthetic_dir = os.path.join(output_dir, 'hybrid_resampled')
    os.makedirs(synthetic_dir, exist_ok=True)
    
    new_paths = []
    new_labels = []
    
    for idx, (img, label) in enumerate(tqdm(zip(X_resampled, y_resampled), total=len(X_resampled), desc="Saving")):
        filename = f"resampled_{idx:06d}_{label}.png"
        filepath = os.path.join(synthetic_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        new_paths.append(filepath)
        new_labels.append(int(label))
    
    return new_paths, new_labels


def create_metadata(image_paths, labels, train_idx, val_idx, test_idx,
                    output_dir, resampled_paths=None, resampled_labels=None):
    """Create metadata with resampled training data"""
    
    print(f"\n{'='*70}")
    print("CREATING METADATA")
    print('='*70)
    
    # Use resampled data for training if provided
    if resampled_paths:
        train_paths = resampled_paths
        train_labels = resampled_labels
    else:
        train_paths = [image_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
    
    # Validation and test (unchanged)
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"\nDataset summary:")
    print(f"  Train: {len(train_paths):,}")
    print(f"  Val:   {len(val_paths):,}")
    print(f"  Test:  {len(test_paths):,}")
    
    # Save metadata
    metadata = {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'val_paths': val_paths,
        'val_labels': val_labels,
        'test_paths': test_paths,
        'test_labels': test_labels,
        'img_size': 244,
        'resampling_method': 'hybrid_smote_undersample' if resampled_paths else 'none',
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Saved metadata.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw/IDC_regular_ps50_idx5')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--apply-hybrid', action='store_true',
                       help='Apply hybrid SMOTE + undersampling')
    parser.add_argument('--smote-strategy', type=float, default=0.7,
                       help='SMOTE sampling strategy (default: 0.7)')
    parser.add_argument('--undersample-strategy', type=float, default=0.85,
                       help='Undersampling strategy (default: 0.85)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("IDC DATASET PREPROCESSING - HYBRID APPROACH")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Get image paths
    image_paths, labels = get_image_paths(args.data_dir)
    
    # Step 2: Create splits
    train_idx, val_idx, test_idx = create_splits(image_paths, labels)
    
    # Step 3: Apply hybrid resampling (optional)
    resampled_paths = None
    resampled_labels = None
    
    if args.apply_hybrid:
        resampled_paths, resampled_labels = apply_hybrid_smote(
            image_paths, labels, train_idx, args.output_dir,
            smote_strategy=args.smote_strategy,
            undersample_strategy=args.undersample_strategy
        )
    else:
        print(f"\n⚠️  Hybrid resampling disabled - using original data")
    
    # Step 4: Create metadata
    create_metadata(
        image_paths, labels, train_idx, val_idx, test_idx,
        args.output_dir, resampled_paths, resampled_labels
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  - data/processed/metadata.pkl")
    if args.apply_hybrid:
        print(f"  - data/processed/hybrid_resampled/ ({len(resampled_paths):,} images)")
    print(f"\nNext step: python scripts/train.py --config configs/train.json")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
