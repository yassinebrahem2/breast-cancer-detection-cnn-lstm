"""
Preprocess IDC Dataset - Disk-based SMOTE
Generates synthetic minority samples and saves to disk
Memory efficient: Never loads all images at once
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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


def generate_smote_synthetic_images(image_paths, labels, train_idx, 
                                     output_dir, img_size=244, batch_size=5000):
    """
    Generate synthetic minority samples using SMOTE and save to disk
    Process in batches to avoid loading all images at once
    
    Paper: "The study used Synthetic Minority Over-sampling Technique (SMOTE) 
           to artificially augment the minority class of malignant cases"
    """
    print(f"\n{'='*70}")
    print("GENERATING SYNTHETIC SAMPLES USING SMOTE")
    print('='*70)
    
    train_paths = np.array(image_paths)[train_idx]
    train_labels = np.array([labels[i] for i in train_idx])
    
    neg_count = (train_labels == 0).sum()
    pos_count = (train_labels == 1).sum()
    
    print(f"\nOriginal training distribution:")
    print(f"  Negative (class 0): {neg_count:,}")
    print(f"  Positive (class 1): {pos_count:,}")
    print(f"  Ratio: {neg_count/pos_count:.2f}:1")
    
    # Create output directory for synthetic images
    synthetic_dir = os.path.join(output_dir, 'synthetic')
    os.makedirs(synthetic_dir, exist_ok=True)
    
    # Process in batches to manage memory
    num_batches = (len(train_paths) + batch_size - 1) // batch_size
    print(f"\nProcessing {num_batches} batches of {batch_size:,} images...")
    
    all_synthetic_paths = []
    all_synthetic_labels = []
    synthetic_count = 0
    
    for batch_idx in tqdm(range(num_batches), desc="SMOTE batches"):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(train_paths))
        
        batch_paths = train_paths[start:end]
        batch_labels = train_labels[start:end]
        
        # Load batch images
        X_batch = []
        y_batch = []
        
        for path, label in zip(batch_paths, batch_labels):
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X_batch.append(img)
                y_batch.append(label)
            except Exception as e:
                continue
        
        if len(X_batch) == 0:
            continue
        
        X_batch = np.array(X_batch, dtype=np.uint8)
        y_batch = np.array(y_batch)
        
        # Apply SMOTE to this batch
        if len(np.unique(y_batch)) > 1:  # Both classes present
            minority_count = min((y_batch == 0).sum(), (y_batch == 1).sum())
            k_neighbors = min(5, minority_count - 1)
            
            if k_neighbors > 0:
                # Flatten for SMOTE
                X_flat = X_batch.reshape(X_batch.shape[0], -1)
                
                # Apply SMOTE
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_flat_bal, y_bal = smote.fit_resample(X_flat, y_batch)
                
                # Reshape back
                X_bal = X_flat_bal.reshape(-1, img_size, img_size, 3).astype(np.uint8)
                
                # Save synthetic samples (only new ones created by SMOTE)
                original_count = len(X_batch)
                synthetic_batch = X_bal[original_count:]
                synthetic_labels_batch = y_bal[original_count:]
                
                # Write synthetic images to disk
                for img, label in zip(synthetic_batch, synthetic_labels_batch):
                    filename = f"synthetic_{synthetic_count:06d}_{label}.png"
                    filepath = os.path.join(synthetic_dir, filename)
                    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    all_synthetic_paths.append(filepath)
                    all_synthetic_labels.append(label)
                    synthetic_count += 1
    
    print(f"\n✓ Generated {synthetic_count:,} synthetic images")
    
    return all_synthetic_paths, all_synthetic_labels


def create_balanced_dataset(image_paths, labels, train_idx, val_idx, test_idx,
                           output_dir, synthetic_paths=None, synthetic_labels=None):
    """Create balanced training dataset by combining original + synthetic images"""
    
    print(f"\n{'='*70}")
    print("CREATING BALANCED DATASET METADATA")
    print('='*70)
    
    # Original training images
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    # Add synthetic images to training set
    if synthetic_paths:
        train_paths.extend(synthetic_paths)
        train_labels.extend(synthetic_labels)
    
    # Validation and test (unchanged)
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"\nBalanced training set:")
    print(f"  Original: {len(train_idx):,}")
    print(f"  Synthetic: {len(synthetic_paths) if synthetic_paths else 0:,}")
    print(f"  Total: {len(train_paths):,}")
    
    # Count class distribution
    train_labels_array = np.array(train_labels)
    neg_count = (train_labels_array == 0).sum()
    pos_count = (train_labels_array == 1).sum()
    
    print(f"  - Negative: {neg_count:,}")
    print(f"  - Positive: {pos_count:,}")
    
    # Save metadata
    metadata = {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'val_paths': val_paths,
        'val_labels': val_labels,
        'test_paths': test_paths,
        'test_labels': test_labels,
        'img_size': 244,
        'synthetic_count': len(synthetic_paths) if synthetic_paths else 0,
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Saved metadata.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw/IDC_regular_ps50_idx5')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--apply-smote', action='store_true', 
                       help='Generate synthetic samples with SMOTE')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("IDC DATASET PREPROCESSING - DISK-BASED SMOTE")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Get image paths
    image_paths, labels = get_image_paths(args.data_dir)
    
    # Step 2: Create splits
    train_idx, val_idx, test_idx = create_splits(image_paths, labels)
    
    # Step 3: Generate synthetic images (optional)
    synthetic_paths = []
    synthetic_labels = []
    
    if args.apply_smote:
        synthetic_paths, synthetic_labels = generate_smote_synthetic_images(
            image_paths, labels, train_idx, args.output_dir
        )
    else:
        print(f"\n⚠️  SMOTE disabled - using original imbalanced data")
    
    # Step 4: Create balanced metadata
    create_balanced_dataset(
        image_paths, labels, train_idx, val_idx, test_idx,
        args.output_dir, synthetic_paths, synthetic_labels
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  - data/processed/metadata.pkl")
    if args.apply_smote:
        print(f"  - data/processed/synthetic/ ({len(synthetic_paths):,} images)")
    print(f"\nNext step: python scripts/train.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
