"""
Disk-Based Hybrid SMOTE + Undersampling
Memory-efficient implementation for large datasets
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
from tqdm import tqdm
import pickle
from collections import Counter


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
    
    # Train+Val / Test
    train_val_idx, test_idx = train_test_split(
        range(len(image_paths)),
        test_size=test_size,
        random_state=42,
        stratify=labels_array
    )
    
    # Train / Val
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


def apply_hybrid_resampling_disk(image_paths, labels, train_idx, output_dir,
                                  smote_strategy=0.7, undersample_strategy=0.85,
                                  img_size=244, batch_size=10000):
    """
    Disk-based hybrid SMOTE + undersampling
    Processes data in small batches to manage memory
    """
    print(f"\n{'='*70}")
    print("HYBRID SMOTE + UNDERSAMPLING (DISK-BASED)")
    print('='*70)
    
    train_paths = np.array(image_paths)[train_idx]
    train_labels = np.array([labels[i] for i in train_idx])
    
    neg_count = (train_labels == 0).sum()
    pos_count = (train_labels == 1).sum()
    
    print(f"\nOriginal training distribution:")
    print(f"  Negative (class 0): {neg_count:,} ({neg_count/len(train_labels)*100:.1f}%)")
    print(f"  Positive (class 1): {pos_count:,} ({pos_count/len(train_labels)*100:.1f}%)")
    
    # Calculate target counts
    target_minority = int(neg_count * smote_strategy)
    target_majority = int(target_minority / undersample_strategy)
    
    print(f"\nTarget distribution:")
    print(f"  After SMOTE (minority → {smote_strategy:.0%} of majority):")
    print(f"    Class 1: {pos_count:,} → {target_minority:,}")
    print(f"  After Undersampling (majority → {1/undersample_strategy:.2f}x minority):")
    print(f"    Class 0: {neg_count:,} → {target_majority:,}")
    print(f"  Final ratio: {target_majority/(target_majority+target_minority)*100:.1f}% / {target_minority/(target_majority+target_minority)*100:.1f}%")
    
    # Separate by class
    class0_paths = train_paths[train_labels == 0]
    class1_paths = train_paths[train_labels == 1]
    
    print(f"\n{'='*70}")
    print("STEP 1: SMOTE on Minority Class (Class 1)")
    print('='*70)
    
    # Calculate how many synthetic samples needed
    num_synthetic_needed = target_minority - pos_count
    print(f"Need to generate {num_synthetic_needed:,} synthetic class 1 images")
    
    synthetic_dir = os.path.join(output_dir, 'synthetic_hybrid')
    os.makedirs(synthetic_dir, exist_ok=True)
    
    synthetic_paths = []
    synthetic_count = 0
    
    # For SMOTE to work, we need BOTH classes in each batch
    # Strategy: Process in batches with mixed class samples
    # Each batch contains: some class 0 + some class 1, then SMOTE oversamples class 1
    
    # Determine batch composition: 70% class 0, 30% class 1 in each batch (matching original ratio)
    total_train = len(class0_paths) + len(class1_paths)
    num_batches = max(1, (total_train + batch_size - 1) // batch_size)
    
    # How many class 0 and class 1 per batch
    class0_per_batch = int(batch_size * 0.7)
    class1_per_batch = batch_size - class0_per_batch
    
    print(f"Processing in {num_batches} batches of ~{batch_size:,} images (mixed classes)...")
    print(f"  Each batch: ~{class0_per_batch} class 0 + ~{class1_per_batch} class 1")
    
    # Shuffle indices for random sampling
    np.random.seed(42)
    class0_indices = np.random.permutation(len(class0_paths))
    class1_indices = np.random.permutation(len(class1_paths))
    
    class0_idx = 0
    class1_idx = 0
    
    for batch_idx in tqdm(range(num_batches), desc="SMOTE batches"):
        # Get batch indices for both classes
        batch_class0_end = min(class0_idx + class0_per_batch, len(class0_paths))
        batch_class1_end = min(class1_idx + class1_per_batch, len(class1_paths))
        
        if class0_idx >= len(class0_paths) or class1_idx >= len(class1_paths):
            # Wrap around if needed
            class0_idx = class0_idx % len(class0_paths)
            class1_idx = class1_idx % len(class1_paths)
            batch_class0_end = min(class0_idx + class0_per_batch, len(class0_paths))
            batch_class1_end = min(class1_idx + class1_per_batch, len(class1_paths))
        
        batch_class0_indices = class0_indices[class0_idx:batch_class0_end]
        batch_class1_indices = class1_indices[class1_idx:batch_class1_end]
        
        # Load images from both classes
        X_batch = []
        y_batch = []
        
        # Load class 0 samples
        for idx in batch_class0_indices:
            try:
                img = cv2.imread(class0_paths[idx])
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X_batch.append(img)
                y_batch.append(0)
            except:
                continue
        
        # Load class 1 samples
        for idx in batch_class1_indices:
            try:
                img = cv2.imread(class1_paths[idx])
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X_batch.append(img)
                y_batch.append(1)
            except:
                continue
        
        class0_idx = batch_class0_end
        class1_idx = batch_class1_end
        
        if len(X_batch) < 2 or len(set(y_batch)) < 2:
            continue
        
        X_batch = np.array(X_batch, dtype=np.uint8)
        y_batch = np.array(y_batch, dtype=int)
        
        # Count class 1 in this batch
        num_class1_in_batch = np.sum(y_batch == 1)
        
        # Calculate how many synthetic class 1 samples to generate from this batch
        samples_to_generate = min(
            int(num_synthetic_needed * (num_class1_in_batch / pos_count)),
            num_synthetic_needed - synthetic_count
        )
        
        if samples_to_generate <= 0:
            break
        
        # Target: original class 1 + synthetic
        target_class1 = num_class1_in_batch + samples_to_generate
        
        # SMOTE k_neighbors must be < minority class count
        k_neighbors = min(5, num_class1_in_batch - 1)
        if k_neighbors < 1:
            continue
        
        try:
            # Flatten for SMOTE
            X_flat = X_batch.reshape(len(X_batch), -1)
            
            # Apply SMOTE to oversample class 1
            smote = SMOTE(
                sampling_strategy={1: target_class1},
                k_neighbors=k_neighbors,
                random_state=42 + batch_idx
            )
            X_flat_res, y_res = smote.fit_resample(X_flat, y_batch)
            
            # Extract only the NEW synthetic class 1 samples
            # Original samples come first, synthetic samples are at the end
            original_count = len(X_batch)
            X_synthetic = X_flat_res[original_count:]
            y_synthetic = y_res[original_count:]
            
            # Filter to keep only class 1 synthetics
            class1_mask = y_synthetic == 1
            X_synthetic_class1 = X_synthetic[class1_mask]
            X_synthetic_class1 = X_synthetic_class1.reshape(-1, img_size, img_size, 3).astype(np.uint8)
            
            # Save synthetic images to disk
            for img in X_synthetic_class1:
                filename = f"synthetic_class1_{synthetic_count:06d}.png"
                filepath = os.path.join(synthetic_dir, filename)
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                synthetic_paths.append(filepath)
                synthetic_count += 1
                
                if synthetic_count >= num_synthetic_needed:
                    break
        except Exception as e:
            print(f"Batch {batch_idx} failed: {e}")
            continue
        
        if synthetic_count >= num_synthetic_needed:
            break
    
    print(f"✓ Generated {synthetic_count:,} synthetic class 1 images")
    
    # Combine original class 1 + synthetic
    all_class1_paths = list(class1_paths) + synthetic_paths
    all_class1_labels = [1] * len(all_class1_paths)
    
    print(f"\n{'='*70}")
    print("STEP 2: Random Undersampling of Majority Class (Class 0)")
    print('='*70)
    
    print(f"Undersampling class 0: {len(class0_paths):,} → {target_majority:,}")
    
    # Random undersample class 0
    np.random.seed(42)
    undersample_indices = np.random.choice(
        len(class0_paths),
        size=target_majority,
        replace=False
    )
    undersampled_class0_paths = class0_paths[undersample_indices]
    undersampled_class0_labels = [0] * len(undersampled_class0_paths)
    
    print(f"✓ Undersampled to {len(undersampled_class0_paths):,} class 0 images")
    
    # Combine both classes
    final_paths = list(undersampled_class0_paths) + all_class1_paths
    final_labels = undersampled_class0_labels + all_class1_labels
    
    print(f"\n{'='*70}")
    print("FINAL BALANCED TRAINING SET")
    print('='*70)
    print(f"Total images: {len(final_paths):,}")
    print(f"  Class 0: {final_labels.count(0):,} ({final_labels.count(0)/len(final_labels)*100:.1f}%)")
    print(f"  Class 1: {final_labels.count(1):,} ({final_labels.count(1)/len(final_labels)*100:.1f}%)")
    print(f"  Synthetic: {synthetic_count:,}")
    
    return final_paths, final_labels, synthetic_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw/IDC_regular_ps50_idx5')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--apply-hybrid', action='store_true')
    parser.add_argument('--smote-strategy', type=float, default=0.7)
    parser.add_argument('--undersample-strategy', type=float, default=0.85)
    parser.add_argument('--batch-size', type=int, default=2000)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("IDC DATASET PREPROCESSING - HYBRID DISK-BASED APPROACH")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image paths
    image_paths, labels = get_image_paths(args.data_dir)
    
    # Create splits
    train_idx, val_idx, test_idx = create_splits(image_paths, labels)
    
    # Apply hybrid resampling if requested
    if args.apply_hybrid:
        train_paths, train_labels, synthetic_count = apply_hybrid_resampling_disk(
            image_paths, labels, train_idx, args.output_dir,
            smote_strategy=args.smote_strategy,
            undersample_strategy=args.undersample_strategy,
            batch_size=args.batch_size
        )
    else:
        train_paths = [image_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        synthetic_count = 0
    
    # Validation and test (unchanged)
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    # Save metadata
    metadata = {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'val_paths': val_paths,
        'val_labels': val_labels,
        'test_paths': test_paths,
        'test_labels': test_labels,
        'img_size': 244,
        'synthetic_count': synthetic_count,
        'method': 'hybrid_smote_undersample' if args.apply_hybrid else 'none',
        'smote_strategy': args.smote_strategy if args.apply_hybrid else None,
        'undersample_strategy': args.undersample_strategy if args.apply_hybrid else None,
    }
    
    with open(os.path.join(args.output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Metadata saved to metadata.pkl")
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nNext step: python scripts/train.py --config configs/train.json")


if __name__ == '__main__':
    main()
