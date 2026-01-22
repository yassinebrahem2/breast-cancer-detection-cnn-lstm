"""
Download IDC Breast Histopathology Dataset from Kaggle
Requires: Kaggle API credentials (~/.kaggle/kaggle.json)
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def setup_kaggle_api():
    """Check if Kaggle API is configured"""
    try:
        import kaggle
        print("✓ Kaggle API found")
        return True
    except OSError as e:
        print("✗ Kaggle API credentials not found!")
        print("\nSetup Instructions:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False


def download_dataset(dataset_name, output_dir):
    """Download dataset from Kaggle"""
    import kaggle
    
    print(f"\nDownloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    kaggle.api.dataset_download_files(
        dataset_name,
        path=output_dir,
        unzip=True,
        quiet=False
    )
    
    print(f"✓ Dataset downloaded to {output_dir}")


def verify_dataset(data_path):
    """Verify dataset structure and count files"""
    print("\nVerifying dataset structure...")
    
    if not os.path.exists(data_path):
        print(f"✗ Dataset not found at {data_path}")
        return False
    
    # Count patient folders
    patient_dirs = [d for d in os.listdir(data_path) 
                   if os.path.isdir(os.path.join(data_path, d))]
    
    print(f"✓ Found {len(patient_dirs)} patient folders")
    
    # Count total images
    total_images = 0
    total_negative = 0
    total_positive = 0
    
    for patient_dir in patient_dirs[:5]:  # Sample first 5 patients
        patient_path = os.path.join(data_path, patient_dir)
        
        for class_dir in os.listdir(patient_path):
            class_path = os.path.join(patient_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            images = [f for f in os.listdir(class_path) if f.endswith('.png')]
            
            if class_dir == '0':
                total_negative += len(images)
            elif class_dir == '1':
                total_positive += len(images)
            
            total_images += len(images)
    
    print(f"✓ Sample count (first 5 patients):")
    print(f"  - Negative (class 0): {total_negative:,} images")
    print(f"  - Positive (class 1): {total_positive:,} images")
    print(f"  - Total: {total_images:,} images")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download IDC Breast Histopathology Dataset from Kaggle'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='paultimothymooney/breast-histopathology-images',
        help='Kaggle dataset identifier'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded data'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify dataset after download'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("IDC BREAST HISTOPATHOLOGY DATASET DOWNLOADER")
    print("="*70)
    
    # Check Kaggle API setup
    if not setup_kaggle_api():
        sys.exit(1)
    
    # Download dataset
    try:
        download_dataset(args.dataset, args.output_dir)
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)
    
    # Verify dataset structure
    if args.verify:
        dataset_path = os.path.join(args.output_dir, 'IDC_regular_ps50_idx5')
        if not verify_dataset(dataset_path):
            sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python scripts/preprocess_data.py")
    print("  2. Run: python scripts/train.py")


if __name__ == '__main__':
    main()
