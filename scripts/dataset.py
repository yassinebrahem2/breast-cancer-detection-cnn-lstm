"""
PyTorch Dataset that loads images on-the-fly from disk
Works with both original and synthetic images
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


class IDCDataset(Dataset):
    """Load images on-demand from disk"""
    
    def __init__(self, image_paths, labels, img_size=244, transform=None):
        """
        Args:
            image_paths: list of full image paths (original + synthetic)
            labels: list of labels (0 or 1)
            img_size: resize to (img_size, img_size)
            transform: torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.img_size = img_size
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Load single image on-demand"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image from disk
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to 244×244 (paper spec)
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Convert to tensor (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            # Apply transforms if provided
            if self.transform:
                img = self.transform(img)
            
            return img, label
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, self.img_size, self.img_size), label


def create_dataloaders(metadata_path, batch_size=32, num_workers=4, pin_memory=True):
    """Create train/val/test dataloaders from preprocessing metadata"""
    
    # Load metadata (saved by preprocess_data.py)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    train_paths = metadata['train_paths']
    train_labels = metadata['train_labels']
    val_paths = metadata['val_paths']
    val_labels = metadata['val_labels']
    test_paths = metadata['test_paths']
    test_labels = metadata['test_labels']
    
    print(f"\nLoading datasets from metadata:")
    print(f"  Train: {len(train_paths):,} images")
    print(f"    - Original: {len([p for p in train_paths if 'synthetic' not in p]):,}")
    print(f"    - Synthetic: {metadata.get('synthetic_count', 0):,}")
    print(f"  Val:   {len(val_paths):,} images")
    print(f"  Test:  {len(test_paths):,} images")
    
    # Training transforms (data augmentation per paper)
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),              # Paper: ±15°
        transforms.RandomHorizontalFlip(p=0.5),    # Paper: flip
        transforms.RandomVerticalFlip(p=0.5),      # Paper: flip
        transforms.RandomAffine(
            degrees=0, 
            scale=(0.8, 1.2)                        # Paper: zoom 20%
        ),
    ])
    
    # Create datasets
    train_ds = IDCDataset(
        train_paths, train_labels,
        img_size=244,
        transform=train_transform
    )
    
    val_ds = IDCDataset(
        val_paths, val_labels,
        img_size=244,
        transform=None  # No augmentation on val
    )
    
    test_ds = IDCDataset(
        test_paths, test_labels,
        img_size=244,
        transform=None  # No augmentation on test
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\n✓ DataLoaders created successfully")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataloaders
    import torch
    
    train_loader, val_loader, test_loader = create_dataloaders(
        'data/processed/metadata.pkl',
        batch_size=32,
        num_workers=4
    )
    
    # Load one batch to test
    print(f"\n{'='*70}")
    print("TESTING DATALOADERS")
    print('='*70)
    
    img_batch, label_batch = next(iter(train_loader))
    print(f"\nBatch loaded:")
    print(f"  Images shape: {img_batch.shape}")
    print(f"  Labels shape: {label_batch.shape}")
    print(f"  Memory per batch: {img_batch.element_size() * img_batch.nelement() / 1e6:.2f} MB")
    print(f"  Negative samples: {(label_batch == 0).sum().item()}")
    print(f"  Positive samples: {(label_batch == 1).sum().item()}")
    print(f"\n✓ DataLoaders working correctly!")
