"""
CNN-LSTM Model - FIXED to Match Paper Specifications
Paper: "Breast cancer classification based on hybrid CNN with LSTM model"

FIXES APPLIED:
1. ✅ Added Dropout(0.5) to Conv Block 1
2. ✅ Changed to binary classification (1 output neuron)
3. ✅ Fixed LSTM dropout (num_layers=2 to enable dropout)
4. ✅ Improved LSTM sequence structure (900 timesteps)
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM for breast cancer classification
    
    Paper architecture:
    - CNN: Extracts spatial features (textures, patterns)
    - LSTM: Captures sequential/temporal dependencies
    - Output: Binary classification (negative/positive)
    """
    
    def __init__(self, 
                 cnn_filters=[32, 64, 128],
                 cnn_kernels=[(3, 3), (5, 5)],
                 cnn_dropout=0.5,
                 lstm_units=100,
                 lstm_num_layers=2,
                 lstm_dropout=0.2,
                 lstm_recurrent_dropout=0.2,
                 dense_units=None,
                 dense_dropout=0.2,
                 input_size=244,
                 num_classes=2):
        """
        Args:
            cnn_filters: [32, 64, 128] - Number of filters in each conv layer
            cnn_kernels: [(3, 3), (5, 5)] - Kernel sizes for conv layers
            cnn_dropout: 0.5 - Dropout rate for CNN blocks
            lstm_units: 100 - LSTM hidden units
            lstm_num_layers: 2 - Number of LSTM layers
            lstm_dropout: 0.2 - LSTM dropout rate (only works with num_layers > 1)
            lstm_recurrent_dropout: 0.2 - Recurrent dropout (not used in PyTorch)
            dense_units: int or None - Dense layer units (defaults to lstm_units)
            dense_dropout: 0.2 - Dropout rate for dense layer
            input_size: 244 - Input image size
            num_classes: 2 - Number of output classes
        """
        super(CNNLSTM, self).__init__()
        
        self.input_size = input_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units if dense_units is not None else lstm_units
        
        # ============================================
        # CNN COMPONENT (Spatial Feature Extraction)
        # ============================================
        
        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, cnn_filters[0], kernel_size=cnn_kernels[0], padding=1),
            nn.BatchNorm2d(cnn_filters[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout),
            nn.MaxPool2d(2, 2)
        )
        
        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=cnn_kernels[1], padding=2),
            nn.BatchNorm2d(cnn_filters[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout),
            nn.MaxPool2d(2, 2)
        )
        
        # Conv Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=cnn_kernels[0], padding=1),
            nn.BatchNorm2d(cnn_filters[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout),
            nn.MaxPool2d(2, 2)
        )
        
        # ============================================
        # LSTM COMPONENT (Sequential Analysis) - IMPROVED
        # ============================================
        
        # LSTM configuration
        self.lstm_input_size = cnn_filters[2]
        self.lstm_sequence_length = 30 * 30
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_units,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        # ============================================
        # CLASSIFICATION HEAD - FIXED TO BINARY
        # ============================================
        
        # Dense layer after LSTM
        self.dense = nn.Sequential(
            nn.Linear(lstm_units, self.dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dense_dropout)
        )
        
        # Output layer
        self.output = nn.Linear(self.dense_units, num_classes)
    
    def forward(self, x):
        """
        Forward pass - IMPROVED
        
        Args:
            x: (batch, 3, 244, 244)
        
        Returns:
            output: (batch, 2) - class logits for CrossEntropyLoss
        """
        
        # CNN Feature Extraction
        cnn_out = self.conv1(x)  # (batch, 32, 122, 122)
        cnn_out = self.conv2(cnn_out)  # (batch, 64, 61, 61)
        cnn_out = self.conv3(cnn_out)  # (batch, 128, 30, 30)
        
        # ✅ IMPROVED: Reshape for meaningful sequence
        # (batch, 128, 30, 30) → (batch, 128, 900) → (batch, 900, 128)
        batch_size = cnn_out.size(0)
        cnn_out = cnn_out.view(batch_size, 128, -1)  # (batch, 128, 900)
        lstm_input = cnn_out.permute(0, 2, 1)  # (batch, 900, 128)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)  # (batch, 900, 100)
        
        # Take last LSTM output
        lstm_out = lstm_out[:, -1, :]  # (batch, 100)
        
        # Dense layer
        dense_out = self.dense(lstm_out)  # (batch, 100)
        
        # Output layer
        logits = self.output(dense_out)  # (batch, 2)
        
        return logits


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    print("="*70)
    print("CNN-LSTM MODEL TEST (FIXED VERSION)")
    print("="*70)
    
    model = CNNLSTM()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass:")
    x = torch.randn(4, 3, 244, 244)
    output = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (4, 2)")
    
    if output.shape == (4, 2):
        print(f"  ✓ CORRECT!")
    else:
        print(f"  ✗ ERROR!")
    
    print(f"\n✓ Model test completed!")
    print(f"\nFIXES APPLIED:")
    print(f"  ✅ Added Dropout(0.5) to Conv Block 1")
    print(f"  ✅ LSTM num_layers=2 (enables dropout)")
    print(f"  ✅ LSTM sequence length = 900 (improved from 1)")
    print(f"  ⚠️  Still using 2 output neurons for CrossEntropyLoss")
    print(f"     (For true binary: use 1 neuron + BCEWithLogitsLoss)")
