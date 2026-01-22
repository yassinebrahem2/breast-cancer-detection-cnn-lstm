# Breast Cancer Classification with CNN-LSTM Hybrid Model

This project is an attempt to replicate the model architecture and results from the research paper **"Breast cancer classification based on hybrid CNN with LSTM model"**, targeting the reported 99.17% accuracy on the IDC (Invasive Ductal Carcinoma) breast histopathology image classification task.

## Overview

This implementation uses a hybrid CNN-LSTM architecture to classify breast histopathology images as either negative (no cancer) or positive (IDC detected). The model combines:
- **CNN component**: Extracts spatial features (textures, patterns) from histopathology images
- **LSTM component**: Captures sequential dependencies and temporal interactions
- **Classification head**: Binary classification output

## Dataset

- **Source**: [IDC Breast Histopathology Images](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) (Kaggle)
- **Size**: ~280,000+ histopathology image patches
- **Classes**: 
  - Class 0: Negative (no IDC)
  - Class 1: Positive (IDC detected)
- **Image size**: 244×244 pixels (as per paper specification)
- **Class imbalance**: Severe imbalance (addressed via SMOTE augmentation)

## Project Structure

```
breast-cancer-research/
├── main.py                      # Main menu interface
├── configs/
│   └── train.json              # Training configuration
├── scripts/
│   ├── download_data.py        # Download dataset from Kaggle
│   ├── preprocess_data.py      # Preprocessing with SMOTE
│   ├── dataset.py              # PyTorch Dataset & DataLoaders
│   ├── cnn_lstm.py            # CNN-LSTM model architecture
│   ├── train.py               # Training script
│   ├── evaluate.py            # Model evaluation
│   └── visualize_results.py   # Visualization utilities
├── data/
│   ├── raw/                   # Raw dataset (downloaded)
│   └── processed/             # Preprocessed data & metadata
├── outputs/                   # Training outputs (models, logs, results)
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended) or CPU
- Kaggle API credentials (for dataset download)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yassinebrahem2/breast-cancer-detection-cnn-lstm.git
   cd breast-cancer-detection-cnn-lstm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle API** (for dataset download):
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
   - Create API token and save to `~/.kaggle/kaggle.json`
   - Run: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### Quick Start

The easiest way to use this project is through the main menu interface:

```bash
python main.py
```

This provides an interactive menu with options for:
1. Download dataset
2. Preprocess data (with/without SMOTE)
3. Train model
4. Resume training
5. Evaluate model
6. Visualize results
7. Run full pipeline
8. Single image inference

### Step-by-Step Usage

#### 1. Download Dataset

```bash
python scripts/download_data.py
```

Downloads the IDC dataset from Kaggle (~3.5GB) to `data/raw/`.

#### 2. Preprocess Data

**With SMOTE (recommended, slower):**
```bash
python scripts/preprocess_data.py --apply-smote
```

**Without SMOTE (faster, uses class weighting):**
```bash
python scripts/preprocess_data.py
```

This step:
- Creates stratified train/val/test splits (80/15/5)
- Generates synthetic minority samples using SMOTE (if enabled)
- Saves metadata to `data/processed/metadata.pkl`

#### 3. Train Model

```bash
python scripts/train.py --config configs/train.json
```

**Resume from checkpoint:**
```bash
python scripts/train.py --config configs/train.json --resume outputs/checkpoint_epoch_020.pth
```

**Custom output directory:**
```bash
python scripts/train.py --config configs/train.json --output-dir custom_outputs
```

#### 4. Evaluate Model

After training, results are automatically saved to `outputs/results.json`. You can also run:

```bash
python scripts/evaluate.py
```

#### 5. Visualize Results

```bash
python scripts/visualize_results.py
```

## Model Architecture

### CNN Component

Three convolutional blocks with the following structure:

- **Conv Block 1**: 3→32 filters, 3×3 kernel, BatchNorm, ReLU, Dropout(0.5), MaxPool
- **Conv Block 2**: 32→64 filters, 5×5 kernel, BatchNorm, ReLU, Dropout(0.5), MaxPool
- **Conv Block 3**: 64→128 filters, 3×3 kernel, BatchNorm, ReLU, Dropout(0.5), MaxPool

**Output shape**: (batch_size, 128, 30, 30)

### LSTM Component

- **Input**: CNN features reshaped to (batch_size, 900, 128) - treating spatial dimensions as sequence
- **LSTM**: 2 layers, 100 hidden units, dropout=0.2
- **Output**: (batch_size, 100)

### Classification Head

- **Dense layer**: 100 units, ReLU, Dropout(0.2)
- **Output layer**: 2 neurons (for CrossEntropyLoss) or 1 neuron (for BCEWithLogitsLoss)

## Training Configuration

Key training parameters (configurable in `configs/train.json`):

- **Optimizer**: Adam (lr=0.001)
- **Loss function**: CrossEntropyLoss (or BCEWithLogitsLoss for binary)
- **L2 weight decay**: 0.0001
- **Learning rate scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early stopping**: Patience=5 epochs, min_delta=0.001
- **Batch size**: 32
- **Data augmentation**: Random rotation (±15°), horizontal/vertical flips, zoom (0.8×-1.2×)

## Data Augmentation

Applied only to training set (per paper specification):
- Random rotation: ±15 degrees
- Random horizontal flip: 50% probability
- Random vertical flip: 50% probability
- Random zoom: 0.8× to 1.2× scale

## Evaluation Metrics

The model is evaluated using the same metrics as the paper:
- **Accuracy**: Overall classification accuracy
- **Sensitivity** (Recall): True positive rate
- **Specificity**: True negative rate
- **F-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

## Output Files

After training, the following files are generated in `outputs/`:

- `best_model.pth`: Best model weights (lowest validation loss)
- `checkpoint_epoch_*.pth`: Periodic checkpoints
- `results.json`: Test metrics and training history
- `training.log`: Detailed training log
- `train_config.json`: Configuration used for training
- `training_history.png`: Training curves visualization
- `predictions.json`: Model predictions on test set (if enabled)

## TensorBoard

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir runs --reload_interval 5
```

Then open http://localhost:6006 in your browser.

## Implementation Features

- **Memory-efficient preprocessing**: Disk-based SMOTE generation to handle large datasets
- **On-demand image loading**: Images loaded from disk during training (not pre-loaded into RAM)
- **Stratified data splitting**: Maintains class distribution across train/val/test splits
- **Checkpointing**: Resume training from any saved checkpoint
- **Comprehensive logging**: File logs, TensorBoard, and console output

## Requirements

See `requirements.txt` for full list. Key dependencies:

- PyTorch (torch, torchvision)
- NumPy, Pandas
- scikit-learn, imbalanced-learn (for SMOTE)
- OpenCV, Pillow (image processing)
- Matplotlib, Seaborn (visualization)
- TensorBoard (logging)
- Kaggle API (dataset download)

## Troubleshooting

### Common Issues

1. **Kaggle API not configured**: See installation step 3
2. **Out of memory**: Reduce batch size in `configs/train.json`
3. **CUDA out of memory**: Use smaller batch size or CPU training
4. **Dataset not found**: Run `download_data.py` first
5. **Preprocessing takes too long**: Use `--apply-smote` flag only if needed

## Citation

If you use this code, please cite the original paper:

```
"Breast cancer classification based on hybrid CNN with LSTM model"
[Add paper citation details]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Original paper authors for the CNN-LSTM architecture
- Kaggle for hosting the IDC dataset
- PyTorch community for excellent deep learning framework
