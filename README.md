# Image Classification using PyTorch

A deep learning project for image classification using PyTorch and CNN architecture.

## Project Overview

This project implements an image classification system using Convolutional Neural Networks (CNN) with PyTorch. It includes functionality for data preprocessing, augmentation, model training, evaluation, and visualization.

## Features

- **Image preprocessing**: Load and transform images into tensor format
- **Data augmentation**: Enhance training data through random transformations
- **Model architecture**: CNN implementation for image classification
- **Training pipeline**: Complete training loop with evaluation
- **Visualization**: Training/validation metrics plotting
- **Configuration**: YAML-based configuration system

## Project Structure

```
image-classification-pytorch/
├── config.yaml          # Configuration settings
├── main.py              # Main application entry point
├── cnn.py               # CNN model architecture
├── training.py          # Training loop implementation
├── image_preprocessor.py # Image loading and preparation
├── utils.py             # Utility functions
├── data/                # Data directory
│   └── animals/         # Example dataset structure
├── models/              # Saved model directory (auto-created)
└── plots/               # Visualization outputs (auto-created)
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- NumPy
- pandas
- matplotlib
- OpenCV
- scikit-learn
- PyYAML

## Installation

```bash
# Clone the repository
git clone https://github.com/yunusahin1/image-classification-pytorch.git
cd image-classification-pytorch

# Install dependencies
pip install torch torchvision numpy pandas matplotlib opencv-python scikit-learn pyyaml
```

## Usage

### Data Preparation

Place your image data in the `data` directory with the following structure:

```
data/
└── animals/
    ├── cat/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   └── ...
    ├── dog/
    │   ├── dog1.jpg
    │   └── ...
    └── ...
```

The folder names will be used as class labels.

### Configuration

Edit `config.yaml` to adjust:

```yaml
preprocessing:
  augmentation: true    # Enable/disable data augmentation

training:
  epochs: 10            # Number of training epochs
  learning_rate: 0.001  # Learning rate for optimizer
  batch_size: 32        # Batch size for training

# Model saving options
save_model: true        # Whether to save the trained model
model_dir: "./models"   # Directory to save models

# Visualization options
plot_training: true     # Whether to plot training metrics
save_plots: true        # Save plots instead of showing interactively
plots_dir: "./plots"    # Directory to save plots
```

### Running the Project

```bash
python main.py
```

This will:
1. Load and preprocess images
2. Apply data augmentation (if enabled)
3. Train the CNN model
4. Evaluate on the test set
5. Save the model (if enabled)
6. Generate training/validation plots (if enabled)

## Model Architecture

The CNN architecture consists of:
- Convolutional layers for feature extraction
- Max pooling layers for spatial reduction
- Fully connected layers for classification
- ReLU activation functions
- Dropout for regularization

## Customization

- To modify the CNN architecture, edit `cnn.py`
- For augmentation techniques, update the `data_augmentation` function in `image_preprocessor.py`
- Advanced training strategies can be implemented in `training.py`

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.