# Neural Network Implementation

[![Python Package](https://github.com/vishvak2000/final-nn/actions/workflows/python-package.yml/badge.svg)](https://github.com/vishvak2000/final-nn/actions/workflows/python-package.yml)

## Overview

This project implements a neural network class from scratch in Python. The implementation includes:

- A fully connected neural network with configurable architecture
- Forward and backward propagation
- Multiple activation functions
- Binary cross-entropy and mean squared error loss functions
- Mini-batch gradient descent

## Applications

The neural network is applied to two practical tasks:

1. **Autoencoder**: A 64x16x64 dimensionality reduction model trained on the MNIST digits dataset
2. **Transcription Factor Binding Classifier**: A model that predicts whether DNA sequences are binding sites for the yeast transcription factor Rap1

## Usage

```python
from nn.nn import NeuralNetwork

# Define architecture
nn_arch = [
    {'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
    {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}
]

# Initialize network
nn = NeuralNetwork(
    nn_arch=nn_arch, 
    lr=0.01,
    batch_size=32,
    epochs=100,
    loss_function='mean_squared_error'
)

# Train network
train_loss, val_loss = nn.fit(X_train, y_train, X_val, y_val)

# Make predictions
predictions = nn.predict(X_test)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/USERNAME/REPO-NAME.git
cd REPO-NAME

# Install the package
pip install -e .
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn