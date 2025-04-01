# GPT-Style Transformer Implementation

## Overview
This project implements a GPT2-style Transformer model from scratch using PyTorch. The model is designed for text generation and follows the core principles of the original Transformer architecture, including multi-head self-attention and positional embeddings.

## Architecture
The model follows a standard GPT-like Transformer architecture with the following components:

- **Token Embedding Layer**: Maps input tokens to dense vector representations.
- **Positional Embeddings**: Encodes position information since the Transformer lacks inherent sequence order.
- **Multi-Head Self-Attention**: Computes attention across different token positions for contextual understanding.
- **Feed-Forward Layers**: Applies non-linearity and transformations after attention mechanisms.
- **Layer Normalization and Residual Connections**: Helps in stabilizing training and preserving gradient flow.
- **Decoder-Only Architecture**: Uses causal masking to prevent future token leakage during training.

## Implementation Details
- The model parameters are defined using a configuration dictionary.
- A `CustomDataset` class prepares input text, tokenizes it, and generates training sequences.
- The training loop optimizes the model using cross-entropy loss.
- Dropout regularization is used to prevent overfitting.

## Installation
Ensure you have the required dependencies installed:
```bash
pip install torch tiktoken
```

## Usage
Run the script to train the model on your dataset:
```bash
python transformer.py
```

## Future Improvements
- Implement mixed-precision training for efficiency.
- Introduce attention optimizations like FlashAttention.
- Experiment with different weight initialization techniques.

