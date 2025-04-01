# GPT-Style Transformer Implementation

## Overview
This project implements a GPT2-style Transformer model from scratch using PyTorch. The model is designed for text generation and follows the core principles of the original Transformer architecture, including multi-head self-attention and positional embeddings.

## Architecture

```mermaid
graph TD;
    InputText -->|B| Tokenization;
    B -->|C| Token Embeddings;
    C -->|D| Positional Encoding;
    D -->|E| Multi-Head Attention;
    E -->|F| Layer Norm & Residual;
    F -->|G| Feed-Forward Layer;
    G -->|H| Layer Norm & Residual;
    H -->|I| Softmax (Output Probabilities);
```

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
pip install -r Requirements.txt
```

## Usage
Run the script to train the model on your dataset:
```bash
python transformer.py
```
## Output Visualization
Observe how the text generated starts getting more coherent towards later epochs

![LLM Output](https://github.com/hkumar00/TransformersFromScratch/blob/main/images/Output.png)

## Future Improvements
- Implement mixed-precision training for efficiency.
- Introduce attention optimizations like FlashAttention.
- Experiment with different weight initialization techniques.

