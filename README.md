# Transformer from Scratch

A PyTorch implementation of the encoder–decoder Transformer from *Attention Is All You Need* (Vaswani et al., 2017), written from the ground up as a learning exercise.

## What's implemented

- Scaled dot-product attention
- Multi-head attention (used for both self-attention and cross-attention)
- Sinusoidal positional encoding
- Encoder layer: multi-head self-attention → feed-forward, each wrapped in residual + LayerNorm
- Decoder layer: masked multi-head self-attention → cross-attention over encoder output → feed-forward, each wrapped in residual + LayerNorm
- Causal mask for decoder self-attention
- Full encoder–decoder stack with input projection (for continuous inputs) and output projection
- End-to-end training loop on a synthetic copy task

## Sanity-check task

The script trains the model to copy sequences of random 80-dimensional vectors (standing in for mel-spectrogram frames). The decoder is fed zeros as input — no teacher forcing — which means the model is forced to learn the copy entirely through cross-attention from the encoder. This is a deliberately harder setup than shifted-target teacher forcing: it directly exercises whether the cross-attention pathway works.

The task is a working sanity check for the architecture, not a benchmark. Loss descends steadily over 20 epochs, which is the expected behaviour.

## Architecture

Following the paper's base configuration:

| Hyperparameter | Value |
|---|---|
| Model dimension (d_model) | 512 |
| Attention heads | 8 |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Feed-forward dimension | 2048 |
| Input dimension | 80 (mel-style) |

## Training

| Setting | Value |
|---|---|
| Optimizer | Adam, lr=1e-4 |
| Loss | MSE |
| Batch size | 32 |
| Dataset size | 640 synthetic sequences |
| Sequence length | 32 |
| Epochs | 20 |

## Usage

```bash
git clone https://github.com/juliakorovsky/transformer_from_scratch
cd transformer_from_scratch
python transformer.py
```

## Requirements

- `torch`

## Notes

The goal was faithfulness to the paper over framework cleverness.

## Reference

Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
