# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaEarth Foundations is a PyTorch implementation of Google DeepMind's geospatial foundation model for generating Earth embeddings (64D vectors on unit sphere S^63) from multi-temporal satellite imagery. This is a framework/reference implementation - not fully trained on the complete dataset.

## Build and Development Commands

```bash
# Install dependencies (uses uv package manager)
uv pip install -r requirements.txt
uv pip install -e .

# Run training (sanity check with synthetic data)
python -m alphaearth.run_train

# Testing and linting
pytest                  # Unit tests
black .                 # Code formatting
flake8 .               # Linting
mypy src/              # Type checking
```

## Architecture

### Core Processing Pipeline

```
Input: source_data (B,T,H,W,C) + timestamps (B,T) + valid_periods (B,2)
    ↓
IndividualSourceEncoder → per-source projection
    ↓
STPEncoder (15 blocks) → Space/Time/Precision multi-resolution processing
    ↓
TemporalSummarizer → 64D embeddings on S^63
    ↓
VonMisesFisherDecoder → source reconstructions
```

### Key Components

| Module | Location | Purpose |
|--------|----------|---------|
| `AlphaEarthFoundations` | `architecture/aef_module.py` | Main model class with teacher-student branches |
| `STPEncoder` | `architecture/encoder.py` | Multi-resolution encoder with 3 pathways |
| `STPBlock` | `architecture/STPBlock.py` | Single block with Space/Time/Precision operators |
| `AEFLoss` | `loss_function.py` | Combined loss: reconstruction + uniformity + consistency + text |
| `Trainer` | `training.py` | Training loop with Adam optimizer |
| `AEFDataset` | `data.py` | Synthetic data generator for testing |

### STP Encoder Architecture

Three simultaneous pathways with information exchange via Laplacian pyramid:
- **Space Operator** (`stp_operators.py`): ViT-style spatial attention at 1/16 resolution
- **Time Operator** (`stp_operators.py`): Temporal attention with sinusoidal encoding at 1/8 resolution
- **Precision Operator** (`stp_operators.py`): 3x3 convolutions at 1/2 resolution

### Loss Function Weights (Equation 3)

```python
total = 1.0 * reconstruction + 0.05 * uniformity + 0.02 * consistency + 0.001 * text
```

## Data Flow

- **Input**: Sentinel-2 imagery (5 bands: B2, B3, B4, B8, B11) at 10m resolution
- **Temporal**: Variable-length sequences, timestamps in milliseconds
- **Output**: 64D unit vectors (von Mises-Fisher distribution on S^63)

### STAC Data Ingestion

`src/utils/stac_ingest.py` provides tools for fetching real Sentinel-2 data:
- Queries AWS Earth Search API
- Downloads COG tiles, resamples to 128x128 patches
- Outputs .npz files for `AEFNPZDataset`

## Extending for Downstream Tasks

`src/extending-aef-for-dataset-generation/` contains patterns for using embeddings with simple classifiers (RF, GBT, LogReg) for tasks like vegetation type classification.
