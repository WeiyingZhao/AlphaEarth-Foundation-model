# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaEarth Foundations (AEF) - Unofficial PyTorch implementation of the AlphaEarth Foundations model for generating universal Earth embeddings from multi-source satellite imagery. Based on the paper "AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data" (Brown et al., 2025).

Key features:
- Von Mises-Fisher embeddings: 64-byte embeddings on unit sphere (S^63) with κ=8000
- Multi-source support: Sentinel-2, Sentinel-1, Landsat, GEDI, ERA5, GRACE, GLO-30, PALSAR-2, NLCD
- Space-Time-Precision (STP) encoder with 15 blocks for simultaneous spatial/temporal/high-frequency modeling
- CLIP-based text adapter for text-image alignment

## Commands

```bash
# Setup
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Training (synthetic data)
python src/alphaearth/run_train.py

# Run tests
PYTHONPATH=$PYTHONPATH:$(pwd)/src python src/alphaearth/test_aef.py
# Or with pytest
python -m pytest src/alphaearth/test_aef.py -v

# Download GEE data
python -m alphaearth.data_prep.download_gee_data \
    --project_id YOUR_GCP_PROJECT_ID \
    --output_dir ./data/gee_chips \
    --samples 100 \
    --workers 4
```

## Architecture

```
src/alphaearth/
├── architecture/
│   ├── aef_module.py        # Main AlphaEarthFoundations model class
│   ├── encoder.py           # STPEncoder - Space-Time-Precision encoder
│   ├── STPBlock.py          # Individual STP block implementation
│   ├── decoder.py           # VonMisesFisherDecoder for reconstruction
│   ├── encoder_utils.py     # IndividualSourceEncoder, SinusoidalTimeEncoding, SummaryPeriodEncoder
│   ├── text_adapter.py      # CLIP-based TextAdapter for text-image alignment
│   ├── stp_operators.py     # STP operators (attention, FFN)
│   └── laplacian_pyramid_exchange.py  # LearnedSpatialResampling between pathways
├── data.py                  # AEFDataset (synthetic), AEFNPZDataset (pre-extracted chips)
├── training.py              # Trainer class with LR schedule (warmup + linear decay)
├── loss_function.py         # AEFLoss: reconstruction + uniformity + consistency + text
└── run_train.py             # Training entry point
```

## Core Concepts

**Three-Pathway STP Encoder**: Space (1/16L, d_s=1024), Time (1/8L, d_t=512), Precision (1/2L, d_p=128) pathways process features at different resolutions, exchanging information via learned spatial resampling.

**Model Sizes**:
- `"small"`: d_p=64, d_t=256, d_s=512, 6 blocks (dev/testing)
- `"large"`: d_p=128, d_t=512, d_s=1024, 15 blocks (paper spec)

**Loss Components** (Equation 3):
- Reconstruction: L1 for continuous sources, CrossEntropy for categorical (NLCD)
- Batch Uniformity (b=0.05): Embeddings uniformly distributed on S^63
- Consistency (c=0.02): Teacher-student with input perturbations
- Text Alignment (d=0.001): CLIP-style contrastive loss

**Data Format**:
- Input tensors: `(B, T, H, W, C)` with timestamps `(B, T)` in milliseconds
- Valid period: `(start_ms, end_ms)` tuple defining summary time range
- NPZ chips from `AEFNPZDataset` contain per-source arrays with `ts_{source}` timestamps

## Key Patterns

**Forward Pass** (`aef_module.py:240`):
1. Per-source encoding → concatenate along channels
2. STP encoder produces features at precision resolution
3. TemporalSummarizer pools across time → 64D unit vectors
4. VonMisesFisherDecoder reconstructs observations

**Teacher-Student** (`aef_module.py:187`): Student receives perturbed inputs (random frame drops, half-year drops, source drops) to learn robust embeddings.

**Training** (`training.py:88`): Piecewise linear LR schedule: 0→1e-4 over [0, 1k), then 1e-4→0 over [1k, 100k].
