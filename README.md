# AlphaEarth Foundations

AlphaEarth Foundations (AEF) is an unofficial PyTorch implementation of the **AlphaEarth Foundations** model. It features a novel **Embedding Field** paradigm designed to generate universal, highly general Earth embeddings for accurate and efficient global environmental monitoring.

## Key Features

- **Space-Time-Precision (STP) Encoder**: Simultaneous modeling of spatial, temporal, and high-frequency details.
- **Continuous Time Support**: Temporal interpolation and extrapolation using sinusoidal timecodes.
- **Von Mises-Fisher Embeddings**: 64-byte embeddings distributed on the unit sphere ($S^{63}$) with fixed $\kappa=8000$.
- **Text-Image Alignment**: CLIP-based Text Adapter for aligning geospatial features with semantic text descriptions.
- **Multi-Source Support**: Handles **10 data sources** including Optical (Sentinel-2, Landsat), Radar (Sentinel-1, PALSAR-2), LiDAR (GEDI), and Environmental (ERA5, GRACE, GLO-30).
- **Comprehensive Loss Function**: Implementation of reconstruction, batch uniformity, consistency, and text alignment losses.

## Installation

```bash
# Clone the repository
git clone https://github.com/brayden-zhang/alphaearth-foundations.git
cd alphaearth-foundations

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH to include source directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

## Data Preparation

### Google Earth Engine Data Download

This repository includes a robust script to download and prepare training data from Google Earth Engine (GEE). It supports multiple sensors and handles cloud masking, reprojection, and saving to `.npz` format.

**Prerequisites:**
1. A Google Cloud Project.
2. Authenticated Earth Engine CLI:
   ```bash
   earthengine authenticate
   ```

**Usage:**

```bash
python -m alphaearth.data_prep.download_gee_data \
    --project_id YOUR_GCP_PROJECT_ID \
    --output_dir ./data/gee_chips \
    --samples 100 \
    --workers 4
```

This will download aligned chips from Sentinel-2, Sentinel-1, Landsat, GEDI, ERA5, etc., into the specified output directory.

## Training

The training pipeline uses a `Trainer` class that implements the paper's learning rate schedule (warmup + linear decay) and supports multi-modal data loading.

**Run Training (Synthetic Data Loop):**
```bash
python src/alphaearth/run_train.py
```

**Training Configuration:**
- **Model Size**: Supports `"small"` (dev/test) and `"large"` (paper spec) configurations.
- **Data Loading**: Can load from synthetic data generator or pre-downloaded `.npz` files via `AEFNPZDataset`.

## Testing

A comprehensive test suite is included to verify all components, including loss functions, data loaders, and architecture modules.

**Run Tests:**
```bash
PYTHONPATH=$PYTHONPATH:$(pwd)/src python src/alphaearth/test_aef.py
```

## Architecture Details

- **Encoder**: 15 STP blocks (Large config: $d_p=128, d_t=512, d_s=1024$).
- **Decoder**: Implicit decoders for each source with source-specific loss functions (L1 or Cross-Entropy).
- **Text Adapter**: Projects CLIP text embeddings to the 64-dim AEF embedding space.

## Reference

Based on the paper:
> **AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data**  
> *Brown et al., 2025*  
> [arXiv:2507.22291](https://arxiv.org/abs/2507.22291)

> [!NOTE]
> This is an unofficial implementation. While it faithfully reproduces the architecture and training objectives described in the paper, it is a simplified version compared to the original JAX codebase.
