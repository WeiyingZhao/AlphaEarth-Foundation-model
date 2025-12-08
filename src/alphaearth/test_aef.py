"""
Unit tests for AlphaEarth Foundations with simulated data.
Run with: python -m pytest src/alphaearth/test_aef.py -v
Or directly: python src/alphaearth/test_aef.py
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from alphaearth.loss_function import AEFLoss
from alphaearth.training import Trainer
from alphaearth.data import AEFDataset, AEFNPZDataset, create_aef_dataloader
from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.architecture.encoder_utils import SinusoidalTimeEncoding, SummaryPeriodEncoder


def test_source_configs():
    """Test that all sources from Table S2 are configured."""
    loss = AEFLoss()
    expected_sources = ['sentinel2', 'sentinel1', 'landsat8', 'landsat9', 
                       'palsar2', 'era5', 'gedi', 'grace', 'glo30', 'nlcd']
    for src in expected_sources:
        assert src in loss.source_configs, f"Missing source: {src}"
    print("✓ test_source_configs passed")


def test_loss_weights():
    """Test default loss weights match paper."""
    loss = AEFLoss()
    assert loss.reconstruction_weight == 1.0  # a = 1.0
    assert loss.uniformity_weight == 0.05     # b = 0.05
    assert loss.consistency_weight == 0.02    # c = 0.02
    assert loss.text_weight == 0.001          # d = 0.001
    print("✓ test_loss_weights passed")


def test_batch_uniformity_loss():
    """Test batch uniformity loss computation."""
    loss = AEFLoss()
    embeddings = torch.randn(4, 8, 8, 64)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    
    uni_loss = loss.batch_uniformity_loss(embeddings)
    assert uni_loss.shape == ()  # scalar
    assert uni_loss >= 0
    assert uni_loss <= 1
    print("✓ test_batch_uniformity_loss passed")


def test_consistency_loss():
    """Test teacher-student consistency loss."""
    loss = AEFLoss()
    teacher = torch.randn(4, 8, 8, 64)
    student = teacher + 0.1 * torch.randn_like(teacher)
    
    cons_loss = loss.consistency_loss(teacher, student)
    assert cons_loss.shape == ()
    assert cons_loss >= 0
    
    # Identical should give ~0 loss
    cons_loss_same = loss.consistency_loss(teacher, teacher)
    assert cons_loss_same < 0.01
    print("✓ test_consistency_loss passed")


def test_clip_loss():
    """Test CLIP-style contrastive loss."""
    loss = AEFLoss()
    img_emb = torch.randn(4, 64)
    txt_emb = torch.randn(4, 64)
    
    clip_loss = loss.clip_loss(img_emb, txt_emb)
    assert clip_loss.shape == ()
    assert clip_loss >= 0
    print("✓ test_clip_loss passed")


def test_lr_schedule():
    """Test learning rate schedule (warmup and decay)."""
    model = AlphaEarthFoundations(enable_text_align=False)
    dataloader = create_aef_dataloader(num_samples=10, batch_size=2, num_frames=4)
    trainer = Trainer(model, dataloader, lr=1e-4)
    
    # Warmup phase
    assert trainer._get_lr(0) == 0.0
    assert abs(trainer._get_lr(500) - 0.5e-4) < 1e-8
    assert abs(trainer._get_lr(1000) - 1e-4) < 1e-8
    
    # Decay phase
    assert trainer._get_lr(100000) == 0.0
    print("✓ test_lr_schedule passed")


def test_synthetic_dataset():
    """Test synthetic dataset generation."""
    dataset = AEFDataset(num_samples=10, patch_size=64, num_frames=8, return_text=True)
    
    assert len(dataset) == 10
    
    sample = dataset[0]
    assert 'source_data' in sample
    assert 'timestamps' in sample
    assert 'valid_period' in sample
    assert 'text' in sample
    
    s2 = sample['source_data']['sentinel2']
    assert s2.shape == (8, 64, 64, 5)
    print("✓ test_synthetic_dataset passed")


def test_dataloader_collation():
    """Test dataloader properly collates batches."""
    dataloader = create_aef_dataloader(
        num_samples=10, batch_size=4, num_frames=8, 
        num_workers=0, return_text=True
    )
    
    batch = next(iter(dataloader))
    assert batch['source_data']['sentinel2'].shape[0] == 4
    assert len(batch['valid_periods']) == 4
    assert len(batch['texts']) == 4
    print("✓ test_dataloader_collation passed")


def test_npz_dataset():
    """Test NPZ dataset with simulated data."""
    tmpdir = tempfile.mkdtemp()
    try:
        for i in range(3):
            data = {
                'sentinel2': np.random.rand(4, 32, 32, 5).astype(np.float32),
                'ts_sentinel2': np.array([1577836800000 + j * 86400000 for j in range(4)], dtype=np.float32),
                'sentinel1': np.random.rand(4, 32, 32, 5).astype(np.float32),
                'ts_sentinel1': np.array([1577836800000 + j * 86400000 for j in range(4)], dtype=np.float32),
            }
            np.savez(Path(tmpdir) / f"chip_{i}.npz", **data)
        
        dataset = AEFNPZDataset(tmpdir, sources=['sentinel2', 'sentinel1'])
        assert len(dataset) == 3
        
        sample = dataset[0]
        assert 'sentinel2' in sample['source_data']
        assert 'sentinel1' in sample['source_data']
        print("✓ test_npz_dataset passed")
    finally:
        shutil.rmtree(tmpdir)


def test_sinusoidal_encoding():
    """Test sinusoidal time encoding."""
    enc = SinusoidalTimeEncoding(dim=64)
    timestamps = torch.tensor([[1577836800000, 1577923200000]])
    
    encoded = enc(timestamps)
    assert encoded.shape == (1, 2, 64)
    print("✓ test_sinusoidal_encoding passed")


def test_summary_period_encoder():
    """Test summary period encoder."""
    enc = SummaryPeriodEncoder(dim=64)
    valid_period = torch.tensor([[1577836800000.0, 1609459200000.0]])
    
    query = enc(valid_period)
    assert query.shape == (1, 64)
    print("✓ test_summary_period_encoder passed")


def test_model_forward():
    """Test model forward pass."""
    model = AlphaEarthFoundations(model_size="small", enable_text_align=False)
    
    source_data = {'sentinel2': torch.randn(2, 4, 32, 32, 5)}
    timestamps = {'sentinel2': torch.rand(2, 4) * 1e12}
    valid_periods = [(1577836800000, 1609459200000)] * 2
    
    outputs = model(source_data, timestamps, valid_periods)
    
    assert 'embeddings' in outputs
    assert 'reconstructions' in outputs
    assert 'teacher_embeddings' in outputs
    assert 'student_embeddings' in outputs
    assert 'image_embeddings' in outputs
    print("✓ test_model_forward passed")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    model = AlphaEarthFoundations(model_size="small", enable_text_align=False)
    loss_fn = AEFLoss()
    
    source_data = {'sentinel2': torch.randn(2, 4, 32, 32, 5, requires_grad=True)}
    timestamps = {'sentinel2': torch.rand(2, 4) * 1e12}
    valid_periods = [(1577836800000, 1609459200000)] * 2
    
    outputs = model(source_data, timestamps, valid_periods)
    
    loss_outputs = {
        'embeddings': outputs['embeddings'],
        'teacher_embeddings': outputs['teacher_embeddings'],
        'student_embeddings': outputs['student_embeddings'],
        'image_embeddings': outputs['image_embeddings'],
    }
    
    losses = loss_fn(loss_outputs)
    losses['total'].backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0, "No gradients computed"
    print(f"✓ test_gradient_flow passed (gradients in {grad_count} parameters)")


def test_training_step():
    """Test a single training step."""
    model = AlphaEarthFoundations(model_size="small", enable_text_align=False)
    dataloader = create_aef_dataloader(
        num_samples=4, batch_size=2, num_frames=4, 
        patch_size=32, num_workers=0
    )
    trainer = Trainer(model, dataloader, lr=1e-4)
    
    # Run 2 steps
    trainer.train(max_steps=2, log_every=1)
    print("✓ test_training_step passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running AlphaEarth Foundations Tests")
    print("=" * 60)
    
    tests = [
        test_source_configs,
        test_loss_weights,
        test_batch_uniformity_loss,
        test_consistency_loss,
        test_clip_loss,
        test_lr_schedule,
        test_synthetic_dataset,
        test_dataloader_collation,
        test_npz_dataset,
        test_sinusoidal_encoding,
        test_summary_period_encoder,
        test_model_forward,
        test_gradient_flow,
        test_training_step,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
