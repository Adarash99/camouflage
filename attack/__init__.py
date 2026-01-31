"""
Adversarial Attack Pipeline

This package implements the adversarial camouflage attack system for
autonomous vehicle object detection evasion.

Components:
- detector_pytorch: PyTorch EfficientDet with pre-NMS access (RECOMMENDED)
- detector: TensorFlow EfficientDet wrapper (deprecated - NMS gradient issue)
- loss_pytorch: PyTorch attack loss functions
- loss: TensorFlow attack loss functions
- bridge: TensorFlow ↔ PyTorch conversion utilities
- gradient_test_pytorch: End-to-end gradient flow verification (PyTorch)

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

from .detector_pytorch import EfficientDetPyTorch
from .loss_pytorch import attack_loss_pytorch, attack_loss_with_stats_pytorch
from .bridge import tf_to_torch, torch_to_tf, numpy_to_torch, torch_to_numpy

# Legacy TensorFlow implementations (NMS gradient issue)
from .detector import EfficientDetWrapper
from .loss import attack_loss

__all__ = [
    # PyTorch (Recommended)
    'EfficientDetPyTorch',
    'attack_loss_pytorch',
    'attack_loss_with_stats_pytorch',
    # Bridge utilities
    'tf_to_torch',
    'torch_to_tf',
    'numpy_to_torch',
    'torch_to_numpy',
    # Legacy TensorFlow (deprecated)
    'EfficientDetWrapper',
    'attack_loss',
]
