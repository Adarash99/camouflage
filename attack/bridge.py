#!/usr/bin/env python3
"""
TensorFlow ↔ PyTorch Bridge

Utilities for converting tensors between TensorFlow and PyTorch,
enabling hybrid pipelines (TF renderer + PyTorch detector).

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

import tensorflow as tf
import torch
import numpy as np


def tf_to_torch(tf_tensor, device='cpu'):
    """
    Convert TensorFlow tensor to PyTorch tensor.

    Args:
        tf_tensor: TensorFlow tensor [batch, H, W, C] (NHWC)
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        torch_tensor: PyTorch tensor [batch, C, H, W] (NCHW)
    """
    # Convert to NumPy
    if isinstance(tf_tensor, tf.Tensor):
        np_array = tf_tensor.numpy()
    else:
        np_array = np.array(tf_tensor)

    # Convert to PyTorch
    torch_tensor = torch.from_numpy(np_array).to(device)

    # NHWC → NCHW (if 4D image tensor)
    if torch_tensor.ndim == 4:
        torch_tensor = torch_tensor.permute(0, 3, 1, 2)

    return torch_tensor


def torch_to_tf(torch_tensor):
    """
    Convert PyTorch tensor to TensorFlow tensor.

    Args:
        torch_tensor: PyTorch tensor [batch, C, H, W] (NCHW)

    Returns:
        tf_tensor: TensorFlow tensor [batch, H, W, C] (NHWC)
    """
    # Move to CPU and convert to NumPy
    np_array = torch_tensor.detach().cpu().numpy()

    # NCHW → NHWC (if 4D image tensor)
    if np_array.ndim == 4:
        np_array = np_array.transpose(0, 2, 3, 1)

    # Convert to TensorFlow
    tf_tensor = tf.convert_to_tensor(np_array)

    return tf_tensor


def torch_to_numpy(torch_tensor):
    """
    Convert PyTorch tensor to NumPy array.

    Args:
        torch_tensor: PyTorch tensor

    Returns:
        np_array: NumPy array
    """
    return torch_tensor.detach().cpu().numpy()


def numpy_to_torch(np_array, device='cpu'):
    """
    Convert NumPy array to PyTorch tensor.

    Args:
        np_array: NumPy array
        device: PyTorch device

    Returns:
        torch_tensor: PyTorch tensor
    """
    return torch.from_numpy(np_array).to(device)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("TF ↔ PYTORCH BRIDGE TEST")
    print("=" * 70)
    print()

    # Test TF → PyTorch
    print("Testing TF → PyTorch conversion...")
    tf_tensor = tf.random.uniform([2, 500, 500, 3])
    print(f"  TF tensor shape: {tf_tensor.shape} (NHWC)")

    torch_tensor = tf_to_torch(tf_tensor)
    print(f"  PyTorch tensor shape: {torch_tensor.shape} (NCHW)")
    print(f"  Device: {torch_tensor.device}")
    print()

    # Test PyTorch → TF
    print("Testing PyTorch → TF conversion...")
    torch_tensor = torch.rand(2, 3, 500, 500)
    print(f"  PyTorch tensor shape: {torch_tensor.shape} (NCHW)")

    tf_tensor = torch_to_tf(torch_tensor)
    print(f"  TF tensor shape: {tf_tensor.shape} (NHWC)")
    print()

    # Test round-trip
    print("Testing round-trip (TF → PyTorch → TF)...")
    original = tf.random.uniform([1, 100, 100, 3])
    converted = torch_to_tf(tf_to_torch(original))

    diff = tf.reduce_max(tf.abs(original - converted)).numpy()
    print(f"  Original shape: {original.shape}")
    print(f"  Converted shape: {converted.shape}")
    print(f"  Max difference: {diff:.8f}")
    print(f"  ✓ Round-trip successful" if diff < 1e-6 else "  ✗ Round-trip failed")
    print()

    print("=" * 70)
    print("BRIDGE TEST COMPLETE ✓")
    print("=" * 70)
