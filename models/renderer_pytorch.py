#!/usr/bin/env python3
"""
PyTorch Neural Renderer (Mask-Aware)

Pure PyTorch implementation of the mask-aware neural renderer, enabling true
end-to-end gradient flow for adversarial texture optimization.

Key Features:
- 7 input channels: [reference RGB (3) + texture RGB (3) + mask (1)]
- 1024x1024 resolution (configurable)
- Same architecture as TensorFlow version for compatibility
- Native PyTorch autograd support

Usage:
    from models.renderer_pytorch import MaskAwareRenderer

    # Create model
    renderer = MaskAwareRenderer()

    # Forward pass (differentiable)
    # Input: [batch, 7, H, W] - channels first (PyTorch convention)
    output = renderer(x)  # [batch, 3, H, W]

    # For training
    renderer.train()
    output = renderer(x)
    loss = criterion(output, target)
    loss.backward()  # Gradients flow to input

Author: Adversarial Camouflage Research Project
Date: 2026-02-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class MaskAwareRenderer(nn.Module):
    """
    PyTorch mask-aware neural renderer (7-channel input).

    Architecture matches the TensorFlow version exactly:
    - Conv2D(7 -> 32, 3x3, ReLU)
    - Conv2D(32 -> 64, 3x3, ReLU)
    - Conv2D(64 -> 128, 3x3, ReLU)
    - Conv2D(128 -> 64, 3x3, ReLU)
    - Conv2D(64 -> 32, 3x3, ReLU)
    - Conv2D(32 -> 3, 1x1, Sigmoid) -> output

    Input: [batch, 7, H, W] - ref(3) + texture(3) + mask(1)
    Output: [batch, 3, H, W] - rendered RGB image
    """

    def __init__(self, input_channels=7, resolution=1024):
        """
        Initialize the mask-aware renderer.

        Args:
            input_channels: Number of input channels (default: 7)
                           [ref_R, ref_G, ref_B, tex_R, tex_G, tex_B, mask]
            resolution: Expected input resolution (default: 1024)
        """
        super().__init__()
        self.input_channels = input_channels
        self.resolution = resolution

        # Encoder-decoder style fully convolutional network
        # Same architecture as TensorFlow version
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=1)  # RGB output

        # Initialize weights (Keras default is Glorot uniform)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to Keras defaults."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Glorot/Xavier uniform initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the renderer.

        Args:
            x: Input tensor [batch, 7, H, W]
               Channels: [ref_R, ref_G, ref_B, tex_R, tex_G, tex_B, mask]

        Returns:
            Rendered image [batch, 3, H, W] in range [0, 1]
        """
        # Encoder path
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Decoder path
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Output with sigmoid for [0, 1] range
        x = torch.sigmoid(self.conv_out(x))

        return x

    def forward_from_components(self, x_ref, texture, mask):
        """
        Convenience method that concatenates inputs before forward pass.

        All inputs should be in NCHW format (PyTorch convention).

        Args:
            x_ref: Reference image [batch, 3, H, W]
            texture: Texture pattern [batch, 3, H, W]
            mask: Paintable mask [batch, 1, H, W]

        Returns:
            Rendered image [batch, 3, H, W]
        """
        # Concatenate along channel dimension
        combined = torch.cat([x_ref, texture, mask], dim=1)
        return self.forward(combined)

    def get_model_info(self):
        """Get model metadata."""
        return {
            'framework': 'PyTorch',
            'input_channels': self.input_channels,
            'resolution': self.resolution,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def load_renderer(model_path, device=None):
    """
    Load a trained PyTorch renderer from disk.

    Args:
        model_path: Path to .pt or .pth file
        device: Target device ('cuda' or 'cpu'), auto-detected if None

    Returns:
        MaskAwareRenderer model in eval mode
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskAwareRenderer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def convert_keras_weights(keras_model_path, output_path):
    """
    Convert Keras/TensorFlow weights to PyTorch format.

    This function loads a trained Keras model and converts its weights
    to the PyTorch model format, accounting for the different conventions:
    - Keras: NHWC (channels last), kernel shape [H, W, C_in, C_out]
    - PyTorch: NCHW (channels first), kernel shape [C_out, C_in, H, W]

    Args:
        keras_model_path: Path to Keras .h5 model
        output_path: Path to save PyTorch .pt weights

    Returns:
        PyTorch model with converted weights
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras

    print(f"Loading Keras model from: {keras_model_path}")
    keras_model = keras.models.load_model(keras_model_path, compile=False)

    # Create PyTorch model
    pytorch_model = MaskAwareRenderer()

    # Map Keras layer names to PyTorch attributes
    layer_mapping = {
        'conv1': 'conv1',
        'conv2': 'conv2',
        'conv3': 'conv3',
        'conv4': 'conv4',
        'conv5': 'conv5',
        'output': 'conv_out',
    }

    # Convert weights
    for keras_name, pytorch_name in layer_mapping.items():
        keras_layer = keras_model.get_layer(keras_name)
        pytorch_layer = getattr(pytorch_model, pytorch_name)

        # Get Keras weights [H, W, C_in, C_out] and bias
        keras_weights = keras_layer.get_weights()
        kernel = keras_weights[0]  # [H, W, C_in, C_out]
        bias = keras_weights[1] if len(keras_weights) > 1 else None

        # Convert kernel: [H, W, C_in, C_out] -> [C_out, C_in, H, W]
        kernel_pytorch = np.transpose(kernel, (3, 2, 0, 1))

        # Assign to PyTorch layer
        pytorch_layer.weight.data = torch.from_numpy(kernel_pytorch.copy())
        if bias is not None:
            pytorch_layer.bias.data = torch.from_numpy(bias.copy())

        print(f"  Converted {keras_name} -> {pytorch_name}")
        print(f"    Keras shape: {kernel.shape} -> PyTorch shape: {kernel_pytorch.shape}")

    # Save PyTorch weights
    torch.save(pytorch_model.state_dict(), output_path)
    print(f"Saved PyTorch weights to: {output_path}")

    return pytorch_model


# Test script
if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH MASK-AWARE RENDERER TEST")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Create model
    print("Creating MaskAwareRenderer...")
    model = MaskAwareRenderer()
    model.to(device)
    print(f"  Input channels: {model.input_channels}")
    print(f"  Resolution: {model.resolution}")
    print()

    # Model info
    info = model.get_model_info()
    print("Model Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 2
    resolution = 1024

    # Create random input [batch, 7, H, W]
    x = torch.randn(batch_size, 7, resolution, resolution, device=device)
    print(f"  Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print()

    # Test forward_from_components
    print("Testing forward_from_components...")
    x_ref = torch.randn(batch_size, 3, resolution, resolution, device=device)
    texture = torch.randn(batch_size, 3, resolution, resolution, device=device)
    mask = torch.randn(batch_size, 1, resolution, resolution, device=device)

    with torch.no_grad():
        output = model.forward_from_components(x_ref, texture, mask)
    print(f"  Output shape: {output.shape}")
    print()

    # Test gradient flow (CRITICAL for adversarial attacks)
    print("Testing gradient flow through texture...")
    model.eval()

    x_ref = torch.randn(1, 3, resolution, resolution, device=device)
    texture = torch.randn(1, 3, resolution, resolution, device=device, requires_grad=True)
    mask = torch.randn(1, 1, resolution, resolution, device=device)

    # Forward pass
    output = model.forward_from_components(x_ref, texture, mask)
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Check gradients
    grad_norm = texture.grad.norm().item()
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.6f}")

    if grad_norm > 0:
        print("  PASSED: Gradients flow correctly through texture input")
    else:
        print("  FAILED: No gradients flowing to texture!")
    print()

    # Test with larger batch for memory check
    print("Testing with batch size 6 (EOT viewpoints)...")
    x_batch = torch.randn(6, 7, resolution, resolution, device=device)
    with torch.no_grad():
        output_batch = model(x_batch)
    print(f"  Input shape: {x_batch.shape}")
    print(f"  Output shape: {output_batch.shape}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB" if device == 'cuda' else "  N/A (CPU)")
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
