#!/usr/bin/env python3
"""
Texture Applicator for Adversarial Camouflage Pipeline

This module provides the TextureApplicator class for applying adversarial textures
to vehicle reference images using a trained neural renderer.

Usage:
    from texture_applicator import TextureApplicator

    applicator = TextureApplicator()
    rendered = applicator.apply(x_ref, texture)

Author: Adversarial Camouflage Research Project
Date: 2026-01-30
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow import keras
import numpy as np


class TextureApplicator:
    """
    Applies adversarial textures to vehicle reference images using a trained neural renderer.

    The renderer is loaded once during initialization and kept in memory for efficient
    repeated application during the attack optimization loop.

    Inputs:
        - Reference image (x_ref): Vehicle with neutral gray color
        - Texture (eta_exp): Adversarial pattern to apply
        Both in shape (500, 500, 3) or (batch, 500, 500, 3), float32, [0, 1]

    Output:
        - Rendered image: Vehicle with applied camouflage texture
        Same shape as inputs, float32, [0, 1]
    """

    def __init__(self, model_path='models/k3_100epch_wo_custom_loss_model.h5'):
        """
        Initialize the texture applicator by loading the neural renderer model.

        Args:
            model_path: Path to the trained renderer .h5 file
                       Default: 'models/k3_100epch_wo_custom_loss_model.h5'

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        print(f"Initializing TextureApplicator...")

        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                f"Current directory: {os.getcwd()}\n"
                f"Please ensure the model file exists."
            )

        # Load the neural renderer model
        try:
            print(f"  Loading model: {model_path}")
            self.model = keras.models.load_model(model_path, compile=False)
            print(f"  ✓ Model loaded successfully")
            print(f"  Input shapes: {[inp.shape for inp in self.model.inputs]}")
            print(f"  Output shape: {self.model.output.shape}")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def apply(self, x_ref, texture):
        """
        Apply texture to reference image using the neural renderer.

        This method automatically handles both single images and batches:
        - Single: (500, 500, 3) → (500, 500, 3)
        - Batch: (batch, 500, 500, 3) → (batch, 500, 500, 3)

        Args:
            x_ref: Reference image (vehicle with neutral color)
                   Shape: (500, 500, 3) or (batch, 500, 500, 3)
                   Type: float32 or float64, range [0, 1]

            texture: Texture pattern to apply (eta_exp in renderer terms)
                     Shape: (500, 500, 3) or (batch, 500, 500, 3)
                     Type: float32 or float64, range [0, 1]

        Returns:
            rendered: Vehicle image with applied texture
                      Shape: Same as inputs (preserves batch dimension)
                      Type: float32, range [0, 1]

        Raises:
            ValueError: If shapes don't match or are incorrect
        """
        # Convert to numpy arrays if tensors
        x_ref_np = x_ref.numpy() if isinstance(x_ref, tf.Tensor) else np.array(x_ref)
        texture_np = texture.numpy() if isinstance(texture, tf.Tensor) else np.array(texture)

        # Detect if inputs are single images (3D) or batched (4D)
        is_single_ref = (x_ref_np.ndim == 3)
        is_single_texture = (texture_np.ndim == 3)

        # Both must be same format (single or batch)
        if is_single_ref != is_single_texture:
            raise ValueError(
                f"Reference and texture must have matching dimensions.\n"
                f"Got x_ref: {x_ref_np.shape}, texture: {texture_np.shape}"
            )

        # Expand to batch dimension if single images
        if is_single_ref:
            x_ref_batch = np.expand_dims(x_ref_np, axis=0)
            texture_batch = np.expand_dims(texture_np, axis=0)
            was_single = True
        else:
            x_ref_batch = x_ref_np
            texture_batch = texture_np
            was_single = False

        # Validate shapes
        self._validate_inputs(x_ref_batch, texture_batch)

        # Convert to float32 if needed
        x_ref_batch = x_ref_batch.astype(np.float32)
        texture_batch = texture_batch.astype(np.float32)

        # Apply renderer
        rendered_batch = self.model([x_ref_batch, texture_batch])

        # Convert to numpy
        rendered_np = rendered_batch.numpy() if isinstance(rendered_batch, tf.Tensor) else rendered_batch

        # Squeeze back to single image if input was single
        if was_single:
            rendered_np = np.squeeze(rendered_np, axis=0)

        return rendered_np

    def _validate_inputs(self, x_ref, texture):
        """
        Validate input shapes and value ranges.

        Args:
            x_ref: Reference image (batch format)
            texture: Texture pattern (batch format)

        Raises:
            ValueError: If validation fails
        """
        # Check shapes match
        if x_ref.shape != texture.shape:
            raise ValueError(
                f"Reference and texture must have matching shapes.\n"
                f"Got x_ref: {x_ref.shape}, texture: {texture.shape}"
            )

        # Check batch dimension exists
        if x_ref.ndim != 4:
            raise ValueError(
                f"Expected 4D batch format (batch, H, W, C), got {x_ref.ndim}D: {x_ref.shape}"
            )

        # Check spatial dimensions
        batch, height, width, channels = x_ref.shape
        if height != 500 or width != 500:
            raise ValueError(
                f"Expected spatial dimensions (500, 500), got ({height}, {width})\n"
                f"The trained model requires exactly 500x500 images."
            )

        # Check channels
        if channels != 3:
            raise ValueError(
                f"Expected 3 color channels (RGB), got {channels}"
            )

        # Check value range (warn if outside [0, 1])
        x_ref_min, x_ref_max = x_ref.min(), x_ref.max()
        texture_min, texture_max = texture.min(), texture.max()

        if x_ref_min < 0 or x_ref_max > 1:
            print(f"WARNING: x_ref values outside [0, 1]: [{x_ref_min:.4f}, {x_ref_max:.4f}]")
            print("         Did you forget to normalize? Expected float in [0, 1] range.")

        if texture_min < 0 or texture_max > 1:
            print(f"WARNING: texture values outside [0, 1]: [{texture_min:.4f}, {texture_max:.4f}]")
            print("         Did you forget to normalize? Expected float in [0, 1] range.")

    def get_model_info(self):
        """
        Get information about the loaded model for debugging.

        Returns:
            dict: Model metadata including input/output shapes and names
        """
        return {
            'input_names': [inp.name for inp in self.model.inputs],
            'input_shapes': [inp.shape for inp in self.model.inputs],
            'output_name': self.model.output.name,
            'output_shape': self.model.output.shape,
            'total_params': self.model.count_params(),
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("TEXTURE APPLICATOR TEST")
    print("=" * 70)
    print()

    # Initialize applicator
    applicator = TextureApplicator()
    print()

    # Test with random single image
    print("Testing with single image (500, 500, 3)...")
    x_ref_single = np.random.rand(500, 500, 3).astype(np.float32)
    texture_single = np.random.rand(500, 500, 3).astype(np.float32)

    rendered_single = applicator.apply(x_ref_single, texture_single)
    print(f"  Input shape: {x_ref_single.shape}")
    print(f"  Output shape: {rendered_single.shape}")
    print(f"  Output range: [{rendered_single.min():.4f}, {rendered_single.max():.4f}]")
    print("  ✓ Single image test passed")
    print()

    # Test with batch
    print("Testing with batch (6, 500, 500, 3) for EOT...")
    x_ref_batch = np.random.rand(6, 500, 500, 3).astype(np.float32)
    texture_batch = np.random.rand(6, 500, 500, 3).astype(np.float32)

    rendered_batch = applicator.apply(x_ref_batch, texture_batch)
    print(f"  Input shape: {x_ref_batch.shape}")
    print(f"  Output shape: {rendered_batch.shape}")
    print(f"  Output range: [{rendered_batch.min():.4f}, {rendered_batch.max():.4f}]")
    print("  ✓ Batch test passed")
    print()

    # Show model info
    print("Model Information:")
    info = applicator.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
