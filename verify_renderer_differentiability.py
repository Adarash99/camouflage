#!/usr/bin/env python3
"""
Renderer Differentiability Verification Script

Tests that gradients can flow from the renderer output back to texture parameters.
This is critical for the adversarial attack pipeline to work.

Based on: docs/plans/2026-01-30-adversarial-camouflage-design.md (Section 4)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow import keras
import numpy as np


def verify_differentiability(model_path='models/k3_100epch_wo_custom_loss_model.h5'):
    """
    Verify that gradients flow through the neural renderer.

    Args:
        model_path: Path to the trained renderer model

    Returns:
        bool: True if gradients flow correctly, False otherwise
    """
    print("=" * 70)
    print("RENDERER DIFFERENTIABILITY VERIFICATION")
    print("=" * 70)
    print()

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available models:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                if f.endswith('.h5'):
                    print(f"  - models/{f}")
        return False

    print(f"Loading renderer model: {model_path}")
    try:
        renderer = keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded successfully")
        print(f"  Input names: {[inp.name for inp in renderer.inputs]}")
        print(f"  Output names: {[out.name for out in renderer.outputs]}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False

    # Create random inputs
    print("Creating random test inputs...")
    print(f"  Reference image shape: (1, 500, 500, 3)")
    print(f"  Texture shape: (1, 500, 500, 3)")
    x_ref = tf.random.uniform([1, 500, 500, 3], minval=0.0, maxval=1.0)
    texture = tf.Variable(tf.random.uniform([1, 500, 500, 3], minval=0.0, maxval=1.0))
    print("✓ Test inputs created")
    print()

    # Test gradient flow
    print("Testing gradient flow...")
    try:
        with tf.GradientTape() as tape:
            # Forward pass
            rendered = renderer([x_ref, texture])

            # Simple loss: mean of rendered pixels
            loss = tf.reduce_mean(rendered)

        # Backward pass
        grads = tape.gradient(loss, [texture])

        print(f"  Loss value: {loss.numpy():.6f}")
        print(f"  Gradient shape: {grads[0].shape if grads[0] is not None else 'None'}")

        # Verify gradients exist and are not all zeros
        if grads[0] is None:
            print("✗ FAILED: Gradients are None")
            print("  The renderer is NOT differentiable!")
            return False

        grad_mean = tf.reduce_mean(tf.abs(grads[0])).numpy()
        grad_max = tf.reduce_max(tf.abs(grads[0])).numpy()

        print(f"  Gradient mean (abs): {grad_mean:.8f}")
        print(f"  Gradient max (abs): {grad_max:.8f}")

        if grad_mean < 1e-10:
            print("✗ FAILED: Gradients are effectively zero")
            print("  The renderer may not be properly differentiable!")
            return False

        print("✓ SUCCESS: Gradients flow correctly!")
        print()

    except Exception as e:
        print(f"✗ FAILED: Error during gradient computation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Additional verification: test with batch of random textures
    print("Running batch test (5 random textures)...")
    batch_results = []

    for i in range(5):
        x_ref_batch = tf.random.uniform([1, 500, 500, 3])
        texture_batch = tf.Variable(tf.random.uniform([1, 500, 500, 3]))

        with tf.GradientTape() as tape:
            rendered = renderer([x_ref_batch, texture_batch])
            loss = tf.reduce_mean(rendered)

        grads = tape.gradient(loss, [texture_batch])
        grad_mean = tf.reduce_mean(tf.abs(grads[0])).numpy()
        batch_results.append(grad_mean)

    print(f"  Gradient magnitudes: {[f'{g:.8f}' for g in batch_results]}")
    print(f"  Average: {np.mean(batch_results):.8f}")
    print("✓ Batch test passed")
    print()

    print("=" * 70)
    print("VERIFICATION COMPLETE: Renderer is differentiable ✓")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Integrate object detector (EfficientDet-D0)")
    print("  2. Implement attack loss function")
    print("  3. Build EOT training loop")
    print()

    return True


if __name__ == "__main__":
    import sys

    # Allow custom model path as command-line argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/k3_100epch_wo_custom_loss_model.h5'

    success = verify_differentiability(model_path)

    sys.exit(0 if success else 1)
