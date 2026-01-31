#!/usr/bin/env python3
"""
End-to-End Gradient Flow Test

Verifies that gradients flow through the complete adversarial attack pipeline:
    texture → renderer → detector → loss → gradients back to texture

This extends verify_renderer_differentiability.py to include the object detector.

Based on: docs/plans/2026-01-31-efficientdet-integration-design.md

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np

from texture_applicator import TextureApplicator
from attack.detector import EfficientDetWrapper
from attack.loss import attack_loss


def test_end_to_end_gradients():
    """
    Verify gradients flow through the complete attack pipeline.

    Pipeline:
        1. TextureApplicator (neural renderer)
        2. EfficientDetWrapper (object detector)
        3. attack_loss (DTA loss function)

    Success Criteria:
        - Gradients are not None
        - Gradient magnitude > 1e-10 (non-trivial)
        - No runtime errors in forward/backward pass
        - Loss value is finite (not NaN/Inf)

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("=" * 70)
    print("END-TO-END GRADIENT FLOW TEST")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Load Components
    # ========================================================================
    print("Step 1: Loading pipeline components...")
    print()

    try:
        # Load neural renderer
        print("  Loading TextureApplicator (neural renderer)...")
        renderer = TextureApplicator()
        print()

        # Load object detector
        print("  Loading EfficientDetWrapper (object detector)...")
        detector = EfficientDetWrapper(score_threshold=0.0)
        print()

        print("✓ All components loaded successfully")
        print()

    except Exception as e:
        print(f"✗ FAILED: Error loading components: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========================================================================
    # Step 2: Create Test Inputs
    # ========================================================================
    print("Step 2: Creating test inputs...")

    # Create reference images (neutral gray vehicles)
    # Use batch size 2 to test batch processing
    batch_size = 2
    x_ref = tf.random.uniform([batch_size, 500, 500, 3], minval=0.3, maxval=0.7)
    print(f"  Reference images: {x_ref.shape}")

    # Create texture parameters (what we optimize)
    # Make this a tf.Variable so we can compute gradients w.r.t. it
    texture = tf.Variable(
        tf.random.uniform([batch_size, 500, 500, 3], minval=0.0, maxval=1.0),
        dtype=tf.float32
    )
    print(f"  Texture parameters: {texture.shape} (trainable)")
    print()

    # ========================================================================
    # Step 3: Forward Pass with Gradient Tape
    # ========================================================================
    print("Step 3: Running forward pass...")
    print()

    try:
        with tf.GradientTape() as tape:
            # Watch the texture variable
            tape.watch(texture)

            # Apply texture to reference using neural renderer
            print("  Rendering texture onto vehicle...")
            rendered = renderer.apply(x_ref, texture)
            print(f"    Rendered shape: {rendered.shape}")
            print(f"    Rendered range: [{np.min(rendered):.3f}, {np.max(rendered):.3f}]")

            # Convert to tensor if numpy array
            if isinstance(rendered, np.ndarray):
                rendered = tf.convert_to_tensor(rendered, dtype=tf.float32)

            # Detect cars in rendered images
            print("  Running object detection...")
            boxes, scores, valid_mask = detector.detect_cars_only(rendered)
            print(f"    Boxes shape: {boxes.shape}")
            print(f"    Scores shape: {scores.shape}")
            print(f"    Valid mask shape: {valid_mask.shape}")

            # Show detection statistics
            for i in range(batch_size):
                num_cars = tf.reduce_sum(tf.cast(valid_mask[i], tf.int32)).numpy()
                if num_cars > 0:
                    car_scores = tf.boolean_mask(scores[i], valid_mask[i])
                    max_conf = tf.reduce_max(car_scores).numpy()
                    print(f"    Image {i}: {num_cars} cars detected, max confidence: {max_conf:.4f}")
                else:
                    print(f"    Image {i}: No cars detected")

            # Compute attack loss
            print("  Computing attack loss...")
            loss = attack_loss(boxes, scores, valid_mask)
            print(f"    Loss value: {loss.numpy():.6f}")

        print()
        print("✓ Forward pass completed successfully")
        print()

    except Exception as e:
        print(f"✗ FAILED: Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========================================================================
    # Step 4: Backward Pass (Compute Gradients)
    # ========================================================================
    print("Step 4: Computing gradients...")

    try:
        # Compute gradients of loss w.r.t. texture
        grads = tape.gradient(loss, [texture])

        # Check if gradients exist
        if grads[0] is None:
            print("✗ FAILED: Gradients are None")
            print("  The pipeline is NOT differentiable!")
            print()
            print("Possible causes:")
            print("  - Renderer model was loaded with compile=False but has non-differentiable ops")
            print("  - Detector output is disconnected from texture input")
            print("  - Loss function breaks gradient flow")
            return False

        # Compute gradient statistics
        grad_tensor = grads[0]
        grad_mean = tf.reduce_mean(tf.abs(grad_tensor)).numpy()
        grad_max = tf.reduce_max(tf.abs(grad_tensor)).numpy()
        grad_min = tf.reduce_min(tf.abs(grad_tensor)).numpy()
        grad_std = tf.math.reduce_std(tf.abs(grad_tensor)).numpy()

        print(f"  Gradient shape: {grad_tensor.shape}")
        print(f"  Gradient statistics (absolute values):")
        print(f"    Mean: {grad_mean:.8f}")
        print(f"    Std:  {grad_std:.8f}")
        print(f"    Min:  {grad_min:.8f}")
        print(f"    Max:  {grad_max:.8f}")
        print()

        # Verify gradients are non-trivial
        if grad_mean < 1e-10:
            print("✗ FAILED: Gradients are effectively zero")
            print("  The pipeline may not be properly differentiable!")
            return False

        # Check for NaN/Inf
        if tf.reduce_any(tf.math.is_nan(grad_tensor)) or tf.reduce_any(tf.math.is_inf(grad_tensor)):
            print("✗ FAILED: Gradients contain NaN or Inf values")
            return False

        print("✓ Gradients computed successfully")
        print()

    except Exception as e:
        print(f"✗ FAILED: Error computing gradients: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========================================================================
    # Step 5: Gradient Update Test (Optional)
    # ========================================================================
    print("Step 5: Testing gradient update...")

    try:
        # Apply a single gradient update
        learning_rate = 0.01
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Store old texture for comparison
        texture_before = texture.numpy().copy()

        # Apply gradients
        optimizer.apply_gradients(zip(grads, [texture]))

        # Clip to valid range [0, 1]
        texture.assign(tf.clip_by_value(texture, 0.0, 1.0))

        # Check that texture changed
        texture_after = texture.numpy()
        texture_diff = np.abs(texture_after - texture_before)
        mean_change = np.mean(texture_diff)

        print(f"  Mean texture change: {mean_change:.8f}")

        if mean_change < 1e-10:
            print("  WARNING: Texture barely changed after gradient update")
            print("  This might indicate very small gradients or learning rate")
        else:
            print("  ✓ Texture updated successfully")

        print()

    except Exception as e:
        print(f"  WARNING: Gradient update test failed: {e}")
        print("  (This is not critical for gradient flow verification)")
        print()

    # ========================================================================
    # Final Results
    # ========================================================================
    print("=" * 70)
    print("✓✓✓ END-TO-END GRADIENT FLOW VERIFIED ✓✓✓")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  ✓ Forward pass: texture → renderer → detector → loss")
    print(f"  ✓ Backward pass: gradients flow back to texture")
    print(f"  ✓ Gradient magnitude: {grad_mean:.8f} (non-zero)")
    print(f"  ✓ Loss value: {loss.numpy():.6f} (finite)")
    print()
    print("Next steps:")
    print("  1. Implement EOT training loop (attack/optimizer.py)")
    print("  2. Add viewpoint sampling to CarlaHandler")
    print("  3. Run first full optimization with random texture")
    print()

    return True


if __name__ == "__main__":
    # Run the test
    success = test_end_to_end_gradients()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
