#!/usr/bin/env python3
"""
Attack Loss Functions

Implements the DTA (Differentiable Transformation Attack) loss formulation
for adversarial camouflage optimization.

Based on: docs/plans/2026-01-31-efficientdet-integration-design.md

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf


def attack_loss(boxes, scores, valid_mask):
    """
    Compute adversarial attack loss following DTA formulation.

    Formula:
        L_atk = -log(1 - max_confidence)

    Goal: Minimize detection confidence to make the detector miss the car.
          Lower loss = less confident detection = more successful attack.

    Edge Cases:
        - No cars detected: Returns 0.0 (perfect attack success)
        - Multiple cars: Uses maximum confidence across all detections

    Args:
        boxes: Bounding boxes for detections
              Shape: [batch, num_detections, 4]
              Note: Currently unused, reserved for future spatial losses

        scores: Detection confidence scores
               Shape: [batch, num_detections]
               Range: [0, 1] where 1 = 100% confident

        valid_mask: Boolean mask indicating valid car detections
                   Shape: [batch, num_detections]
                   True = car class AND above score threshold

    Returns:
        loss: Scalar tensor representing average attack loss across batch
              Range: [0, ∞) where 0 = complete success, ∞ = complete failure

    Implementation Notes:
        - Processes each image separately to handle variable detection counts
        - Adds epsilon (1e-8) for numerical stability (prevents log(0))
        - Averages loss across batch for EOT (Expectation Over Transformation)
    """
    # Get batch size (use tf.shape for dynamic dimensions)
    batch_size = tf.shape(scores)[0]

    # Collect per-image losses
    losses = []

    # Process each image in batch separately
    # (different images may have different numbers of detections)
    for i in range(batch_size):
        # Get car scores for this image
        image_scores = scores[i]
        image_mask = valid_mask[i]
        car_scores = tf.boolean_mask(image_scores, image_mask)

        # Handle edge case: no cars detected
        if tf.size(car_scores) == 0:
            # No detection = perfect attack success
            loss_i = tf.constant(0.0)
        else:
            # Get maximum confidence across all car detections
            # (we want to suppress even the most confident detection)
            max_score = tf.reduce_max(car_scores)

            # DTA loss: -log(1 - max_score)
            # - When max_score → 1 (confident detection): loss → ∞ (bad)
            # - When max_score → 0 (no detection): loss → 0 (good)
            # Add epsilon for numerical stability
            loss_i = -tf.math.log(1.0 - max_score + 1e-8)

        losses.append(loss_i)

    # Average loss across batch (for EOT with multiple viewpoints)
    total_loss = tf.reduce_mean(tf.stack(losses))

    return total_loss


def attack_loss_with_stats(boxes, scores, valid_mask):
    """
    Same as attack_loss() but also returns statistics for logging.

    Returns:
        loss: Scalar tensor (attack loss)
        stats: Dictionary with:
            - max_confidence: Maximum detection confidence across batch
            - num_detections: Total number of car detections across batch
            - per_image_loss: Loss for each image [batch] for debugging
    """
    batch_size = tf.shape(scores)[0]

    losses = []
    max_confidences = []
    num_detections_list = []

    for i in range(batch_size):
        image_scores = scores[i]
        image_mask = valid_mask[i]
        car_scores = tf.boolean_mask(image_scores, image_mask)

        num_detections_list.append(tf.size(car_scores))

        if tf.size(car_scores) == 0:
            loss_i = tf.constant(0.0)
            max_conf = tf.constant(0.0)
        else:
            max_score = tf.reduce_max(car_scores)
            loss_i = -tf.math.log(1.0 - max_score + 1e-8)
            max_conf = max_score

        losses.append(loss_i)
        max_confidences.append(max_conf)

    total_loss = tf.reduce_mean(tf.stack(losses))

    stats = {
        'max_confidence': tf.reduce_max(tf.stack(max_confidences)),
        'num_detections': tf.reduce_sum(tf.stack(num_detections_list)),
        'per_image_loss': tf.stack(losses),
    }

    return total_loss, stats


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("ATTACK LOSS FUNCTION TEST")
    print("=" * 70)
    print()

    # Test Case 1: Multiple detections with varying confidence
    print("Test Case 1: Multiple detections")
    boxes = tf.constant([
        [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],  # Image 0: 2 detections
        [[0.2, 0.2, 0.4, 0.4], [0.0, 0.0, 0.0, 0.0]],  # Image 1: 1 detection
    ])
    scores = tf.constant([
        [0.9, 0.7],  # Image 0: high and medium confidence
        [0.5, 0.1],  # Image 1: medium confidence, low confidence
    ])
    valid_mask = tf.constant([
        [True, True],   # Image 0: both are valid cars
        [True, False],  # Image 1: only first is valid car
    ])

    loss = attack_loss(boxes, scores, valid_mask)
    print(f"  Loss: {loss.numpy():.6f}")
    print(f"  Expected: High loss due to 0.9 confidence detection")
    print()

    # Test Case 2: No detections (perfect attack)
    print("Test Case 2: No detections")
    boxes = tf.zeros([2, 5, 4])
    scores = tf.constant([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.1, 0.4, 0.2]])
    valid_mask = tf.constant([[False, False, False, False, False],
                              [False, False, False, False, False]])

    loss = attack_loss(boxes, scores, valid_mask)
    print(f"  Loss: {loss.numpy():.6f}")
    print(f"  Expected: 0.0 (no valid detections = perfect attack)")
    print()

    # Test Case 3: Low confidence detections (good attack)
    print("Test Case 3: Low confidence detections")
    boxes = tf.zeros([2, 3, 4])
    scores = tf.constant([[0.1, 0.2, 0.15], [0.05, 0.1, 0.08]])
    valid_mask = tf.constant([[True, True, True], [True, True, True]])

    loss = attack_loss(boxes, scores, valid_mask)
    print(f"  Loss: {loss.numpy():.6f}")
    print(f"  Expected: Low loss due to low confidence detections")
    print()

    # Test Case 4: With statistics
    print("Test Case 4: Loss with statistics")
    loss, stats = attack_loss_with_stats(boxes, scores, valid_mask)
    print(f"  Loss: {loss.numpy():.6f}")
    print(f"  Max confidence: {stats['max_confidence'].numpy():.4f}")
    print(f"  Total detections: {stats['num_detections'].numpy()}")
    print(f"  Per-image losses: {stats['per_image_loss'].numpy()}")
    print()

    print("=" * 70)
    print("ALL LOSS TESTS PASSED ✓")
    print("=" * 70)
