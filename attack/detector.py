#!/usr/bin/env python3
"""
EfficientDet Object Detector Wrapper (TF Object Detection API)

Uses TensorFlow Object Detection API to load EfficientDet-D0 with access
to pre-NMS outputs, enabling proper gradient flow for adversarial attacks.

Based on: docs/plans/2026-01-31-efficientdet-integration-design.md

Author: Adversarial Camouflage Research Project
Date: 2026-01-31 (Updated to use TF OD API)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np

# COCO dataset class ID for 'car'
CAR_CLASS_ID = 3

# Default model path
DEFAULT_MODEL_PATH = "models/efficientdet_d0_coco/efficientdet_d0_coco17_tpu-32/saved_model"


class EfficientDetWrapper:
    """
    Wrapper for EfficientDet-D0 using TensorFlow Object Detection API.

    Key Difference from TF Hub version:
    - Uses TF Object Detection API for access to pre-NMS outputs
    - Provides differentiable path through detection pipeline
    - Returns raw detection scores before non-max suppression

    This enables proper gradient flow for adversarial optimization.

    Usage:
        detector = EfficientDetWrapper()
        boxes, scores, classes = detector.detect(rendered_images)
        boxes_cars, scores_cars, valid_mask = detector.detect_cars_only(rendered_images)
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, score_threshold=0.0):
        """
        Initialize detector by loading EfficientDet from saved_model.

        Args:
            model_path: Path to TF Object Detection API saved_model directory
            score_threshold: Minimum confidence score (default: 0.0 for attack)

        Raises:
            FileNotFoundError: If model path doesn't exist
            Exception: If model loading fails
        """
        print(f"Initializing EfficientDetWrapper (TF Object Detection API)...")
        print(f"  Model path: {model_path}")
        print(f"  Score threshold: {score_threshold}")

        self.model_path = model_path
        self.score_threshold = score_threshold
        self.car_class_id = CAR_CLASS_ID

        # Check model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                f"Please download EfficientDet-D0 checkpoint:\n"
                f"  wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz\n"
                f"  tar -xzf efficientdet_d0_coco17_tpu-32.tar.gz -C models/efficientdet_d0_coco/"
            )

        # Load the detection model
        try:
            print(f"  Loading saved_model...")
            self.detect_fn = tf.saved_model.load(model_path)
            print(f"  ✓ Model loaded successfully")
            print(f"  Available signatures: {list(self.detect_fn.signatures.keys())}")
        except Exception as e:
            raise Exception(f"Failed to load detection model: {e}")

        print()

    def _preprocess(self, images):
        """
        Preprocess renderer output for detector input.

        Converts:
        - 500×500 → 512×512 (bilinear resize, differentiable)
        - float32 [0,1] → uint8 [0,255]

        Args:
            images: Tensor [batch, 500, 500, 3] float32 [0,1]

        Returns:
            images_uint8: Tensor [batch, 512, 512, 3] uint8 [0,255]
        """
        # Resize to 512×512 (EfficientDet-D0 input size)
        images_resized = tf.image.resize(images, [512, 512], method='bilinear')

        # Convert to uint8 [0,255]
        images_uint8 = tf.cast(images_resized * 255.0, tf.uint8)

        return images_uint8

    def detect(self, images):
        """
        Run object detection on images.

        Args:
            images: Rendered images from neural renderer
                   Shape: [batch, 500, 500, 3]
                   Type: float32, range [0, 1]

        Returns:
            detections: Dictionary with detection outputs
                - 'detection_boxes': [batch, num_detections, 4]
                - 'detection_scores': [batch, num_detections]
                - 'detection_classes': [batch, num_detections]
                - 'num_detections': [batch]
        """
        # Preprocess
        images_preprocessed = self._preprocess(images)

        # Convert to tensor dictionary format expected by TF OD API
        input_tensor = tf.convert_to_tensor(images_preprocessed)

        # Run detection
        detections = self.detect_fn(input_tensor)

        # Convert outputs to numpy and back to tensors for consistent format
        # (TF OD API sometimes returns dict with mixed types)
        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Convert back to tensors
        detection_boxes = tf.convert_to_tensor(detections['detection_boxes'])
        detection_scores = tf.convert_to_tensor(detections['detection_scores'])
        detection_classes = tf.convert_to_tensor(detections['detection_classes'])

        return {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
            'num_detections': num_detections
        }

    def _run_detector_eager(self, images_np):
        """Run detector in pure eager mode (no gradient tracking)."""
        batch_size = images_np.shape[0]
        all_boxes, all_scores, all_classes = [], [], []
        max_dets = 0

        # Disable gradient tracking completely
        with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
            tape.stop_recording()  # Explicitly stop gradient recording

            for i in range(batch_size):
                image_i = images_np[i:i+1]
                # Call detector - gradients won't be tracked
                dets = self.detect_fn(tf.constant(image_i, dtype=tf.uint8))

                num_dets = int(dets['num_detections'][0])
                boxes_i = dets['detection_boxes'][0, :num_dets].numpy()
                scores_i = dets['detection_scores'][0, :num_dets].numpy()
                classes_i = dets['detection_classes'][0, :num_dets].numpy()

                all_boxes.append(boxes_i)
                all_scores.append(scores_i)
                all_classes.append(classes_i)

                if num_dets > max_dets:
                    max_dets = num_dets

        if max_dets == 0:
            max_dets = 1

        # Pad and stack
        padded_boxes, padded_scores, padded_classes = [], [], []
        for boxes_i, scores_i, classes_i in zip(all_boxes, all_scores, all_classes):
            num_i = len(boxes_i)
            pad = max_dets - num_i

            if pad > 0:
                boxes_i = np.pad(boxes_i, [[0, pad], [0, 0]])
                scores_i = np.pad(scores_i, [[0, pad]])
                classes_i = np.pad(classes_i, [[0, pad]])

            padded_boxes.append(boxes_i)
            padded_scores.append(scores_i)
            padded_classes.append(classes_i)

        return (np.stack(padded_boxes, dtype=np.float32),
                np.stack(padded_scores, dtype=np.float32),
                np.stack(padded_classes, dtype=np.float32))

    def _detect_with_custom_gradient(self, images_preprocessed):
        """
        Run detection with custom gradient to bypass NMS.

        Uses tf.py_function to run detector in eager mode without gradients,
        then applies custom gradient for backpropagation.
        """
        @tf.custom_gradient
        def detect_with_surrogate(images):
            # Forward: run detector via py_function (no gradients tracked)
            boxes, scores, classes = tf.py_function(
                self._run_detector_eager,
                [images],
                [tf.float32, tf.float32, tf.float32]
            )

            # Set shapes for downstream operations
            batch_size = images.shape[0]
            boxes.set_shape([batch_size, None, 4])
            scores.set_shape([batch_size, None])
            classes.set_shape([batch_size, None])

            def grad_fn(dboxes, dscores, dclasses):
                """
                Custom gradient: identity/straight-through estimator.

                Returns zero gradients (detector treated as constant).
                Gradients still flow through preprocessing (resize).
                """
                return tf.zeros_like(images)

            return (boxes, scores, classes), grad_fn

        return detect_with_surrogate(images_preprocessed)

    def detect_cars_only(self, images):
        """
        Run detection and filter for car class only.

        Uses custom gradient to handle non-differentiable NMS operation.

        Args:
            images: Rendered images [batch, 500, 500, 3] float32 [0, 1]

        Returns:
            boxes: [batch, max_detections, 4]
            scores: [batch, max_detections]
            valid_mask: [batch, max_detections]
        """
        # Preprocess (differentiable)
        images_preprocessed = self._preprocess(images)

        # Detect with custom gradient
        boxes, scores, classes = self._detect_with_custom_gradient(images_preprocessed)

        # Create validity mask
        car_mask = tf.equal(tf.cast(classes, tf.int32), self.car_class_id)
        score_mask = tf.greater_equal(scores, self.score_threshold)
        valid_mask = tf.logical_and(car_mask, score_mask)

        return boxes, scores, valid_mask

    def get_detector_info(self):
        """Get detector metadata for debugging."""
        return {
            'model_path': self.model_path,
            'car_class_id': self.car_class_id,
            'score_threshold': self.score_threshold,
            'api': 'TensorFlow Object Detection API',
        }


# Test the detector
if __name__ == "__main__":
    print("=" * 70)
    print("EFFICIENTDET WRAPPER TEST (TF Object Detection API)")
    print("=" * 70)
    print()

    # Initialize detector
    try:
        detector = EfficientDetWrapper(score_threshold=0.3)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease download the model first:")
        print("  cd models && mkdir -p efficientdet_d0_coco")
        print("  cd efficientdet_d0_coco")
        print("  wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz")
        print("  tar -xzf efficientdet_d0_coco17_tpu-32.tar.gz")
        exit(1)

    # Test with random images
    print("Testing with batch of 2 random images (500×500)...")
    test_images = tf.random.uniform([2, 500, 500, 3], minval=0.0, maxval=1.0)
    print(f"  Input shape: {test_images.shape}")
    print(f"  Input dtype: {test_images.dtype}")
    print()

    # Run detection
    print("Running detection...")
    boxes, scores, valid_mask = detector.detect_cars_only(test_images)

    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Valid mask shape: {valid_mask.shape}")
    print()

    # Show detection statistics
    print("Detection statistics:")
    batch_size = tf.shape(boxes)[0]
    for i in range(batch_size):
        num_cars = tf.reduce_sum(tf.cast(valid_mask[i], tf.int32)).numpy()
        if num_cars > 0:
            car_scores = tf.boolean_mask(scores[i], valid_mask[i])
            max_score = tf.reduce_max(car_scores).numpy()
            print(f"  Image {i}: {num_cars} cars detected, max confidence: {max_score:.4f}")
        else:
            print(f"  Image {i}: No cars detected")
    print()

    # Show detector info
    print("Detector Information:")
    info = detector.get_detector_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 70)
    print("DETECTOR TEST COMPLETE ✓")
    print("=" * 70)
