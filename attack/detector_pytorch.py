#!/usr/bin/env python3
"""
PyTorch EfficientDet Wrapper with Pre-NMS Access

Uses Ross Wightman's effdet library to access raw detection outputs
BEFORE Non-Max Suppression, enabling proper gradient flow for
adversarial attacks.

Based on: docs/plans/2026-01-31-nms-gradient-issue.md

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn.functional as F
import numpy as np
from effdet import create_model, get_efficientdet_config
from effdet.anchors import Anchors, AnchorLabeler

# COCO class ID for 'car'
CAR_CLASS_ID = 2  # Note: PyTorch COCO uses 0-indexed (car=2), TF uses 1-indexed (car=3)


class EfficientDetPyTorch:
    """
    PyTorch EfficientDet wrapper with pre-NMS access for adversarial attacks.

    Key Features:
    - Access to raw class logits (before sigmoid/NMS)
    - Access to raw box predictions (before NMS)
    - Differentiable forward pass (gradients flow properly)
    - Compatible with TensorFlow renderer via numpy bridge

    Usage:
        detector = EfficientDetPyTorch()
        class_logits, box_preds = detector.forward_pre_nms(images_torch)
        # Compute loss on raw logits (differentiable!)
    """

    def __init__(self, model_name='tf_efficientdet_d0', pretrained=True, device=None):
        """
        Initialize PyTorch EfficientDet model.

        Args:
            model_name: EfficientDet variant (d0-d7)
            pretrained: Load pretrained weights
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        print(f"Initializing PyTorch EfficientDet...")
        print(f"  Model: {model_name}")
        print(f"  Pretrained: {pretrained}")

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        print(f"  Device: {self.device}")

        # Create model
        try:
            self.model = create_model(
                model_name,
                pretrained=pretrained,
                num_classes=90,  # COCO has 90 classes
                image_size=(512, 512),  # Input size for EfficientDet-D0
            )
            self.model.to(self.device)
            self.model.eval()  # Inference mode
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load EfficientDet model: {e}")

        # Get model config for anchor generation
        self.config = get_efficientdet_config(model_name)
        self.car_class_id = CAR_CLASS_ID

        # Initialize anchors
        self.anchors = Anchors.from_config(self.config)

        print()

    def preprocess(self, images_np):
        """
        Preprocess images for EfficientDet.

        Args:
            images_np: NumPy array [batch, H, W, 3] float32 [0, 1] (NHWC from TF)

        Returns:
            images_torch: PyTorch tensor [batch, 3, H, W] float32 [0, 1] (NCHW)
        """
        # Convert to PyTorch tensor
        images_torch = torch.from_numpy(images_np).to(self.device)

        # NHWC → NCHW
        images_torch = images_torch.permute(0, 3, 1, 2)

        # Resize to 512×512 if needed
        if images_torch.shape[2:] != (512, 512):
            images_torch = F.interpolate(
                images_torch,
                size=(512, 512),
                mode='bilinear',
                align_corners=False
            )

        return images_torch

    def forward_pre_nms(self, images_torch):
        """
        Forward pass returning PRE-NMS outputs (NO gradients).

        This method is for inference/evaluation only. For training with
        gradient flow, use forward_pre_nms_with_grad() instead.

        Args:
            images_torch: [batch, 3, 512, 512] float32 [0, 1]

        Returns:
            class_logits: [batch, num_anchors, num_classes] raw logits
            box_preds: [batch, num_anchors, 4] box deltas
        """
        with torch.no_grad():
            return self._forward_pre_nms_impl(images_torch)

    def forward_pre_nms_with_grad(self, images_torch):
        """
        Forward pass returning PRE-NMS outputs WITH gradient tracking.

        This is the key method for adversarial attacks with end-to-end
        gradient flow. Unlike forward_pre_nms(), this method does NOT
        wrap the computation in torch.no_grad(), allowing gradients to
        flow back through the detector to the renderer and texture.

        Args:
            images_torch: [batch, 3, 512, 512] float32 [0, 1]
                         Should have requires_grad=True somewhere upstream

        Returns:
            class_logits: [batch, num_anchors, num_classes] raw logits (with grad)
            box_preds: [batch, num_anchors, 4] box deltas (with grad)
        """
        return self._forward_pre_nms_impl(images_torch)

    def _forward_pre_nms_impl(self, images_torch):
        """
        Internal implementation of pre-NMS forward pass.

        Shared by both forward_pre_nms() and forward_pre_nms_with_grad().

        Args:
            images_torch: [batch, 3, 512, 512] float32 [0, 1]

        Returns:
            class_logits: [batch, num_anchors, num_classes] raw logits
            box_preds: [batch, num_anchors, 4] box deltas
        """
        # Run model forward pass
        # EfficientDet returns a list of outputs (one per FPN level)
        # We need to concatenate them
        outputs = self.model(images_torch)

        # Handle different output formats from effdet
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            class_out, box_out = outputs

            # If outputs are lists of tensors (multi-level FPN), concatenate
            if isinstance(class_out, list):
                # Flatten spatial dimensions: [B, C, H, W] → [B, H*W, C]
                class_out_flat = []
                for c in class_out:
                    B, C, H, W = c.shape
                    c_flat = c.permute(0, 2, 3, 1).reshape(B, H*W, C)
                    class_out_flat.append(c_flat)
                class_out = torch.cat(class_out_flat, dim=1)

            if isinstance(box_out, list):
                # Flatten spatial dimensions: [B, 4, H, W] → [B, H*W, 4]
                box_out_flat = []
                for b in box_out:
                    B, C, H, W = b.shape
                    b_flat = b.permute(0, 2, 3, 1).reshape(B, H*W, C)
                    box_out_flat.append(b_flat)
                box_out = torch.cat(box_out_flat, dim=1)

        else:
            raise ValueError(f"Unexpected output format from effdet: {type(outputs)}")

        return class_out, box_out

    def detect_cars_only(self, images_np, score_threshold=0.5):
        """
        Run full detection pipeline with NMS (for evaluation, not training).

        Args:
            images_np: NumPy [batch, H, W, 3] float32 [0, 1]
            score_threshold: Confidence threshold

        Returns:
            boxes: List of [N, 4] arrays (one per image)
            scores: List of [N] arrays
            classes: List of [N] arrays
        """
        # Preprocess
        images_torch = self.preprocess(images_np)

        with torch.no_grad():
            # Get raw outputs
            class_out, box_out = self.model(images_torch)

            # Apply sigmoid to get probabilities
            class_probs = torch.sigmoid(class_out)  # [batch, anchors, classes]

            # Get car class probabilities
            car_probs = class_probs[:, :, self.car_class_id]  # [batch, anchors]

            # Simple per-image NMS (for evaluation)
            results_boxes = []
            results_scores = []
            results_classes = []

            batch_size = images_torch.shape[0]
            for i in range(batch_size):
                # Get car scores for this image
                scores_i = car_probs[i]  # [anchors]
                boxes_i = box_out[i]     # [anchors, 4]

                # Filter by threshold
                mask = scores_i > score_threshold
                filtered_scores = scores_i[mask].cpu().numpy()
                filtered_boxes = boxes_i[mask].cpu().numpy()

                if len(filtered_scores) > 0:
                    # Simple NMS (keep top-k by score)
                    top_k = min(100, len(filtered_scores))
                    indices = np.argsort(-filtered_scores)[:top_k]

                    results_boxes.append(filtered_boxes[indices])
                    results_scores.append(filtered_scores[indices])
                    results_classes.append(np.full(len(indices), self.car_class_id))
                else:
                    results_boxes.append(np.zeros((0, 4)))
                    results_scores.append(np.zeros((0,)))
                    results_classes.append(np.zeros((0,)))

            return results_boxes, results_scores, results_classes

    def get_max_car_confidence(self, class_logits):
        """
        Get maximum car confidence from raw logits (for loss computation).

        Args:
            class_logits: [batch, anchors, classes] raw logits

        Returns:
            max_conf: [batch] maximum car confidence per image
        """
        # Get car logits
        car_logits = class_logits[:, :, self.car_class_id]  # [batch, anchors]

        # Apply sigmoid to get probabilities
        car_probs = torch.sigmoid(car_logits)

        # Get max confidence per image
        max_conf, _ = torch.max(car_probs, dim=1)  # [batch]

        return max_conf

    def get_detector_info(self):
        """Get detector metadata."""
        return {
            'framework': 'PyTorch',
            'model': self.config.name,
            'car_class_id': self.car_class_id,
            'device': str(self.device),
            'input_size': (512, 512),
            'differentiable': True,  # Pre-NMS outputs are differentiable!
        }


# Test script
if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH EFFICIENTDET TEST")
    print("=" * 70)
    print()

    # Initialize detector
    detector = EfficientDetPyTorch()

    # Test with random images (NumPy format from TensorFlow)
    print("Testing with batch of 2 random images (500×500)...")
    test_images_np = np.random.rand(2, 500, 500, 3).astype(np.float32)
    print(f"  Input shape (NumPy): {test_images_np.shape}")
    print()

    # Test preprocessing
    print("Testing preprocessing...")
    images_torch = detector.preprocess(test_images_np)
    print(f"  Output shape (PyTorch): {images_torch.shape}")
    print(f"  Device: {images_torch.device}")
    print()

    # Test pre-NMS forward pass
    print("Testing pre-NMS forward pass...")
    class_out, box_out = detector.forward_pre_nms(images_torch)
    print(f"  Class logits shape: {class_out.shape}")
    print(f"  Box predictions shape: {box_out.shape}")
    print()

    # Test max car confidence extraction
    print("Testing max car confidence...")
    max_conf = detector.get_max_car_confidence(class_out)
    print(f"  Max confidence shape: {max_conf.shape}")
    print(f"  Max confidence values: {max_conf.cpu().numpy()}")
    print()

    # Test full detection with NMS
    print("Testing full detection with NMS...")
    boxes, scores, classes = detector.detect_cars_only(test_images_np, score_threshold=0.3)
    print(f"  Image 0: {len(boxes[0])} cars detected")
    print(f"  Image 1: {len(boxes[1])} cars detected")
    print()

    # Show detector info
    print("Detector Information:")
    info = detector.get_detector_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 70)
    print("PYTORCH DETECTOR TEST COMPLETE ✓")
    print("=" * 70)
