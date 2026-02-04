#!/usr/bin/env python3
"""
EOT (Expectation Over Transformation) Trainer

Trains adversarial camouflage textures across multiple viewpoints using
finite difference gradients through a hybrid TF/PyTorch pipeline.

Architecture:
    Texture (TF) → Renderer (TF) → Detector (PyTorch) → Loss (PyTorch)
    Gradients computed via finite differences for detector part.

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import torch
import numpy as np
import cv2
import json
from pathlib import Path
from datetime import datetime

from attack.loss_pytorch import attack_loss_pytorch
from attack.logger import CSVLogger


def create_viewpoint_configs():
    """
    Returns standard 6-viewpoint EOT configuration.

    Viewpoints arranged in 60° increments around vehicle:
    - 0°: Front view
    - 60°: Front-right
    - 120°: Back-right
    - 180°: Back
    - 240°: Back-left
    - 300°: Front-left

    All use pitch=-15° (looking down) and distance=8m.

    Returns:
        List of 6 dicts with {yaw, pitch, distance}
    """
    return [
        {'yaw': 0,   'pitch': -15, 'distance': 8},
        {'yaw': 60,  'pitch': -15, 'distance': 8},
        {'yaw': 120, 'pitch': -15, 'distance': 8},
        {'yaw': 180, 'pitch': -15, 'distance': 8},
        {'yaw': 240, 'pitch': -15, 'distance': 8},
        {'yaw': 300, 'pitch': -15, 'distance': 8},
    ]


def tile_texture_for_batch(texture, batch_size=6):
    """
    Repeat single texture to create batch for EOT.

    Args:
        texture: Single texture [500, 500, 3] or [1, 500, 500, 3]
        batch_size: Number of repetitions (default: 6 for viewpoints)

    Returns:
        Batched texture [batch_size, 500, 500, 3]
    """
    # Handle both 3D and 4D inputs
    if len(texture.shape) == 3:
        texture_single = texture
    elif len(texture.shape) == 4 and texture.shape[0] == 1:
        texture_single = texture[0]
    else:
        raise ValueError(f"Expected shape [500,500,3] or [1,500,500,3], got {texture.shape}")

    # Repeat along batch dimension
    texture_batch = tf.repeat(tf.expand_dims(texture_single, axis=0), batch_size, axis=0)

    return texture_batch


def upsample_texture(coarse_texture, target_size=(500, 500)):
    """
    Upsample coarse texture to full resolution using bicubic interpolation.

    Args:
        coarse_texture: Coarse texture [H, W, 3] or [1, H, W, 3]
        target_size: Target resolution (height, width), default (500, 500)

    Returns:
        Upsampled texture with same dimensionality as input
    """
    squeeze_output = False
    if len(coarse_texture.shape) == 3:
        coarse_texture = tf.expand_dims(coarse_texture, 0)
        squeeze_output = True

    upsampled = tf.image.resize(coarse_texture, target_size, method='bicubic')

    if squeeze_output:
        upsampled = tf.squeeze(upsampled, 0)

    return upsampled


def visualize_texture(texture_np, save_path):
    """
    Save texture as PNG for visual inspection.

    Args:
        texture_np: NumPy array [500, 500, 3] float32 [0, 1]
        save_path: Path to save PNG file
    """
    # Convert [0, 1] float to [0, 255] uint8
    texture_uint8 = (texture_np * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    texture_bgr = cv2.cvtColor(texture_uint8, cv2.COLOR_RGB2BGR)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), texture_bgr)


# Default training configuration
DEFAULT_CONFIG = {
    'learning_rate': 0.01,
    'num_iterations': 1000,
    'checkpoint_every': 100,
    'log_every': 10,
    'epsilon': 1e-4,  # Finite difference epsilon
    'optimizer': 'adam',
    'output_dir': 'experiments/phase1_eot/',
    'coarse_size': 128,  # Texture parameterization size (128×128 → 500×500)
}


# NOTE: Alternative gradient strategies (documented for future):
# - Zero-order optimization (CMA-ES, NES): No gradients, more robust
# - PyTorch gradient bridging: Enable detector grads, convert to TF
# See: docs/plans/2026-01-31-eot-training-loop-design.md


class EOTTrainer:
    """
    Expectation Over Transformation (EOT) trainer for adversarial textures.

    Optimizes a single texture across multiple viewpoints to minimize
    object detection confidence using finite difference gradients.

    Usage:
        trainer = EOTTrainer(carla, detector, renderer, viewpoints, config)
        results = trainer.train()
    """

    def __init__(self, carla_handler, detector, renderer, viewpoints, config):
        """
        Initialize EOT trainer.

        Args:
            carla_handler: CarlaHandler instance (must have vehicle spawned)
            detector: EfficientDetPyTorch instance
            renderer: TextureApplicator instance
            viewpoints: List of viewpoint dicts from create_viewpoint_configs()
            config: Training configuration dict (see DEFAULT_CONFIG)
        """
        self.carla = carla_handler
        self.detector = detector
        self.renderer = renderer
        self.viewpoints = viewpoints
        self.config = {**DEFAULT_CONFIG, **config}  # Merge with defaults

        # Validate inputs
        if self.carla.vehicle is None:
            raise ValueError("CarlaHandler must have a vehicle spawned")

        if len(self.viewpoints) != 6:
            print(f"Warning: Expected 6 viewpoints, got {len(self.viewpoints)}")

        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.viz_dir = self.output_dir / 'visualizations'
        self.final_dir = self.output_dir / 'final'

        self.checkpoint_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        self.final_dir.mkdir(exist_ok=True)

        # Save configuration
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"EOTTrainer initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Viewpoints: {len(self.viewpoints)}")
        print(f"  Config: {self.config}")

    def capture_reference_images(self):
        """
        Capture reference images from all viewpoints.

        This is called once at the start of training. Reference images show
        the vehicle with neutral color from each viewpoint.

        Returns:
            tf.Tensor: Batch of reference images [6, 500, 500, 3] float32 [0, 1]
        """
        print(f"Capturing reference images from {len(self.viewpoints)} viewpoints...")

        x_refs = []

        for i, vp in enumerate(self.viewpoints):
            # Position camera
            self.carla.set_camera_viewpoint(vp['yaw'], vp['pitch'], vp['distance'])

            # Capture image
            img = self.carla.get_image()  # Returns [H, W, 3] uint8 BGR

            # Convert to RGB float [0, 1]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0

            x_refs.append(img_float)

            print(f"  Viewpoint {i}: yaw={vp['yaw']:3d}° pitch={vp['pitch']:3d}° dist={vp['distance']}m")

        # Stack into batch
        x_ref_batch = tf.constant(np.stack(x_refs, axis=0), dtype=tf.float32)

        print(f"✓ Captured reference batch: {x_ref_batch.shape}")

        # Save reference images for debugging
        for i, img in enumerate(x_refs):
            save_path = self.output_dir / f'reference_view_{i}_yaw_{self.viewpoints[i]["yaw"]:03d}.png'
            visualize_texture(img, str(save_path))

        print(f"✓ Saved reference images to {self.output_dir}/reference_view_*.png")

        return x_ref_batch

    def initialize_texture(self, init_type='random_uniform'):
        """
        Initialize texture parameters as tf.Variable.

        Texture is initialized at coarse resolution (default 128×128) and
        upsampled to 500×500 before rendering. This reduces parameter count
        from 750k to ~49k for faster optimization.

        Args:
            init_type: Initialization strategy
                - 'random_uniform': Random values in [0, 1] (Phase 1 default)
                - 'random_normal': Normal distribution clipped to [0, 1]
                - 'constant': Constant gray (0.5)

        Returns:
            tf.Variable: Texture [coarse_size, coarse_size, 3] float32 [0, 1]
        """
        coarse_size = self.config['coarse_size']
        print(f"Initializing texture: {init_type} at {coarse_size}×{coarse_size}")

        if init_type == 'random_uniform':
            texture_init = tf.random.uniform([coarse_size, coarse_size, 3], minval=0.0, maxval=1.0)
        elif init_type == 'random_normal':
            texture_init = tf.random.normal([coarse_size, coarse_size, 3], mean=0.5, stddev=0.1)
            texture_init = tf.clip_by_value(texture_init, 0.0, 1.0)
        elif init_type == 'constant':
            texture_init = tf.ones([coarse_size, coarse_size, 3]) * 0.5
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        texture = tf.Variable(texture_init, dtype=tf.float32, trainable=True)

        param_count = coarse_size * coarse_size * 3
        print(f"  ✓ Texture shape: {texture.shape} ({param_count:,} parameters)")
        print(f"  ✓ Upsampled to: 500×500 for rendering")
        print(f"  ✓ Initial mean: {tf.reduce_mean(texture).numpy():.4f}")
        print(f"  ✓ Initial std: {tf.math.reduce_std(texture).numpy():.4f}")

        # Save initial texture (upsampled for visualization)
        texture_full = upsample_texture(texture, (500, 500))
        visualize_texture(texture_full.numpy(), str(self.viz_dir / 'texture_iter_0000.png'))
        np.save(str(self.checkpoint_dir / 'texture_iter_0000.npy'), texture.numpy())

        return texture

    def _forward_pass(self, texture, x_ref_batch):
        """
        Single forward pass through renderer + detector.

        Pipeline:
            1. Upsample coarse texture to full resolution (128×128 → 500×500)
            2. Tile texture for all viewpoints
            3. Render via neural renderer (TensorFlow)
            4. Convert to PyTorch and run detector
            5. Compute attack loss
            6. Extract metrics

        Args:
            texture: tf.Variable or tf.Tensor [coarse_size, coarse_size, 3]
            x_ref_batch: Reference images [6, 500, 500, 3]

        Returns:
            loss_value: Scalar float (attack loss)
            metrics: Dict with max_confidence, mean_confidence, per_view_conf
        """
        # 1. Upsample coarse texture to full resolution
        texture_full = upsample_texture(texture, (500, 500))

        # 2. Tile texture for all viewpoints
        texture_batch = tile_texture_for_batch(texture_full, batch_size=len(self.viewpoints))

        # 2. Render all views through neural renderer
        rendered_batch_tf = self.renderer.apply(x_ref_batch.numpy(), texture_batch.numpy())

        # 3. Convert to PyTorch format and preprocess
        rendered_torch = self.detector.preprocess(rendered_batch_tf)

        # 4. Run detector (in no_grad mode)
        class_logits, box_preds = self.detector.forward_pre_nms(rendered_torch)

        # 5. Compute attack loss
        loss_tensor = attack_loss_pytorch(class_logits, car_class_id=2)
        loss_value = loss_tensor.item()

        # 6. Extract metrics
        max_conf = self.detector.get_max_car_confidence(class_logits)

        metrics = {
            'max_confidence': torch.max(max_conf).item(),
            'mean_confidence': torch.mean(max_conf).item(),
            'per_view_conf': max_conf.cpu().numpy().tolist(),
        }

        return loss_value, metrics

    def _compute_gradients_finite_diff(self, texture, x_ref_batch, baseline_loss):
        """
        Compute gradients using central finite differences.

        Efficient approach: Perturb entire texture in random direction rather
        than per-pixel perturbations (which would require 750k forward passes).

        Algorithm:
            1. Generate random perturbation direction (normalized)
            2. Compute loss at texture + ε * direction
            3. Compute loss at texture - ε * direction
            4. Gradient ≈ (loss+ - loss-) / (2ε) * direction

        Args:
            texture: Current texture tf.Variable [500, 500, 3]
            x_ref_batch: Reference images [6, 500, 500, 3]
            baseline_loss: Current loss value (optional, for efficiency)

        Returns:
            gradients: tf.Tensor [500, 500, 3] (same shape as texture)
        """
        epsilon = self.config['epsilon']

        # 1. Generate random perturbation direction
        perturbation = tf.random.normal(texture.shape)
        perturbation = perturbation / tf.norm(perturbation)  # Normalize to unit vector

        # 2. Forward perturbation: texture + ε
        texture_plus = texture + epsilon * perturbation
        texture_plus = tf.clip_by_value(texture_plus, 0.0, 1.0)
        loss_plus, _ = self._forward_pass(texture_plus, x_ref_batch)

        # 3. Backward perturbation: texture - ε
        texture_minus = texture - epsilon * perturbation
        texture_minus = tf.clip_by_value(texture_minus, 0.0, 1.0)
        loss_minus, _ = self._forward_pass(texture_minus, x_ref_batch)

        # 4. Central difference: ∇f ≈ (f(x+ε) - f(x-ε)) / (2ε)
        grad_magnitude = (loss_plus - loss_minus) / (2 * epsilon)

        # 5. Gradient is in the direction of perturbation
        gradients = grad_magnitude * perturbation

        return gradients

    def _log_iteration(self, iteration, loss, metrics, gradients):
        """
        Log training progress to console.

        Args:
            iteration: Current iteration number
            loss: Current loss value
            metrics: Dict from _forward_pass()
            gradients: Current gradients tensor
        """
        grad_mag = tf.reduce_mean(tf.abs(gradients)).numpy()

        print(
            f"Iteration {iteration:4d}/{self.config['num_iterations']} | "
            f"Loss: {loss:.4f} | "
            f"Max Conf: {metrics['max_confidence']:.4f} | "
            f"Mean Conf: {metrics['mean_confidence']:.4f} | "
            f"Grad: {grad_mag:.6f}"
        )

    def _save_checkpoint(self, iteration, texture, metrics):
        """
        Save texture checkpoint and visualization.

        Args:
            iteration: Current iteration number
            texture: Current texture tf.Variable (coarse resolution)
            metrics: Current metrics dict
        """
        # Save numpy checkpoint (coarse resolution for resuming)
        texture_np = texture.numpy()
        checkpoint_path = self.checkpoint_dir / f'texture_iter_{iteration:04d}.npy'
        np.save(str(checkpoint_path), texture_np)

        # Save visualization (upsampled for viewing)
        texture_full = upsample_texture(texture, (500, 500))
        viz_path = self.viz_dir / f'texture_iter_{iteration:04d}.png'
        visualize_texture(texture_full.numpy(), str(viz_path))

        print(f"  Checkpoint saved: {checkpoint_path.name}")

    def _save_final_results(self, final_texture, training_history):
        """
        Save final results including texture, visualizations, and summary.

        Args:
            final_texture: Final optimized texture [coarse_size, coarse_size, 3]
            training_history: List of training data rows from CSV logger
        """
        # Save final texture (coarse resolution for resuming)
        np.save(str(self.final_dir / 'texture_final.npy'), final_texture)

        # Save visualization (upsampled for viewing)
        final_texture_tf = tf.constant(final_texture)
        final_texture_full = upsample_texture(final_texture_tf, (500, 500))
        visualize_texture(final_texture_full.numpy(), str(self.final_dir / 'texture_final.png'))

        # Create summary
        if len(training_history) > 0:
            last_row = training_history[-1]
            summary = {
                'training_config': self.config,
                'final_metrics': {
                    'loss': float(last_row[1]) if len(last_row) > 1 else None,
                    'max_confidence': float(last_row[2]) if len(last_row) > 2 else None,
                    'mean_confidence': float(last_row[3]) if len(last_row) > 3 else None,
                    'iterations_completed': len(training_history),
                },
                'timestamp': datetime.now().isoformat(),
            }
        else:
            summary = {'error': 'No training data recorded'}

        # Save summary JSON
        with open(self.final_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Final results saved to {self.final_dir}/")

    def train(self):
        """
        Main EOT training loop.

        Algorithm:
            1. Capture reference images from all viewpoints
            2. Initialize texture as tf.Variable
            3. Setup optimizer (Adam)
            4. For each iteration:
                a. Compute loss via forward pass
                b. Compute gradients via finite differences
                c. Update texture with optimizer
                d. Clip texture to [0, 1]
                e. Log metrics
                f. Save checkpoints
            5. Save final results

        Returns:
            dict: {
                'texture': Final texture array [500, 500, 3],
                'final_loss': Final loss value,
                'history': List of all training data rows
            }
        """
        print("=" * 70)
        print("STARTING EOT TRAINING")
        print("=" * 70)
        print()

        # === SETUP PHASE ===
        print("Setup Phase:")
        print("-" * 70)

        # 1. Capture reference images
        x_ref_batch = self.capture_reference_images()
        print()

        # 2. Initialize texture
        texture = self.initialize_texture('random_uniform')
        print()

        # 3. Setup optimizer
        optimizer = tf.optimizers.Adam(learning_rate=self.config['learning_rate'])
        print(f"Optimizer: Adam (lr={self.config['learning_rate']})")
        print()

        # 4. Initialize CSV loggers
        main_logger = CSVLogger(str(self.output_dir / 'training_log.csv'))
        main_logger.write_header([
            'iteration', 'loss', 'max_conf', 'mean_conf',
            'grad_magnitude', 'texture_mean', 'texture_std',
            'view_0_conf', 'view_1_conf', 'view_2_conf',
            'view_3_conf', 'view_4_conf', 'view_5_conf'
        ])

        viewpoint_logger = CSVLogger(str(self.output_dir / 'per_viewpoint_analysis.csv'))
        viewpoint_logger.write_header([
            'iteration', 'viewpoint_id', 'yaw', 'pitch', 'distance', 'confidence'
        ])
        print()

        # === TRAINING LOOP ===
        print("Training Phase:")
        print("-" * 70)

        for iteration in range(self.config['num_iterations']):

            # Forward pass
            current_loss, metrics = self._forward_pass(texture, x_ref_batch)

            # Compute gradients via finite differences
            gradients = self._compute_gradients_finite_diff(texture, x_ref_batch, current_loss)

            # Optimization step
            optimizer.apply_gradients([(gradients, texture)])

            # Clip texture to valid range
            texture.assign(tf.clip_by_value(texture, 0.0, 1.0))

            # Logging
            if iteration % self.config['log_every'] == 0:
                self._log_iteration(iteration, current_loss, metrics, gradients)

                # Main log
                main_logger.write_row([
                    iteration,
                    current_loss,
                    metrics['max_confidence'],
                    metrics['mean_confidence'],
                    tf.reduce_mean(tf.abs(gradients)).numpy(),
                    tf.reduce_mean(texture).numpy(),
                    tf.math.reduce_std(texture).numpy(),
                ] + metrics['per_view_conf'])

                # Per-viewpoint log
                for vp_id, (vp, conf) in enumerate(zip(self.viewpoints, metrics['per_view_conf'])):
                    viewpoint_logger.write_row([
                        iteration, vp_id, vp['yaw'], vp['pitch'], vp['distance'], conf
                    ])

            # Checkpointing
            if iteration % self.config['checkpoint_every'] == 0 and iteration > 0:
                self._save_checkpoint(iteration, texture, metrics)

        # === FINAL SAVE ===
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        final_texture = texture.numpy()
        self._save_final_results(final_texture, main_logger.data)

        # Close loggers
        main_logger.close()
        viewpoint_logger.close()

        return {
            'texture': final_texture,
            'final_loss': current_loss,
            'history': main_logger.data,
        }


# Test utilities
if __name__ == "__main__":
    print("=" * 70)
    print("EOT TRAINER TEST")
    print("=" * 70)
    print()

    # Test viewpoint configs
    print("Testing create_viewpoint_configs...")
    vps = create_viewpoint_configs()
    assert len(vps) == 6
    assert vps[0]['yaw'] == 0
    assert vps[5]['yaw'] == 300
    print(f"  ✓ Created {len(vps)} viewpoints")
    print()

    # Test texture tiling
    print("Testing tile_texture_for_batch...")
    texture = tf.random.uniform([500, 500, 3])
    tiled = tile_texture_for_batch(texture, batch_size=6)
    assert tiled.shape == (6, 500, 500, 3)
    print(f"  ✓ Tiled {texture.shape} → {tiled.shape}")
    print()

    # Test upsampling
    print("Testing upsample_texture...")
    coarse = tf.random.uniform([128, 128, 3])
    upsampled = upsample_texture(coarse, (500, 500))
    assert upsampled.shape == (500, 500, 3)
    print(f"  ✓ Upsampled {coarse.shape} → {upsampled.shape}")

    # Test with batch dimension
    coarse_batch = tf.random.uniform([1, 128, 128, 3])
    upsampled_batch = upsample_texture(coarse_batch, (500, 500))
    assert upsampled_batch.shape == (1, 500, 500, 3)
    print(f"  ✓ Upsampled batch {coarse_batch.shape} → {upsampled_batch.shape}")
    print()

    # Test visualization
    print("Testing visualize_texture...")
    test_texture = np.random.rand(500, 500, 3).astype(np.float32)
    visualize_texture(test_texture, 'test_output/test_texture.png')
    assert os.path.exists('test_output/test_texture.png')
    print(f"  ✓ Saved test_output/test_texture.png")
    print()

    # Test trainer initialization (mock objects)
    print("Testing EOTTrainer initialization...")

    # Create mock objects
    class MockCarla:
        vehicle = "mock_vehicle"

    class MockDetector:
        pass

    class MockRenderer:
        pass

    mock_config = {
        'learning_rate': 0.01,
        'num_iterations': 100,
        'output_dir': 'test_output/trainer_test/',
        'coarse_size': 128,
    }

    trainer = EOTTrainer(
        MockCarla(),
        MockDetector(),
        MockRenderer(),
        create_viewpoint_configs(),
        mock_config
    )

    assert trainer.output_dir.exists()
    assert trainer.checkpoint_dir.exists()
    assert (trainer.output_dir / 'config.json').exists()
    print(f"  ✓ Trainer initialized with output dir: {trainer.output_dir}")
    print()

    # Test texture initialization (coarse resolution)
    print("Testing initialize_texture (coarse parameterization)...")
    texture = trainer.initialize_texture('random_uniform')
    assert texture.shape == (128, 128, 3), f"Expected (128, 128, 3), got {texture.shape}"
    assert texture.dtype == tf.float32
    assert 0.0 <= tf.reduce_min(texture).numpy() <= 1.0
    assert 0.0 <= tf.reduce_max(texture).numpy() <= 1.0
    print(f"  ✓ Coarse texture initialized: {texture.shape} ({128*128*3:,} params)")

    # Verify upsampling for rendering
    texture_full = upsample_texture(texture, (500, 500))
    assert texture_full.shape == (500, 500, 3)
    print(f"  ✓ Upsampled for rendering: {texture_full.shape}")
    print()

    # Note: capture_reference_images() requires real CARLA connection
    # Note: _forward_pass() requires full pipeline (CARLA + detector + renderer)
    # These will be tested in experiments/phase1_random.py

    print("=" * 70)
    print("TRAINER TESTS PASSED ✓")
    print("=" * 70)
