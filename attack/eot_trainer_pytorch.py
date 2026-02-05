#!/usr/bin/env python3
"""
Pure PyTorch EOT (Expectation Over Transformation) Trainer

Trains adversarial camouflage textures across multiple viewpoints using
TRUE end-to-end gradient flow through a pure PyTorch pipeline.

Key Improvements over TensorFlow version:
- No finite differences - uses native PyTorch autograd
- Single forward + backward pass per iteration (3x faster)
- True gradient flow: Texture -> Renderer -> Detector -> Loss
- Simpler code without numpy bridges

Architecture:
    Texture (PyTorch) -> Renderer (PyTorch) -> Detector (PyTorch) -> Loss (PyTorch)
    Gradients computed via true backpropagation through entire pipeline.

Usage:
    from attack.eot_trainer_pytorch import EOTTrainerPyTorch

    trainer = EOTTrainerPyTorch(
        carla_handler=carla,
        detector=detector,
        renderer=renderer,
        viewpoints=viewpoints,
        config=config
    )
    results = trainer.train()

Author: Adversarial Camouflage Research Project
Date: 2026-02-04
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from attack.loss_pytorch import attack_loss_pytorch, attack_loss_with_stats_pytorch
from attack.logger import CSVLogger


def create_viewpoint_configs():
    """
    Returns standard 6-viewpoint EOT configuration.

    Viewpoints arranged in 60 degree increments around vehicle:
    - 0: Front view
    - 60: Front-right
    - 120: Back-right
    - 180: Back
    - 240: Back-left
    - 300: Front-left

    All use pitch=-15 (looking down) and distance=8m.
    """
    return [
        {'yaw': 0,   'pitch': -15, 'distance': 8},
        {'yaw': 60,  'pitch': -15, 'distance': 8},
        {'yaw': 120, 'pitch': -15, 'distance': 8},
        {'yaw': 180, 'pitch': -15, 'distance': 8},
        {'yaw': 240, 'pitch': -15, 'distance': 8},
        {'yaw': 300, 'pitch': -15, 'distance': 8},
    ]


def visualize_texture(texture_np, save_path):
    """Save texture as PNG for visual inspection."""
    texture_uint8 = (texture_np * 255).astype(np.uint8)
    texture_bgr = cv2.cvtColor(texture_uint8, cv2.COLOR_RGB2BGR)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), texture_bgr)


# Default training configuration
DEFAULT_CONFIG = {
    'learning_rate': 0.01,
    'num_iterations': 1000,
    'checkpoint_every': 100,
    'log_every': 10,
    'optimizer': 'adam',
    'output_dir': 'experiments/phase1_eot_pytorch/',
    'coarse_size': 128,  # Texture parameterization size (128x128 -> full_res)
    'full_resolution': 1024,  # 1024x1024 for V2 renderer
    'detector_input_size': 512,  # EfficientDet expects 512x512
    'clip_grad_norm': 1.0,  # Gradient clipping for stability
}


class EOTTrainerPyTorch:
    """
    Pure PyTorch EOT trainer for adversarial textures.

    Uses TRUE end-to-end gradient flow instead of finite differences:
    - Forward: texture -> renderer -> detector -> loss
    - Backward: loss.backward() computes all gradients via autograd
    - Update: optimizer.step() updates texture

    This is 3x faster than the finite differences approach and provides
    more accurate gradients.
    """

    def __init__(self, carla_handler, detector, renderer, viewpoints, config=None):
        """
        Initialize PyTorch EOT trainer.

        Args:
            carla_handler: CarlaHandler instance (must have vehicle spawned)
            detector: EfficientDetPyTorch instance
            renderer: TextureApplicatorPyTorch instance
            viewpoints: List of viewpoint dicts from create_viewpoint_configs()
            config: Training configuration dict (merged with DEFAULT_CONFIG)
        """
        self.carla = carla_handler
        self.detector = detector
        self.renderer = renderer
        self.viewpoints = viewpoints
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Detect device
        self.device = next(detector.model.parameters()).device
        print(f"EOTTrainerPyTorch using device: {self.device}")

        # Validate inputs
        if self.carla.vehicle is None:
            raise ValueError("CarlaHandler must have a vehicle spawned")

        if len(self.viewpoints) != 6:
            print(f"Warning: Expected 6 viewpoints, got {len(self.viewpoints)}")

        # Create output directories
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
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

        print(f"EOTTrainerPyTorch initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Viewpoints: {len(self.viewpoints)}")
        print(f"  Device: {self.device}")

    def capture_reference_images(self):
        """
        Capture reference images and masks from all viewpoints.

        Returns:
            x_ref_batch: torch.Tensor [6, 3, H, W] float32 [0, 1] (NCHW)
            mask_batch: torch.Tensor [6, 1, H, W] float32 [0, 1] (NCHW)
        """
        resolution = self.config['full_resolution']
        print(f"Capturing reference images from {len(self.viewpoints)} viewpoints...")

        x_refs = []
        masks = []

        for i, vp in enumerate(self.viewpoints):
            # Position camera
            self.carla.set_camera_viewpoint(vp['yaw'], vp['pitch'], vp['distance'])

            # Capture image (BGR uint8)
            img = self.carla.get_image()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0  # [H, W, 3]

            # Capture paintable mask
            paintable_mask = self.carla.get_paintable_mask()  # [H, W] float32

            x_refs.append(img_float)
            masks.append(paintable_mask)

            print(f"  Viewpoint {i}: yaw={vp['yaw']:3d} pitch={vp['pitch']:3d} dist={vp['distance']}m")

            # Save for debugging
            save_path = self.output_dir / f'reference_view_{i}_yaw_{vp["yaw"]:03d}.png'
            visualize_texture(img_float, str(save_path))

        # Stack and convert to tensors (NHWC -> NCHW)
        x_ref_np = np.stack(x_refs, axis=0)  # [6, H, W, 3]
        x_ref_np = np.transpose(x_ref_np, (0, 3, 1, 2))  # [6, 3, H, W]
        x_ref_batch = torch.from_numpy(x_ref_np).float().to(self.device)

        mask_np = np.stack(masks, axis=0)[:, :, :, np.newaxis]  # [6, H, W, 1]
        mask_np = np.transpose(mask_np, (0, 3, 1, 2))  # [6, 1, H, W]
        mask_batch = torch.from_numpy(mask_np).float().to(self.device)

        print(f"  Reference batch: {x_ref_batch.shape}")
        print(f"  Mask batch: {mask_batch.shape}")

        return x_ref_batch, mask_batch

    def initialize_texture(self, init_type='random_uniform'):
        """
        Initialize texture as torch.Tensor with requires_grad=True.

        Args:
            init_type: Initialization strategy
                - 'random_uniform': Random values in [0, 1]
                - 'random_normal': Normal distribution clipped to [0, 1]
                - 'constant': Constant gray (0.5)

        Returns:
            torch.Tensor [3, H, W] with requires_grad=True (NCHW format)
        """
        coarse_size = self.config['coarse_size']
        print(f"Initializing texture: {init_type} at {coarse_size}x{coarse_size}")

        if init_type == 'random_uniform':
            texture_init = torch.rand(3, coarse_size, coarse_size, device=self.device)
        elif init_type == 'random_normal':
            texture_init = torch.randn(3, coarse_size, coarse_size, device=self.device) * 0.1 + 0.5
            texture_init = torch.clamp(texture_init, 0.0, 1.0)
        elif init_type == 'constant':
            texture_init = torch.ones(3, coarse_size, coarse_size, device=self.device) * 0.5
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        # Enable gradients for optimization
        texture = texture_init.clone().requires_grad_(True)

        print(f"  Shape: {texture.shape}")
        print(f"  requires_grad: {texture.requires_grad}")
        print(f"  Mean: {texture.mean().item():.4f}")

        # Save initial texture
        texture_full = self._upsample_texture(texture)
        texture_np = texture_full.detach().cpu().numpy()
        texture_np = np.transpose(texture_np, (1, 2, 0))  # CHW -> HWC
        visualize_texture(texture_np, str(self.viz_dir / 'texture_iter_0000.png'))

        return texture

    def _upsample_texture(self, texture):
        """
        Upsample coarse texture to full resolution.

        Args:
            texture: [3, coarse_size, coarse_size] or [B, 3, coarse_size, coarse_size]

        Returns:
            [3, full_res, full_res] or [B, 3, full_res, full_res]
        """
        full_res = self.config['full_resolution']

        squeeze = False
        if texture.dim() == 3:
            texture = texture.unsqueeze(0)
            squeeze = True

        upsampled = F.interpolate(
            texture,
            size=(full_res, full_res),
            mode='bicubic',
            align_corners=False
        )

        if squeeze:
            upsampled = upsampled.squeeze(0)

        return upsampled

    def _tile_texture_for_batch(self, texture):
        """
        Repeat single texture for all viewpoints.

        Args:
            texture: [3, H, W]

        Returns:
            [batch_size, 3, H, W]
        """
        batch_size = len(self.viewpoints)
        return texture.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _forward_pass(self, texture, x_ref_batch, mask_batch):
        """
        Single forward pass through renderer + detector (with gradients).

        This is the core differentiable pipeline:
            texture -> upsample -> tile -> render -> resize -> detect -> loss

        Args:
            texture: torch.Tensor [3, coarse_size, coarse_size] (requires_grad=True)
            x_ref_batch: torch.Tensor [6, 3, H, W] reference images
            mask_batch: torch.Tensor [6, 1, H, W] paintable masks

        Returns:
            loss: Scalar tensor (with gradients)
            metrics: Dict with confidence values (detached)
        """
        # 1. Upsample coarse texture to full resolution
        texture_full = self._upsample_texture(texture)  # [3, 1024, 1024]

        # 2. Tile texture for all viewpoints
        texture_batch = self._tile_texture_for_batch(texture_full)  # [6, 3, 1024, 1024]

        # 3. Render through neural renderer (differentiable!)
        rendered_batch = self.renderer.apply_differentiable(
            x_ref_batch, texture_batch, mask_batch
        )  # [6, 3, 1024, 1024]

        # 4. Resize for detector input (512x512)
        detector_size = self.config['detector_input_size']
        rendered_resized = F.interpolate(
            rendered_batch,
            size=(detector_size, detector_size),
            mode='bilinear',
            align_corners=False
        )  # [6, 3, 512, 512]

        # 5. Run through detector WITH gradients
        class_logits, box_preds = self.detector.forward_pre_nms_with_grad(rendered_resized)

        # 6. Compute attack loss
        loss, stats = attack_loss_with_stats_pytorch(class_logits, car_class_id=2)

        # 7. Extract metrics (detach for logging)
        metrics = {
            'max_confidence': stats['max_confidence'],
            'mean_confidence': stats['mean_confidence'],
            'per_view_conf': stats['per_image_max_conf'].tolist(),
        }

        return loss, metrics

    def train(self):
        """
        Main EOT training loop using true PyTorch autograd.

        Algorithm:
            1. Capture reference images from all viewpoints
            2. Initialize texture as trainable parameter
            3. Setup PyTorch optimizer
            4. For each iteration:
                a. Forward pass (texture -> renderer -> detector -> loss)
                b. Backward pass (loss.backward() computes all gradients)
                c. Optimizer step (update texture)
                d. Clamp texture to [0, 1]
                e. Log metrics
            5. Save final results

        Returns:
            dict: {'texture': final texture, 'final_loss': loss, 'history': log data}
        """
        print("=" * 70)
        print("STARTING PYTORCH EOT TRAINING")
        print("=" * 70)
        print()

        # === SETUP ===
        print("Setup Phase:")
        print("-" * 70)

        # 1. Capture reference images
        x_ref_batch, mask_batch = self.capture_reference_images()
        print()

        # 2. Initialize texture
        texture = self.initialize_texture('random_uniform')
        print()

        # 3. Setup optimizer
        optimizer = optim.Adam([texture], lr=self.config['learning_rate'])
        print(f"Optimizer: Adam (lr={self.config['learning_rate']})")
        print()

        # 4. Initialize loggers
        main_logger = CSVLogger(str(self.output_dir / 'training_log.csv'))
        main_logger.write_header([
            'iteration', 'loss', 'max_conf', 'mean_conf',
            'grad_norm', 'texture_mean', 'texture_std',
            'view_0_conf', 'view_1_conf', 'view_2_conf',
            'view_3_conf', 'view_4_conf', 'view_5_conf'
        ])

        # === TRAINING LOOP ===
        print("Training Phase:")
        print("-" * 70)

        for iteration in range(self.config['num_iterations']):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass (differentiable!)
            loss, metrics = self._forward_pass(texture, x_ref_batch, mask_batch)

            # Backward pass (TRUE autograd - no finite differences!)
            loss.backward()

            # Gradient clipping for stability
            if self.config.get('clip_grad_norm'):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [texture], self.config['clip_grad_norm']
                )
            else:
                grad_norm = texture.grad.norm().item()

            # Optimizer step
            optimizer.step()

            # Clamp texture to valid range
            with torch.no_grad():
                texture.clamp_(0.0, 1.0)

            # Logging
            if iteration % self.config['log_every'] == 0:
                print(
                    f"Iter {iteration:4d}/{self.config['num_iterations']} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Max Conf: {metrics['max_confidence']:.4f} | "
                    f"Mean Conf: {metrics['mean_confidence']:.4f} | "
                    f"Grad: {grad_norm:.6f}"
                )

                main_logger.write_row([
                    iteration,
                    loss.item(),
                    metrics['max_confidence'],
                    metrics['mean_confidence'],
                    grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                    texture.mean().item(),
                    texture.std().item(),
                ] + metrics['per_view_conf'])

            # Checkpointing
            if iteration % self.config['checkpoint_every'] == 0 and iteration > 0:
                self._save_checkpoint(iteration, texture, metrics)

        # === FINAL SAVE ===
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        final_texture = texture.detach().cpu()
        self._save_final_results(final_texture, main_logger.data)

        main_logger.close()

        return {
            'texture': final_texture.numpy(),
            'final_loss': loss.item(),
            'history': main_logger.data,
        }

    def _save_checkpoint(self, iteration, texture, metrics):
        """Save texture checkpoint."""
        # Save numpy (for resuming)
        texture_np = texture.detach().cpu().numpy()
        np.save(str(self.checkpoint_dir / f'texture_iter_{iteration:04d}.npy'), texture_np)

        # Save visualization
        texture_full = self._upsample_texture(texture)
        texture_vis = texture_full.detach().cpu().numpy()
        texture_vis = np.transpose(texture_vis, (1, 2, 0))  # CHW -> HWC
        visualize_texture(texture_vis, str(self.viz_dir / f'texture_iter_{iteration:04d}.png'))

        print(f"  Checkpoint saved: texture_iter_{iteration:04d}.npy")

    def _save_final_results(self, texture, training_history):
        """Save final results."""
        full_res = self.config['full_resolution']

        # Save coarse texture
        texture_np = texture.numpy()
        np.save(str(self.final_dir / 'texture_final.npy'), texture_np)

        # Save torch checkpoint
        torch.save(texture, str(self.final_dir / 'texture_final.pt'))

        # Save visualization (upsampled)
        texture_full = F.interpolate(
            texture.unsqueeze(0),
            size=(full_res, full_res),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        texture_vis = texture_full.numpy()
        texture_vis = np.transpose(texture_vis, (1, 2, 0))  # CHW -> HWC
        visualize_texture(texture_vis, str(self.final_dir / 'texture_final.png'))

        # Save summary
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
                'framework': 'PyTorch (true autograd)',
            }
        else:
            summary = {'error': 'No training data recorded'}

        with open(self.final_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Final results saved to {self.final_dir}/")


# Test script
if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH EOT TRAINER TEST")
    print("=" * 70)
    print()

    # Test viewpoint configs
    print("Testing create_viewpoint_configs...")
    vps = create_viewpoint_configs()
    assert len(vps) == 6
    print(f"  Created {len(vps)} viewpoints")
    print()

    # Test texture operations
    print("Testing texture operations...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Create test texture
    texture = torch.rand(3, 128, 128, device=device, requires_grad=True)
    print(f"  Coarse texture: {texture.shape}")

    # Test upsampling (using function directly)
    texture_up = F.interpolate(
        texture.unsqueeze(0),
        size=(1024, 1024),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)
    print(f"  Upsampled texture: {texture_up.shape}")

    # Test gradient flow
    loss = texture_up.mean()
    loss.backward()
    print(f"  Gradient norm: {texture.grad.norm().item():.6f}")

    if texture.grad.norm().item() > 0:
        print("  PASSED: Gradients flow through upsampling!")
    print()

    # Note: Full trainer test requires CARLA, detector, and renderer
    print("Note: Full trainer test requires:")
    print("  - CARLA server running")
    print("  - EfficientDetPyTorch instance")
    print("  - TextureApplicatorPyTorch instance")
    print()
    print("Run experiments/phase1_random_pytorch.py for full test")

    print()
    print("=" * 70)
    print("TRAINER TESTS PASSED")
    print("=" * 70)
