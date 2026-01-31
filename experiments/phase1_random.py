#!/usr/bin/env python3
"""
Phase 1: Random Patch EOT Attack

Trains adversarial texture using random initialization to beat DTA baseline.

Usage:
    python experiments/phase1_random.py

Requirements:
    - CARLA server running (./CarlaUE4.sh)
    - conda environment activated (camo or PROJECT)
    - Neural renderer model available

Expected Output:
    experiments/phase1_eot/
    ├── training_log.csv
    ├── per_viewpoint_analysis.csv
    ├── checkpoints/
    ├── visualizations/
    └── final/
        ├── texture_final.npy
        ├── texture_final.png
        └── summary.json
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CarlaHandler import CarlaHandler
from attack.detector_pytorch import EfficientDetPyTorch
from texture_applicator import TextureApplicator
from attack.eot_trainer import EOTTrainer, create_viewpoint_configs


def main():
    """Main entry point for Phase 1 EOT training."""

    print("=" * 70)
    print("PHASE 1: RANDOM PATCH EOT ATTACK")
    print("=" * 70)
    print()

    # === INITIALIZATION ===
    print("Initializing components...")
    print("-" * 70)

    # 1. Initialize CARLA
    print("1. Connecting to CARLA...")
    carla = CarlaHandler(town='Town03', x_res=500, y_res=500)
    carla.spawn_vehicle('vehicle.tesla.model3')
    print("   ✓ CARLA connected, vehicle spawned")
    print()

    # 2. Initialize detector
    print("2. Loading EfficientDet-D0...")
    detector = EfficientDetPyTorch()
    print("   ✓ Detector loaded")
    print()

    # 3. Initialize renderer
    print("3. Loading neural renderer...")
    renderer = TextureApplicator()
    print("   ✓ Renderer loaded")
    print()

    # 4. Configure viewpoints
    print("4. Configuring viewpoints...")
    viewpoints = create_viewpoint_configs()
    print(f"   ✓ {len(viewpoints)} viewpoints configured")
    print()

    # === TRAINING CONFIGURATION ===
    print("Training Configuration:")
    print("-" * 70)

    config = {
        'learning_rate': 0.01,
        'num_iterations': 1000,
        'checkpoint_every': 100,
        'log_every': 10,
        'epsilon': 1e-4,
        'output_dir': 'experiments/phase1_eot/',
    }

    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # === TRAINING ===
    print("Starting training...")
    print("-" * 70)
    print()

    trainer = EOTTrainer(carla, detector, renderer, viewpoints, config)
    results = trainer.train()

    # === RESULTS ===
    print()
    print("=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Iterations completed: {len(results['history'])}")
    print(f"Texture saved to: {config['output_dir']}/final/texture_final.npy")
    print()
    print("Next steps:")
    print("  1. Inspect visualizations in experiments/phase1_eot/visualizations/")
    print("  2. Analyze training_log.csv for convergence")
    print("  3. Evaluate texture against DTA baseline")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
