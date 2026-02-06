#!/usr/bin/env python3
"""
PyTorch Neural Renderer Training Script

Trains the mask-aware neural renderer using MSE loss.

Usage:
    python models/train_renderer.py --dataset dataset/ --epochs 100

    # With validation dataset
    python models/train_renderer.py --dataset dataset/ --val-dataset validation_dataset/

    # Resume from checkpoint
    python models/train_renderer.py --dataset dataset/ --resume checkpoints/epoch_50.pt

Author: Adversarial Camouflage Research Project
Date: 2026-02-04
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from models.renderer_pytorch import MaskAwareRenderer
from models.renderer_dataset import RendererDataset, create_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch with optional AMP."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    use_amp = scaler is not None

    for batch_idx, (x_combined, y_target) in enumerate(train_loader):
        x_combined = x_combined.to(device)
        y_target = y_target.to(device)

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision forward pass
            with autocast():
                output = model(x_combined)
                loss = criterion(output, y_target)
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            output = model(x_combined)
            loss = criterion(output, y_target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Progress logging
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.6f}")

    return total_loss / num_batches


def validate(model, val_loader, criterion, device, use_amp=False):
    """Validate the model with optional AMP."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_combined, y_target in val_loader:
            x_combined = x_combined.to(device)
            y_target = y_target.to(device)

            if use_amp:
                with autocast():
                    output = model(x_combined)
                    loss = criterion(output, y_target)
            else:
                output = model(x_combined)
                loss = criterion(output, y_target)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, scaler=None):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def train_renderer(
    dataset_path,
    output_dir='models/trained/',
    epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    val_split=0.1,
    val_dataset_path=None,
    resume_path=None,
    num_workers=4,
    device=None,
    use_amp=True
):
    """
    Train the mask-aware neural renderer.

    Args:
        dataset_path: Path to training dataset
        output_dir: Directory to save model and logs
        epochs: Number of training epochs
        batch_size: Training batch size (8 recommended for 1024x1024 on 24GB GPU)
        learning_rate: Initial Adam learning rate
        val_split: Fraction of data for validation (if val_dataset_path not provided)
        val_dataset_path: Optional separate validation dataset
        resume_path: Path to checkpoint to resume from
        num_workers: Number of data loading workers
        device: Training device ('cuda' or 'cpu')
        use_amp: Use automatic mixed precision (fp16) for ~50% memory savings

    Returns:
        Trained model
    """
    print("=" * 70)
    print("MASK-AWARE NEURAL RENDERER TRAINING (PyTorch)")
    print("=" * 70)
    print()

    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Create data loaders
    print("Loading datasets...")
    if val_dataset_path:
        # Separate validation dataset
        train_dataset = RendererDataset(dataset_path)
        val_dataset = RendererDataset(val_dataset_path)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
    else:
        # Split training data
        train_loader, val_loader = create_data_loaders(
            dataset_path, batch_size=batch_size, val_split=val_split,
            num_workers=num_workers
        )
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    print()

    # Create model
    print("Creating model...")
    model = MaskAwareRenderer()
    model.to(device)
    info = model.get_model_info()
    print(f"  Total params: {info['total_params']:,}")
    print()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
    )

    # AMP scaler for mixed precision training
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    if scaler:
        print(f"Mixed Precision: Enabled (fp16)")
    else:
        print(f"Mixed Precision: Disabled")
    print()

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_path:
        print(f"Resuming from: {resume_path}")
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, resume_path, device, scaler
        )
        start_epoch += 1  # Start from next epoch
        print(f"  Resumed at epoch {start_epoch}, best loss: {best_val_loss:.6f}")
        print()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }

    # Save config
    config = {
        'dataset_path': str(dataset_path),
        'val_dataset_path': str(val_dataset_path) if val_dataset_path else None,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_split': val_split,
        'device': str(device),
        'use_amp': scaler is not None,
        'start_time': datetime.now().isoformat(),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("Starting training...")
    print("-" * 70)

    training_start_time = datetime.now()

    for epoch in range(start_epoch, epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, use_amp=(scaler is not None))

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  -> New best model saved (val_loss: {val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                checkpoint_dir / f'epoch_{epoch+1:03d}.pt', scaler
            )

    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')

    # Calculate training time
    training_end_time = datetime.now()
    total_time = training_end_time - training_start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Save training history
    history['total_training_time_seconds'] = total_time.total_seconds()
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best validation MSE: {best_val_loss:.6f}")
    print(f"  Final model: {output_dir / 'final_model.pt'}")
    print(f"  Best model: {output_dir / 'best_model.pt'}")
    print(f"  Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 70)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train PyTorch Neural Renderer')

    parser.add_argument('--dataset', type=str, default='dataset/',
                        help='Path to training dataset')
    parser.add_argument('--val-dataset', type=str, default=None,
                        help='Path to validation dataset (optional)')
    parser.add_argument('--output', type=str, default='models/trained/',
                        help='Output directory for model and logs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (8-10 with AMP on 24GB GPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: True)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test with 3 epochs')

    args = parser.parse_args()

    # Quick test mode
    if args.test:
        print("Running quick test (3 epochs)...")
        args.epochs = 3
        args.batch_size = 2

    # Determine AMP setting
    use_amp = args.amp and not args.no_amp

    train_renderer(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        val_dataset_path=args.val_dataset,
        resume_path=args.resume,
        num_workers=args.workers,
        device=args.device,
        use_amp=use_amp
    )


if __name__ == '__main__':
    main()
