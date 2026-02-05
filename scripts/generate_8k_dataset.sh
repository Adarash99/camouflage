#!/bin/bash
# Generate 8000-image dataset with 80/20 train/val split
# Requires CARLA server to be running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Generating 8000-image dataset ==="
echo "Output: dataset_8k/ (6400 train, 1600 val)"
echo ""

# Train set: 6400 images
echo ">>> Generating training set (6400 images)..."
python "$PROJECT_DIR/car_segmentation.py" \
    --output-dir "$PROJECT_DIR/dataset_8k/train" \
    --num-samples 6400 \
    --start-index 0 \
    --resume

echo ""
echo ">>> Training set complete!"
echo ""

# Val set: 1600 images
echo ">>> Generating validation set (1600 images)..."
python "$PROJECT_DIR/car_segmentation.py" \
    --output-dir "$PROJECT_DIR/dataset_8k/val" \
    --num-samples 1600 \
    --start-index 0 \
    --resume

echo ""
echo "=== Dataset generation complete ==="
echo ""
echo "To verify the dataset:"
echo "  python scripts/verify_dataset.py dataset_8k/train 6400"
echo "  python scripts/verify_dataset.py dataset_8k/val 1600"
