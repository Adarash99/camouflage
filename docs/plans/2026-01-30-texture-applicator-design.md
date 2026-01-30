# Texture Applicator Design

**Date:** 2026-01-30
**Purpose:** Reusable component for applying adversarial textures to vehicle reference images using the trained neural renderer
**Target:** Phase 1 attack pipeline integration

---

## Overview

The `TextureApplicator` class provides a clean, object-oriented interface for applying textures to vehicle reference images. It encapsulates the neural renderer model and handles batch dimension management automatically, making it easy to integrate into the adversarial attack pipeline.

## Requirements

- **Input format:** NumPy arrays, float32, normalized [0, 1]
- **Dimensions:** (500, 500, 3) for single images or (batch, 500, 500, 3) for batches
- **Performance:** Model loaded once during initialization for efficient repeated use
- **Flexibility:** Auto-handle both single and batched inputs seamlessly

## Class Design

### TextureApplicator

```python
class TextureApplicator:
    """
    Applies adversarial textures to vehicle reference images using a trained neural renderer.

    The renderer is loaded once during initialization and kept in memory for efficient
    repeated application during the attack optimization loop.
    """
```

### Initialization

```python
def __init__(self, model_path='models/k3_100epch_wo_custom_loss_model.h5'):
    """
    Initialize the texture applicator by loading the neural renderer model.

    Args:
        model_path: Path to the trained renderer .h5 file

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
```

**Implementation:**
1. Suppress TensorFlow warnings (`TF_CPP_MIN_LOG_LEVEL=2`)
2. Validate model path exists
3. Load Keras model with `compile=False`
4. Store model as instance variable
5. Print confirmation message

### Apply Method

```python
def apply(self, x_ref, texture):
    """
    Apply texture to reference image using the neural renderer.

    Args:
        x_ref: Reference image (vehicle with neutral color)
               Shape: (500, 500, 3) or (batch, 500, 500, 3)
               Type: float32, range [0, 1]

        texture: Texture pattern to apply (eta_exp in renderer terms)
                 Shape: (500, 500, 3) or (batch, 500, 500, 3)
                 Type: float32, range [0, 1]

    Returns:
        rendered: Vehicle image with applied texture
                  Shape: Same as inputs (preserves batch dimension)
                  Type: float32, range [0, 1]

    Raises:
        ValueError: If shapes don't match or are incorrect
    """
```

**Implementation:**
1. Detect if inputs are 3D (single) or 4D (batch)
2. If single, expand: `(500, 500, 3)` → `(1, 500, 500, 3)`
3. Validate shapes and types
4. Call renderer: `rendered = self.model([x_ref, texture])`
5. If input was single, squeeze to 3D
6. Return rendered image

### Validation

**Shape validation:**
- Verify both inputs have matching batch dimensions
- Check spatial dimensions are exactly 500x500
- Confirm 3 color channels
- Raise `ValueError` with descriptive message

**Type and range validation:**
- Verify float32 (auto-convert from float64 if needed)
- Warn if values outside [0, 1] range
- Don't auto-clip - force user to fix normalization

**Example error messages:**
- "Expected shape (500, 500, 3) or (batch, 500, 500, 3), got {shape}"
- "Reference and texture must have matching batch dimensions"
- "Values outside [0, 1] range detected - did you mean to normalize?"

### Optional Helper

```python
def get_model_info(self):
    """Returns model input/output shapes and names for debugging."""
```

## Module Structure

**File:** `texture_applicator.py`

**Dependencies:**
- TensorFlow/Keras
- NumPy

**Imports:**
```python
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
```

## Usage Examples

### Single Image

```python
from texture_applicator import TextureApplicator

# Initialize once
applicator = TextureApplicator()

# Apply texture
x_ref = load_reference_image()      # (500, 500, 3), float32, [0, 1]
texture = generate_texture()         # (500, 500, 3), float32, [0, 1]
rendered = applicator.apply(x_ref, texture)  # (500, 500, 3)
```

### Batch Processing (EOT)

```python
# Load 6 viewpoints
x_ref_batch = load_viewpoints()      # (6, 500, 500, 3)
texture_batch = tile_texture(tex, 6) # (6, 500, 500, 3)

# Apply in one call
rendered_batch = applicator.apply(x_ref_batch, texture_batch)  # (6, 500, 500, 3)
```

### Attack Pipeline Integration

```python
applicator = TextureApplicator()
detector = load_detector()

for step in range(num_optimization_steps):
    # Apply current texture
    rendered = applicator.apply(reference, current_texture)

    # Compute attack loss
    detections = detector(rendered)
    loss = compute_attack_loss(detections)

    # Update texture via gradients
    current_texture = optimizer.step(loss)
```

## Design Decisions

### Why OOP?
- Encapsulates model loading/state management
- Easy to import and reuse across pipeline scripts
- Natural fit for stateful ML components

### Why auto-batch handling?
- Single images common during debugging/visualization
- Batches required for EOT training
- Auto-detection eliminates boilerplate

### Why [0, 1] float32?
- Matches TensorFlow/gradient computation conventions
- Consistent with verify_renderer_differentiability.py
- Natural format for adversarial optimization

### Why no file I/O?
- Pipeline operates on in-memory arrays
- File I/O would add unnecessary overhead
- Keeps class focused on single responsibility

## Testing Strategy

Manual verification:
1. Load model successfully
2. Apply to single image → correct shape
3. Apply to batch → correct shape
4. Validate error handling (wrong shapes, types, ranges)
5. Verify output is differentiable (gradients flow)

## Integration Path

1. Create `texture_applicator.py`
2. Test with sample data from dataset
3. Import into attack pipeline script
4. Use in optimization loop with detector

---

**Status:** Design approved, ready for implementation
**Next:** Implement class and test with existing dataset
