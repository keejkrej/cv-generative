# 3D Gaussian Splatting

This directory contains the implementation of 3D Gaussian Splatting, a state-of-the-art method for real-time neural rendering that represents scenes as collections of 3D Gaussians.

## Overview

3D Gaussian Splatting achieves photorealistic real-time rendering without neural networks by:
- Representing scenes as anisotropic 3D Gaussians
- Using differentiable rasterization for optimization
- Adaptive density control during training
- Tile-based rendering for GPU efficiency

## Architecture

### Key Components

1. **3D Gaussian Representation**
   - Position (x, y, z)
   - Anisotropic covariance (3x3 matrix)
   - Opacity (α)
   - Spherical Harmonics (SH) for view-dependent color

2. **Optimization Parameters**
   - Position: Directly optimized
   - Covariance: Factorized as scaling (s) and rotation (r)
   - Appearance: SH coefficients (DC + higher order)

3. **Adaptive Control**
   - Densification: Split and clone based on gradients
   - Pruning: Remove low-opacity and large Gaussians

### Rendering Pipeline

1. **Project 3D Gaussians to 2D**
   ```
   Σ' = J W Σ W^T J^T
   ```
   Where J is the Jacobian of the projection

2. **Sort by Depth**
   - Per-tile sorting for efficiency

3. **α-blending**
   ```
   C = Σ c_i α_i ∏(1 - α_j)
   ```

## Usage

```python
from models.3d_generation.gaussian_splatting import GaussianSplatting

# Initialize model
model = GaussianSplatting()

# Reconstruct from multi-view images
scene = model.reconstruct(
    images=multi_view_images,
    camera_poses=camera_poses,
    num_iterations=30000
)

# Render novel view
viewpoint = {
    "viewmatrix": view_matrix,
    "projmatrix": proj_matrix,
    "tanfovx": tan_fov_x,
    "tanfovy": tan_fov_y,
    "image_width": 1920,
    "image_height": 1080
}

output = model(viewpoint)
rendered_image = output["render"]
```

## Training Details

### Optimization
- **Optimizer**: Adam with separate learning rates
- **Position LR**: 0.00016 → 0.0000016 (exponential decay)
- **Other parameters**: Fixed learning rates
- **Iterations**: 30,000 typical

### Loss Function
```python
L = (1 - λ) * L1 + λ * (1 - SSIM)
```
Where λ = 0.2

### Adaptive Control Schedule
- **Densification**: Every 100 iterations (500-15,000)
- **Opacity reset**: Every 3,000 iterations
- **Gradient threshold**: 0.0002

## Performance

### Speed (RTX 4090)
- **1080p**: 200+ FPS
- **4K**: 100+ FPS
- **Training**: ~5-30 minutes

### Quality Metrics
- **PSNR**: 30+ dB (typical)
- **SSIM**: 0.95+
- **LPIPS**: < 0.05

### Memory Usage
- **Scene storage**: 50-500 MB (scene dependent)
- **Training VRAM**: 12-24 GB
- **Inference VRAM**: 2-4 GB

## Advantages over NeRF

1. **Real-time Rendering**: 100-1000x faster
2. **Explicit Representation**: Editable, interpretable
3. **Training Speed**: 10-100x faster convergence
4. **No Neural Networks**: Pure rasterization

## Implementation Notes

### CUDA Kernels Required
The full implementation requires custom CUDA kernels for:
- Forward rasterization
- Backward pass
- Preprocessing (frustum culling, sorting)

### Simplifications in This Code
This implementation provides the core logic but uses simplified:
- Rasterization (placeholder for CUDA kernel)
- Sorting (not tile-based)
- Gradient computation

For production use, integrate with the official CUDA implementation.

## References

- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Official Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [Awesome 3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)