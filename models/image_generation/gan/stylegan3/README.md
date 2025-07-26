# StyleGAN3

StyleGAN3 introduces alias-free generator architecture that produces high-quality images with equivariance to translation and rotation.

## Architecture

### Key Innovations

1. **Alias-Free Layers**
   - Continuous signal interpretation
   - Careful low-pass filtering
   - Equivariance to sub-pixel translation and rotation

2. **Flexible Layer Specifications**
   - Support for different kernel sizes (1x1 or 3x3)
   - Radial and non-radial filter designs
   - Configurable upsampling filters

3. **Improved Training Dynamics**
   - Better FID scores
   - More stable training
   - Reduced texture sticking artifacts

### Generator Architecture

```
Z → Mapping Network → W → Synthesis Network → Image
                           ↓
                    Style Modulation
```

#### Mapping Network
- 8 fully connected layers
- Maps Z to W intermediate latent space
- Learning rate multiplier for stability
- Moving average of W for truncation

#### Synthesis Network
- Progressive resolution: 4x4 → 1024x1024
- Modulated convolutions with demodulation
- Noise injection for stochastic variation
- Skip connections and ToRGB layers

### Training Improvements

1. **Equivariance Regularization**
   - Translation equivariance
   - Rotation equivariance (optional)

2. **Adaptive Discriminator Augmentation (ADA)**
   - Dynamic augmentation probability
   - Prevents discriminator overfitting

## Usage

```python
from models.image_generation.gan.stylegan3 import StyleGAN3

# Initialize model
model = StyleGAN3()

# Generate images
images = model.generate(
    batch_size=4,
    truncation_psi=0.7,  # Controls diversity vs quality
    noise_mode='const'   # 'const', 'random', or 'none'
)

# Generate from specific latent
z = torch.randn(1, 512)
image = model.generate(z=z)

# Class-conditional generation (if trained conditionally)
class_label = torch.tensor([5])  # Class index
image = model.generate(c=class_label)
```

## Training Configuration

### Recommended Settings

```python
# Generator
g_kwargs = {
    'z_dim': 512,
    'w_dim': 512,
    'mapping_layers': 8,
    'channel_base': 32768,
    'channel_max': 512,
    'magnitude_ema_beta': 0.999
}

# Discriminator  
d_kwargs = {
    'channel_base': 32768,
    'channel_max': 512,
    'mbstd_group_size': 4,
    'epilogue_kwargs': {'mbstd_group_size': 4}
}

# Training
batch_size = 32
learning_rate = 0.0025
r1_gamma = 10.0
ema_kimg = 10.0
```

### Loss Functions

1. **Generator Loss**
   ```python
   L_G = -E[log(D(G(z)))]
   ```

2. **Discriminator Loss**
   ```python
   L_D = -E[log(D(x))] - E[log(1 - D(G(z)))] + (r1_gamma/2) * E[||∇D(x)||²]
   ```

## Performance

### Quality Metrics (AFHQv2 dataset)
- **FID**: 2.79
- **KID**: 0.0014
- **Precision**: 0.89
- **Recall**: 0.52

### Speed (V100 GPU)
- **Training**: ~70 kimg/day
- **Inference**: 30 ms/image (1024x1024)

### Memory Requirements
- **Training**: 16GB VRAM (batch size 32)
- **Inference**: 4GB VRAM

## Comparison with StyleGAN2

| Feature | StyleGAN2 | StyleGAN3 |
|---------|-----------|-----------|
| Translation Equivariance | ❌ | ✅ |
| Rotation Equivariance | ❌ | ✅ (optional) |
| Texture Sticking | Common | Rare |
| Training Stability | Good | Better |
| FID Score | ~2.84 | ~2.79 |

## Implementation Notes

1. **Alias-Free Operations**
   - All upsampling uses careful filtering
   - Convolutions designed for continuous signals
   - Radial filters for rotation equivariance

2. **Numerical Precision**
   - FP16 training supported for speed
   - FP32 for critical operations
   - Gradient scaling for mixed precision

3. **Memory Optimization**
   - Gradient checkpointing available
   - Efficient grouped convolutions
   - Optional CPU offloading

## References

- [Alias-Free GAN Paper](https://arxiv.org/abs/2106.12423)
- [Official Implementation](https://github.com/NVlabs/stylegan3)
- [StyleGAN3 Training Guide](https://github.com/NVlabs/stylegan3/blob/main/docs/train.md)