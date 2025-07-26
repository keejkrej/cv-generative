# BigGAN

BigGAN is a large-scale GAN that achieves high-fidelity, diverse image synthesis on ImageNet at high resolutions.

## Architecture

### Key Components

1. **Class-Conditional Batch Normalization**
   - Conditions batch norm on class embeddings
   - Allows class-specific feature modulation
   - Improves class-conditional generation quality

2. **Self-Attention Mechanism**
   - Applied at 64x64 resolution (configurable)
   - Captures long-range dependencies
   - Improves geometric consistency

3. **Spectral Normalization**
   - Applied to all layers in G and D
   - Stabilizes training dynamics
   - Prevents mode collapse

4. **Orthogonal Initialization**
   - Initializes weights orthogonally
   - Improves gradient flow
   - Faster convergence

### Generator Architecture

```
Z + Class Embedding → Linear → Reshape → GBlocks → Output
                                    ↓
                            Self-Attention (64x64)
```

#### GBlock Structure
- Conditional BatchNorm → ReLU → 3x3 Conv
- Conditional BatchNorm → ReLU → 3x3 Conv
- Skip connection with 1x1 Conv
- Optional upsampling

### Discriminator Architecture

```
Image → Conv → DBlocks → Global Pool → Linear → Score
           ↓                              ↓
    Self-Attention               Class Projection
```

#### DBlock Structure
- ReLU → 3x3 Conv
- ReLU → 3x3 Conv → Optional Downsampling
- Skip connection with 1x1 Conv

## Usage

```python
from models.image_generation.gan.biggan import BigGAN

# Initialize model
model = BigGAN()

# Generate class-conditional images
class_idx = 207  # Golden retriever
images = model.generate(
    y=torch.tensor([class_idx] * 4),
    truncation=0.5  # Lower = higher quality, less diversity
)

# Generate with specific latent
z = torch.randn(1, 128)
y = torch.tensor([100])  # Class index
image = model.generate(z=z, y=y)
```

## Training

### Loss Functions

1. **Generator Loss**
   ```python
   L_G = -E[D(G(z, y), y)] + λ * L_ortho
   ```
   Where L_ortho is orthogonal regularization

2. **Discriminator Loss**
   ```python
   L_D = -E[D(x, y)] + E[D(G(z, y), y)] + λ * R1_penalty
   ```

### Training Configuration

```python
# Recommended settings
config = BigGANConfig(
    resolution=512,
    z_dim=128,
    num_classes=1000,
    g_ch=96,
    d_ch=96,
    g_lr=0.00005,
    d_lr=0.0002,
    g_beta1=0.0,
    g_beta2=0.999,
    d_beta1=0.0,
    d_beta2=0.999
)

# Training parameters
batch_size = 256  # Large batch crucial for BigGAN
n_critic = 1      # Train D and G equally
ema_decay = 0.9999
```

### Techniques for Large-Scale Training

1. **Large Batch Sizes**
   - 256-2048 samples per batch
   - Improves sample diversity
   - Requires multiple GPUs

2. **Moving Average Generator**
   - Maintains EMA of generator weights
   - Used for evaluation
   - Improves sample quality

3. **Truncation Trick**
   - Sample z from truncated normal
   - Trade-off: quality vs diversity
   - Typical values: 0.5 - 1.0

4. **Orthogonal Regularization**
   - Encourages orthogonal weight matrices
   - Improves conditioning
   - Prevents mode collapse

## Performance

### ImageNet 128x128
- **IS**: 166.5
- **FID**: 7.4
- **Precision**: 0.87
- **Recall**: 0.28

### ImageNet 512x512
- **IS**: 241.6
- **FID**: 9.6
- **Precision**: 0.89
- **Recall**: 0.31

### Speed (V100 GPU)
- **Training**: ~1 week (512x512, 1M iterations)
- **Inference**: 50ms per image

### Memory Requirements
- **128x128**: 8GB VRAM (batch size 24)
- **256x256**: 16GB VRAM (batch size 12)
- **512x512**: 32GB VRAM (batch size 6)

## Key Insights

1. **Scaling Benefits GANs**
   - Larger models → better quality
   - Larger batches → more stable
   - More parameters → more modes

2. **Class Conditioning Matters**
   - Shared embeddings help
   - Projection discriminator improves results
   - Class balance important

3. **Architectural Choices**
   - Skip connections crucial
   - Self-attention helps coherence
   - Spectral norm prevents collapse

## Common Issues and Solutions

1. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Add gradient clipping

2. **Mode Collapse**
   - Check orthogonal regularization
   - Increase batch size
   - Reduce truncation

3. **Poor Class Separation**
   - Increase shared embedding dim
   - Balance class distribution
   - Check label smoothing

## References

- [BigGAN Paper](https://arxiv.org/abs/1809.11096)
- [Official TensorFlow Implementation](https://github.com/ajbrock/BigGAN-PyTorch)
- [Training Tips](https://github.com/ajbrock/BigGAN-PyTorch/blob/master/README.md)