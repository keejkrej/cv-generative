# CycleGAN

CycleGAN enables unpaired image-to-image translation by learning mappings between two domains without paired training data.

## Architecture

### Key Components

1. **Cycle Consistency Loss**
   - Ensures F(G(x)) ≈ x and G(F(y)) ≈ y
   - Preserves content while changing style
   - Prevents mode collapse

2. **Two Generators**
   - G_A2B: Translates from domain A to B
   - G_B2A: Translates from domain B to A
   - ResNet-based architecture

3. **Two Discriminators**
   - D_A: Discriminates real vs fake in domain A
   - D_B: Discriminates real vs fake in domain B
   - PatchGAN architecture (70x70 patches)

4. **Identity Loss (Optional)**
   - Ensures G_A2B(b) ≈ b and G_B2A(a) ≈ a
   - Helps preserve color composition
   - Particularly useful for photo editing

### Generator Architecture

```
Input → Conv → Downsample → ResBlocks → Upsample → Conv → Output
                    ↓                        ↑
                  (64→128→256)        (256→128→64)
```

- Initial 7x7 convolution
- 2 downsampling layers (stride 2)
- 9 ResNet blocks (configurable)
- 2 upsampling layers (transposed conv)
- Final 7x7 convolution to output

### Discriminator Architecture

```
Input → Conv Blocks → Output
         ↓
    (64→128→256→512)
```

- PatchGAN discriminator
- Outputs 30x30 grid of predictions
- Each patch classifies 70x70 region
- Encourages sharp, high-frequency details

## Usage

```python
from models.image_generation.gan.cyclegan import CycleGAN

# Initialize model
model = CycleGAN()

# Training step
outputs = model(real_A, real_B)
losses = model.compute_losses(real_A, real_B, outputs)

# Translate images
with torch.no_grad():
    # Horse to Zebra
    zebra = model.translate(horse_image, direction='A2B')
    
    # Zebra to Horse  
    horse = model.translate(zebra_image, direction='B2A')
```

## Training

### Loss Functions

1. **Adversarial Loss**
   ```python
   L_GAN = E[log D_B(b)] + E[log(1 - D_B(G_A2B(a)))]
   ```

2. **Cycle Consistency Loss**
   ```python
   L_cycle = E[||G_B2A(G_A2B(a)) - a||₁] + E[||G_A2B(G_B2A(b)) - b||₁]
   ```

3. **Identity Loss (Optional)**
   ```python
   L_identity = E[||G_B2A(a) - a||₁] + E[||G_A2B(b) - b||₁]
   ```

4. **Total Loss**
   ```python
   L = L_GAN + λ_cycle * L_cycle + λ_identity * L_identity
   ```

### Training Configuration

```python
config = CycleGANConfig(
    ngf=64,           # Generator filters
    ndf=64,           # Discriminator filters
    n_blocks=9,       # ResNet blocks
    lr=0.0002,        # Learning rate
    lambda_cycle=10.0,     # Cycle loss weight
    lambda_identity=0.5,   # Identity loss weight
    pool_size=50      # Image pool size
)

# Optimizers
opt_G, opt_D_A, opt_D_B = model.get_optimizers()

# Learning rate scheduling
# Linear decay to 0 over last 100 epochs
def lr_lambda(epoch):
    return 1.0 - max(0, epoch - 100) / 100
```

### Training Tips

1. **Image Pool**
   - Store 50 previously generated images
   - Sample from pool for discriminator training
   - Reduces model oscillation

2. **Learning Rate Schedule**
   - Keep constant for first half of training
   - Linear decay to 0 for second half
   - Helps fine-tune results

3. **Data Augmentation**
   - Random horizontal flips
   - Random crops and resizes
   - Improves generalization

## Applications

### Style Transfer
- Photo ↔ Painting (Monet, Van Gogh, Ukiyo-e)
- Summer ↔ Winter scenes
- Day ↔ Night conversion

### Object Transfiguration
- Horse ↔ Zebra
- Apple ↔ Orange
- Cat ↔ Dog

### Domain Adaptation
- Synthetic ↔ Real data
- Sim2Real for robotics
- Medical imaging modalities

### Photo Enhancement
- Photo ↔ DSLR quality
- Depth of field effects
- Super-resolution

## Performance

### Typical Results (256x256)
- **Training Time**: 1-2 days (200 epochs)
- **Inference Speed**: 30 FPS on V100
- **Memory**: 8GB VRAM for training

### Quality Metrics
- **FID**: Varies by task (20-80 typical)
- **IS**: Not well-suited for CycleGAN
- **User Studies**: Often preferred over baselines

## Advantages and Limitations

### Advantages
- No paired training data required
- Bidirectional translation
- Preserves content structure
- Works on diverse domains

### Limitations
- Can struggle with geometric changes
- Sometimes changes unintended aspects
- Requires sufficient domain variance
- Training can be unstable

## Implementation Notes

1. **Instance Normalization**
   - Better than batch norm for style transfer
   - Normalizes each sample independently
   - Prevents batch artifacts

2. **Reflection Padding**
   - Better than zero padding for images
   - Reduces border artifacts
   - Used in generators

3. **Least Squares GAN Loss**
   - More stable than vanilla GAN loss
   - Better gradients
   - Reduces mode collapse

## Common Issues and Solutions

1. **Mode Collapse**
   - Increase cycle loss weight
   - Use identity loss
   - Check discriminator isn't too strong

2. **Color Changes**
   - Add identity loss
   - Reduce generator capacity
   - Check data preprocessing

3. **Blurry Results**
   - Reduce cycle loss weight
   - Increase discriminator capacity
   - Use PatchGAN discriminator

## References

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [Official PyTorch Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Project Page](https://junyanz.github.io/CycleGAN/)