# Stable Diffusion Models

This directory contains implementations of Stable Diffusion models including SDXL and SD3.

## Stable Diffusion XL (SDXL)

SDXL is an advanced latent diffusion model that generates high-quality 1024x1024 images with improved prompt adherence and image quality compared to previous versions.

### Architecture

- **Base Resolution**: 1024x1024 (native)
- **Latent Space**: 4-channel latent representation (128x128 for 1024x1024 images)
- **U-Net Architecture**: Enhanced with dual text encoders and conditioning
- **Conditioning**: 
  - Dual text encoders (CLIP ViT-L and OpenCLIP ViT-G)
  - Additional conditioning for aspect ratio and crop coordinates
- **Two-Stage Pipeline**: Base model + optional refiner model

### Key Features

1. **Improved Prompt Adherence**: Better understanding of complex prompts
2. **Higher Resolution**: Native 1024x1024 generation
3. **Better Hands/Faces**: Improved anatomical accuracy
4. **Style Control**: Better control over artistic styles
5. **Conditioning Augmentation**: Size and crop conditioning for better composition

### Usage

```python
from models.image_generation.stable_diffusion import SDXL

# Initialize model
model = SDXL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Generate image
image = model.generate(
    prompt="A majestic lion in a golden savanna at sunset",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=1024,
    width=1024
)
```

### Training Details

- **Dataset**: LAION-5B subset with aesthetic filtering
- **Training Steps**: 1M steps
- **Batch Size**: 2048
- **Learning Rate**: 1e-4 with cosine schedule
- **Hardware**: 96 A100 GPUs

### Performance

- **Inference Speed**: ~5 seconds on A100 (50 steps)
- **Memory Requirements**: 
  - FP32: 13.5GB VRAM
  - FP16: 8.5GB VRAM
  - INT8: 4.5GB VRAM
- **Quality Metrics**:
  - FID: 11.7 (COCO-2017)
  - CLIP Score: 0.82

## Stable Diffusion 3 (SD3)

SD3 represents a major architectural shift from U-Net to Multimodal Diffusion Transformer (MM-DiT).

### Architecture

- **Transformer-Based**: Rectified Flow Transformer instead of U-Net
- **Multimodal Design**: Joint image-text representation learning
- **Flow Matching**: Rectified flow formulation for improved sampling
- **Flexible Resolution**: Supports multiple aspect ratios natively

### Key Improvements

1. **Better Text Rendering**: Can generate readable text in images
2. **Improved Prompt Following**: On par with DALL-E 3
3. **Faster Sampling**: Rectified flow allows fewer sampling steps
4. **Better Composition**: Improved spatial reasoning

### Implementation Status

- [ ] SD3 base implementation
- [ ] Rectified flow sampler
- [ ] T5 text encoder integration
- [x] SDXL implementation
- [ ] VAE implementations
- [ ] Training scripts

### References

- [SDXL Paper](https://arxiv.org/abs/2307.01952)
- [SD3 Paper](https://arxiv.org/abs/2403.03206)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)