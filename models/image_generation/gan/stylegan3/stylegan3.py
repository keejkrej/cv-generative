import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class StyleGAN3Config:
    """Configuration for StyleGAN3."""
    # Architecture
    z_dim: int = 512
    c_dim: int = 0  # Number of class labels (0 = unconditional)
    w_dim: int = 512
    img_resolution: int = 1024
    img_channels: int = 3
    
    # Generator
    synthesis_layers: List[int] = None  # Will be set based on resolution
    channel_base: int = 32768
    channel_max: int = 512
    num_fp16_layers: int = 4
    conv_kernel: int = 1  # 1 or 3
    use_radial_filters: bool = True
    
    # Mapping network
    mapping_layers: int = 8
    mapping_lr_multiplier: float = 0.01
    
    # Training
    r1_gamma: float = 10.0
    blur_init_sigma: float = 0
    blur_fade_kimg: float = 0
    
    def __post_init__(self):
        if self.synthesis_layers is None:
            # Automatically determine layers based on resolution
            self.synthesis_layers = self._get_synthesis_layers()
            
    def _get_synthesis_layers(self) -> List[int]:
        """Determine synthesis layers based on resolution."""
        layers = []
        res = 4
        while res <= self.img_resolution:
            layers.append(res)
            res *= 2
        return layers

class SynthesisLayer(nn.Module):
    """Single synthesis layer with alias-free operations."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        resolution: int,
        kernel_size: int = 3,
        up: int = 1,
        use_noise: bool = True,
        activation: str = 'lrelu',
        resample_filter: Optional[List[float]] = None,
        conv_clamp: Optional[float] = None,
        channels_last: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', self._setup_filter(resample_filter))
        
        # Modulation and convolution
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        )
        
        # Noise injection
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = nn.Parameter(torch.zeros([]))
            
        # Bias and activation
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.act_gain = self._get_activation_gain(activation)
        
    def _setup_filter(self, f: Optional[List[float]]) -> torch.Tensor:
        """Setup resampling filter."""
        if f is None:
            f = [1, 3, 3, 1]  # Default bilinear filter
        f = torch.tensor(f, dtype=torch.float32)
        f = f / f.sum()
        return f
        
    def _get_activation_gain(self, activation: str) -> float:
        """Get gain for activation function."""
        if activation == 'lrelu':
            return np.sqrt(2)
        return 1.0
        
    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise_mode: str = 'random',
        fused_modconv: bool = True,
        gain: float = 1.0
    ) -> torch.Tensor:
        assert noise_mode in ['random', 'const', 'none']
        batch_size = x.shape[0]
        
        # Affine transformation
        styles = self.affine(w)
        
        # Noise injection
        noise = None
        if self.use_noise and noise_mode != 'none':
            if noise_mode == 'random':
                noise = torch.randn(
                    [batch_size, 1, self.resolution, self.resolution],
                    device=x.device
                ) * self.noise_strength
            else:  # const
                noise = self.noise_const * self.noise_strength
                
        # Modulated convolution
        x = self._modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.weight.shape[2] // 2,
            resample_filter=self.resample_filter,
            fused_modconv=fused_modconv
        )
        
        # Bias and activation
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = F.leaky_relu(x + self.bias.to(x.dtype).reshape(1, -1, 1, 1), negative_slope=0.2)
        
        if act_gain != 1:
            x = x * act_gain
            
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
            
        return x
        
    def _modulated_conv2d(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        styles: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        up: int = 1,
        padding: int = 0,
        resample_filter: Optional[torch.Tensor] = None,
        fused_modconv: bool = True
    ) -> torch.Tensor:
        """Modulated convolution with upsampling."""
        batch_size = x.shape[0]
        out_channels, in_channels, kh, kw = weight.shape
        
        # Pre-normalize weights
        weight = weight * (1 / np.sqrt(in_channels * kh * kw))
        
        # Modulate
        weight = weight.unsqueeze(0) * styles.reshape(batch_size, 1, -1, 1, 1)
        
        # Demodulate
        if fused_modconv:
            weight = weight / (weight.square().sum(dim=[2, 3, 4], keepdim=True) + 1e-8).sqrt()
            
        # Reshape for group convolution
        x = x.reshape(1, batch_size * in_channels, x.shape[2], x.shape[3])
        weight = weight.reshape(batch_size * out_channels, in_channels, kh, kw)
        
        # Upsample if needed
        if up > 1:
            x = F.interpolate(x, scale_factor=up, mode='nearest')
            
        # Convolution
        x = F.conv2d(x, weight, padding=padding, groups=batch_size)
        x = x.reshape(batch_size, out_channels, x.shape[2], x.shape[3])
        
        # Add noise
        if noise is not None:
            x = x + noise
            
        return x

class SynthesisBlock(nn.Module):
    """Synthesis block with two layers."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        resolution: int,
        img_channels: int,
        is_last: bool,
        architecture: str = 'skip',
        resample_filter: Optional[List[float]] = None,
        conv_clamp: Optional[float] = 256,
        use_fp16: bool = False,
        fp16_channels_last: bool = False,
        **layer_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.register_buffer('resample_filter', self._setup_filter(resample_filter))
        
        # Main layers
        self.conv0 = SynthesisLayer(
            in_channels, out_channels, w_dim=w_dim, resolution=resolution,
            up=2, resample_filter=resample_filter, conv_clamp=conv_clamp,
            channels_last=fp16_channels_last, **layer_kwargs
        )
        self.conv1 = SynthesisLayer(
            out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=fp16_channels_last, **layer_kwargs
        )
        
        # ToRGB
        self.torgb = ToRGBLayer(
            out_channels, img_channels, w_dim=w_dim,
            conv_clamp=conv_clamp, channels_last=fp16_channels_last
        )
        
        # Architecture-specific layers
        if architecture == 'skip':
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
            
    def _setup_filter(self, f: Optional[List[float]]) -> torch.Tensor:
        if f is None:
            f = [1, 3, 3, 1]
        return torch.tensor(f, dtype=torch.float32)
        
    def forward(
        self,
        x: torch.Tensor,
        img: Optional[torch.Tensor],
        ws: torch.Tensor,
        force_fp32: bool = False,
        fused_modconv: Optional[bool] = None,
        **layer_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        
        if fused_modconv is None:
            fused_modconv = not self.training
            
        # Main path
        if self.architecture == 'skip':
            y = self.skip(x.to(dtype))
            y = F.interpolate(y, scale_factor=2, mode='nearest')
        else:
            y = None
            
        # Apply layers
        x = self.conv0(x.to(dtype), ws[:, 0], fused_modconv=fused_modconv, **layer_kwargs)
        x = self.conv1(x, ws[:, 1], fused_modconv=fused_modconv, **layer_kwargs)
        
        # Skip connection
        if y is not None:
            x = x + y
            
        # Update image
        if img is not None:
            img = F.interpolate(img, scale_factor=2, mode='nearest')
            
        # ToRGB
        y = self.torgb(x, ws[:, 2], fused_modconv=fused_modconv)
        img = img + y if img is not None else y
        
        return x, img

class ToRGBLayer(nn.Module):
    """Convert features to RGB."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        kernel_size: int = 1,
        conv_clamp: Optional[float] = None,
        channels_last: bool = False
    ):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        )
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        
    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        fused_modconv: bool = True
    ) -> torch.Tensor:
        styles = self.affine(w)
        x = self._modulated_conv2d(x, self.weight, styles, fused_modconv=fused_modconv)
        x = x + self.bias.to(x.dtype).reshape(1, -1, 1, 1)
        return x
        
    def _modulated_conv2d(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        styles: torch.Tensor,
        fused_modconv: bool = True
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        out_channels, in_channels, kh, kw = weight.shape
        
        # Pre-normalize
        weight = weight * (1 / np.sqrt(in_channels * kh * kw))
        
        # Modulate
        weight = weight.unsqueeze(0) * styles.reshape(batch_size, 1, -1, 1, 1)
        
        # Demodulate
        if fused_modconv:
            weight = weight / (weight.square().sum(dim=[2, 3, 4], keepdim=True) + 1e-8).sqrt()
            
        # Group convolution
        x = x.reshape(1, batch_size * in_channels, x.shape[2], x.shape[3])
        weight = weight.reshape(batch_size * out_channels, in_channels, kh, kw)
        x = F.conv2d(x, weight, groups=batch_size)
        x = x.reshape(batch_size, out_channels, x.shape[2], x.shape[3])
        
        # Clamp
        if self.conv_clamp is not None:
            x = x.clamp(-self.conv_clamp, self.conv_clamp)
            
        return x

class FullyConnectedLayer(nn.Module):
    """Fully connected layer with equalized learning rate."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = 'linear',
        lr_multiplier: float = 1,
        bias_init: float = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.lr_multiplier = lr_multiplier
        
        # Weight and bias
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], bias_init)) if bias else None
        
        # Activation-specific parameters
        self.weight_gain = 1 / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            b = b * self.bias_gain
            
        # Linear
        if x.ndim == 2:
            x = F.linear(x, w, b)
        else:
            x = x @ w.t() + b.unsqueeze(0).unsqueeze(0)
            
        # Activation
        if self.activation == 'lrelu':
            x = F.leaky_relu(x, negative_slope=0.2) * np.sqrt(2)
            
        return x

class MappingNetwork(nn.Module):
    """Mapping network to generate W latent codes."""
    
    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        w_dim: int,
        num_ws: int,
        num_layers: int = 8,
        embed_features: Optional[int] = None,
        layer_features: Optional[int] = None,
        activation: str = 'lrelu',
        lr_multiplier: float = 0.01,
        w_avg_beta: float = 0.998
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        
        # Embedding for class labels
        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features or w_dim, lr_multiplier=lr_multiplier)
            
        # Main layers
        features = [z_dim + (embed_features or (c_dim if c_dim > 0 else 0))]
        for idx in range(num_layers):
            features.append(layer_features or w_dim)
            
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            in_features = features[idx]
            out_features = features[idx + 1]
            layer = FullyConnectedLayer(
                in_features, out_features, activation=activation, lr_multiplier=lr_multiplier
            )
            self.layers.append(layer)
            
        # Moving average of W
        self.register_buffer('w_avg', torch.zeros([w_dim]))
        
    def forward(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        truncation_psi: float = 1,
        truncation_cutoff: Optional[int] = None,
        update_emas: bool = False
    ) -> torch.Tensor:
        # Embed class labels
        if c is not None:
            y = self.embed(c)
            x = torch.cat([z, y], dim=1)
        else:
            x = z
            
        # Main layers
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
        # Update moving average
        if update_emas:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))
            
        # Broadcast and apply truncation
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        
        if truncation_psi != 1:
            if truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
                
        return x

class SynthesisNetwork(nn.Module):
    """Synthesis network (generator without mapping)."""
    
    def __init__(self, config: StyleGAN3Config):
        super().__init__()
        self.config = config
        
        # Calculate channels for each resolution
        def nf(stage): 
            return np.clip(int(config.channel_base / (2 ** stage)), 1, config.channel_max)
            
        # Input layer
        self.input = nn.Parameter(torch.randn([1, nf(1), 4, 4]))
        
        # Synthesis blocks
        self.blocks = nn.ModuleList()
        for idx, res in enumerate(config.synthesis_layers):
            in_channels = nf(idx + 1)
            out_channels = nf(idx + 2)
            is_last = (res == config.img_resolution)
            block = SynthesisBlock(
                in_channels, out_channels, config.w_dim, res,
                config.img_channels, is_last,
                use_fp16=(idx >= len(config.synthesis_layers) - config.num_fp16_layers)
            )
            self.blocks.append(block)
            
        # Register number of ws
        self.num_ws = sum(block.num_conv + block.num_torgb for block in self.blocks)
        
    def forward(
        self,
        ws: torch.Tensor,
        noise_mode: str = 'random',
        force_fp32: bool = False,
        **layer_kwargs
    ) -> torch.Tensor:
        # Split ws
        ws = ws.split([3] * len(self.blocks), dim=1)
        
        # Initial input
        x = self.input.repeat([ws[0].shape[0], 1, 1, 1])
        img = None
        
        # Apply blocks
        for block, w in zip(self.blocks, ws):
            x, img = block(x, img, w, noise_mode=noise_mode, force_fp32=force_fp32, **layer_kwargs)
            
        return img

class Generator(nn.Module):
    """StyleGAN3 generator."""
    
    def __init__(self, config: StyleGAN3Config):
        super().__init__()
        self.config = config
        
        # Subnetworks
        self.mapping = MappingNetwork(
            config.z_dim, config.c_dim, config.w_dim,
            num_ws=18,  # Will be updated by synthesis network
            lr_multiplier=config.mapping_lr_multiplier
        )
        self.synthesis = SynthesisNetwork(config)
        
        # Update mapping network num_ws
        self.mapping.num_ws = self.synthesis.num_ws
        
    def forward(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        truncation_psi: float = 1,
        truncation_cutoff: Optional[int] = None,
        update_emas: bool = False,
        noise_mode: str = 'random',
        force_fp32: bool = False,
        **synthesis_kwargs
    ) -> torch.Tensor:
        # Map to W
        ws = self.mapping(z, c, truncation_psi, truncation_cutoff, update_emas)
        
        # Synthesize
        img = self.synthesis(ws, noise_mode=noise_mode, force_fp32=force_fp32, **synthesis_kwargs)
        
        return img

class StyleGAN3(nn.Module):
    """Complete StyleGAN3 model with generator and discriminator."""
    
    def __init__(self, config: Optional[StyleGAN3Config] = None):
        super().__init__()
        self.config = config or StyleGAN3Config()
        self.generator = Generator(self.config)
        # Discriminator would be implemented separately
        
    @torch.no_grad()
    def generate(
        self,
        z: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        truncation_psi: float = 1.0,
        noise_mode: str = 'const',
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Generate images."""
        if z is None:
            z = torch.randn([batch_size, self.config.z_dim], device=device)
            
        # Generate images
        img = self.generator(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)
        
        # Convert to RGB [0, 1]
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        
        return img