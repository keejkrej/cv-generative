import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from typing import Optional, List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass 
class BigGANConfig:
    """Configuration for BigGAN."""
    # Architecture
    resolution: int = 512
    z_dim: int = 128
    shared_dim: int = 128
    num_classes: int = 1000
    
    # Generator
    g_ch: int = 96  # Base channel multiplier
    g_depth: List[int] = None  # Blocks per resolution
    g_attn: str = '64'  # Resolutions to use attention
    
    # Discriminator
    d_ch: int = 96
    d_depth: List[int] = None
    d_attn: str = '64'
    
    # Training
    g_lr: float = 0.00005
    d_lr: float = 0.0002
    g_beta1: float = 0.0
    g_beta2: float = 0.999
    d_beta1: float = 0.0
    d_beta2: float = 0.999
    
    # Regularization
    g_spectral_norm: bool = True
    d_spectral_norm: bool = True
    g_init: str = 'ortho'
    d_init: str = 'ortho'
    
    def __post_init__(self):
        if self.g_depth is None:
            # Default depths for different resolutions
            if self.resolution == 512:
                self.g_depth = [2, 2, 2, 2, 2, 1]
            elif self.resolution == 256:
                self.g_depth = [2, 2, 2, 2, 1]
            elif self.resolution == 128:
                self.g_depth = [2, 2, 2, 1]
            else:
                self.g_depth = [2, 2, 1]
                
        if self.d_depth is None:
            self.d_depth = self.g_depth

def l2normalize(v, eps=1e-12):
    """L2 normalize vector."""
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """Spectral normalization for layer weights."""
    
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()
        
    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        
        # Register buffers
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", P(w.data))
        
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
        
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
            
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

class SelfAttention(nn.Module):
    """Self-attention module with spectral normalization."""
    
    def __init__(self, ch, spectral_norm=True):
        super().__init__()
        self.ch = ch
        
        # 1x1 convolutions for Q, K, V
        self.theta = nn.Conv2d(ch, ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(ch, ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(ch, ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(ch // 2, ch, kernel_size=1, padding=0, bias=False)
        
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Apply spectral norm
        if spectral_norm:
            self.theta = SpectralNorm(self.theta)
            self.phi = SpectralNorm(self.phi)
            self.g = SpectralNorm(self.g)
            self.o = SpectralNorm(self.o)
            
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        theta = self.theta(x).view(B, -1, H * W)
        phi = self.phi(x).view(B, -1, H * W)
        g = self.g(x).view(B, -1, H * W)
        
        # Attention
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=1)
        o = self.o(torch.bmm(g, beta).view(B, C // 2, H, W))
        
        return self.gamma * o + x

class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization."""
    
    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
        self.embed_gamma = nn.Linear(num_classes, num_features)
        self.embed_beta = nn.Linear(num_classes, num_features)
        self._initialize()
        
    def _initialize(self):
        nn.init.ones_(self.embed_gamma.weight)
        nn.init.zeros_(self.embed_beta.weight)
        
    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.embed_gamma(y).unsqueeze(-1).unsqueeze(-1)
        beta = self.embed_beta(y).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta

class GBlock(nn.Module):
    """Generator residual block with class-conditional BN."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        upsample: bool = False,
        activation: nn.Module = nn.ReLU(inplace=True),
        spectral_norm: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.activation = activation
        
        # Main path
        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        # Apply spectral norm
        if spectral_norm:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)
            self.skip = SpectralNorm(self.skip)
            
    def forward(self, x, y):
        # Main path
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip(x)

class DBlock(nn.Module):
    """Discriminator residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        activation: nn.Module = nn.ReLU(inplace=True),
        spectral_norm: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.activation = activation
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        # Apply spectral norm
        if spectral_norm:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)
            self.skip = SpectralNorm(self.skip)
            
    def forward(self, x):
        # Main path
        h = self.activation(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            
        # Skip connection
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return h + self.skip(x)

class Generator(nn.Module):
    """BigGAN Generator."""
    
    def __init__(self, config: BigGANConfig):
        super().__init__()
        self.config = config
        self.ch = config.g_ch
        self.z_dim = config.z_dim
        self.num_classes = config.num_classes
        self.shared_dim = config.shared_dim
        
        # Shared embedding for z and class
        self.shared = nn.Linear(config.z_dim + config.shared_dim, 
                               16 * self.ch * 4 * 4)
        
        # Class embedding
        self.class_embed = nn.Embedding(config.num_classes, config.shared_dim)
        
        # Resolution schedule
        res_schedule = self._get_res_schedule(config.resolution)
        
        # Build blocks
        self.blocks = nn.ModuleList()
        ch_in = 16 * self.ch
        
        for i, (res_in, res_out) in enumerate(res_schedule):
            for j in range(config.g_depth[i]):
                ch_out = self._get_ch(res_out)
                upsample = (j == config.g_depth[i] - 1) and (res_in != res_out)
                
                block = GBlock(
                    ch_in, ch_out, config.num_classes,
                    upsample=upsample,
                    spectral_norm=config.g_spectral_norm
                )
                self.blocks.append(block)
                ch_in = ch_out
                
            # Add attention
            if str(res_out) in config.g_attn:
                self.blocks.append(SelfAttention(ch_in, config.g_spectral_norm))
                
        # Output layers
        self.bn_out = nn.BatchNorm2d(ch_in)
        self.conv_out = nn.Conv2d(ch_in, 3, kernel_size=3, padding=1)
        if config.g_spectral_norm:
            self.conv_out = SpectralNorm(self.conv_out)
            
        # Initialize weights
        self._initialize_weights()
        
    def _get_res_schedule(self, resolution):
        """Get resolution progression schedule."""
        schedule = []
        res = 4
        while res <= resolution:
            schedule.append((res, min(res * 2, resolution)))
            res *= 2
        return schedule
        
    def _get_ch(self, res):
        """Get number of channels for resolution."""
        return int(self.ch * (2 ** (np.log2(512) - np.log2(res))))
        
    def _initialize_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, z, y):
        """Forward pass.
        
        Args:
            z: Latent vector [B, z_dim]
            y: Class labels [B]
        """
        # Class embedding
        y_emb = self.class_embed(y)
        
        # Concatenate z and class embedding
        z_y = torch.cat([z, y_emb], dim=1)
        
        # Initial projection
        h = self.shared(z_y)
        h = h.view(h.size(0), -1, 4, 4)
        
        # Pass through blocks
        for block in self.blocks:
            if isinstance(block, GBlock):
                h = block(h, y)
            else:  # SelfAttention
                h = block(h)
                
        # Output
        h = F.relu(self.bn_out(h))
        h = torch.tanh(self.conv_out(h))
        
        return h

class Discriminator(nn.Module):
    """BigGAN Discriminator."""
    
    def __init__(self, config: BigGANConfig):
        super().__init__()
        self.config = config
        self.ch = config.d_ch
        
        # Input layer
        self.conv_in = nn.Conv2d(3, self.ch, kernel_size=3, padding=1)
        if config.d_spectral_norm:
            self.conv_in = SpectralNorm(self.conv_in)
            
        # Resolution schedule
        res_schedule = self._get_res_schedule(config.resolution)
        
        # Build blocks
        self.blocks = nn.ModuleList()
        ch_in = self.ch
        
        for i, (res_in, res_out) in enumerate(reversed(res_schedule)):
            # Add attention first at this resolution
            if str(res_in) in config.d_attn:
                self.blocks.append(SelfAttention(ch_in, config.d_spectral_norm))
                
            for j in range(config.d_depth[-(i+1)]):
                ch_out = self._get_ch(res_out)
                downsample = (j == 0) and (res_in != res_out)
                
                block = DBlock(
                    ch_in, ch_out,
                    downsample=downsample,
                    spectral_norm=config.d_spectral_norm
                )
                self.blocks.append(block)
                ch_in = ch_out
                
        # Output layers
        self.linear = nn.Linear(ch_in * 4 * 4, 1)
        if config.d_spectral_norm:
            self.linear = SpectralNorm(self.linear)
            
        # Class embedding for projection discriminator
        self.embed = nn.Embedding(config.num_classes, ch_in * 4 * 4)
        if config.d_spectral_norm:
            self.embed = SpectralNorm(self.embed)
            
        # Initialize weights
        self._initialize_weights()
        
    def _get_res_schedule(self, resolution):
        """Get resolution progression schedule."""
        schedule = []
        res = 4
        while res <= resolution:
            schedule.append((res, min(res * 2, resolution)))
            res *= 2
        return schedule
        
    def _get_ch(self, res):
        """Get number of channels for resolution."""
        return int(self.ch * (2 ** (np.log2(res) - np.log2(4))))
        
    def _initialize_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x, y):
        """Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            y: Class labels [B]
        """
        # Initial conv
        h = self.conv_in(x)
        
        # Pass through blocks
        for block in self.blocks:
            h = block(h)
            
        # Global sum pooling
        h = F.relu(h)
        h = h.view(h.size(0), -1)
        
        # Output with projection
        out = self.linear(h)
        
        # Class-conditional projection
        if y is not None:
            proj = torch.sum(self.embed(y) * h, dim=1, keepdim=True)
            out = out + proj
            
        return out

class BigGAN(nn.Module):
    """Complete BigGAN model."""
    
    def __init__(self, config: Optional[BigGANConfig] = None):
        super().__init__()
        self.config = config or BigGANConfig()
        self.generator = Generator(self.config)
        self.discriminator = Discriminator(self.config)
        
    @torch.no_grad()
    def generate(
        self,
        z: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        truncation: float = 1.0,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Generate images.
        
        Args:
            z: Latent vectors [B, z_dim]
            y: Class labels [B]
            batch_size: Batch size if z not provided
            truncation: Truncation value for sampling
            device: Device to generate on
        """
        # Sample z if not provided
        if z is None:
            z = torch.randn(batch_size, self.config.z_dim, device=device)
            z = z * truncation
            
        # Sample random class if not provided
        if y is None:
            y = torch.randint(0, self.config.num_classes, (batch_size,), device=device)
            
        # Generate
        images = self.generator(z, y)
        
        # Convert to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        return images