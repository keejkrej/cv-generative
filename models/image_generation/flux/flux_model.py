import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math

@dataclass
class FluxConfig:
    """Configuration for FLUX.1 model."""
    # Model dimensions
    hidden_size: int = 3072
    num_layers: int = 24
    num_heads: int = 24
    mlp_ratio: float = 4.0
    
    # Conditioning
    text_embed_dim: int = 4096
    time_embed_dim: int = 256
    vector_embed_dim: int = 768
    
    # Image settings
    patch_size: int = 2
    in_channels: int = 16  # FLUX uses 16-channel VAE
    out_channels: int = 16
    
    # Architecture choices
    use_rope: bool = True
    use_parallel_blocks: bool = True
    qk_norm: bool = True
    
    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0

class RoPE(nn.Module):
    """Rotary Position Embeddings."""
    
    def __init__(self, dim: int, max_seq_len: int = 16384, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for frequencies
        self._cached_freqs = None
        self._cached_seq_len = 0
        
    def _compute_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._cached_freqs is not None and self._cached_seq_len >= seq_len:
            return self._cached_freqs[:seq_len].to(device)
            
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        
        self._cached_freqs = freqs
        self._cached_seq_len = seq_len
        return freqs
        
    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        seq_len = x.shape[1]
        freqs = self._compute_freqs(seq_len, x.device)
        
        # Reshape for complex numbers
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_complex = torch.view_as_complex(torch.stack([freqs.cos(), freqs.sin()], dim=-1))
        
        # Apply rotation
        x_rotated = x_complex * freqs_complex
        
        # Convert back
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)

class ParallelAttention(nn.Module):
    """Parallel self-attention and cross-attention block."""
    
    def __init__(self, config: FluxConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Self-attention components
        self.q_self = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_self = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_self = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Cross-attention components
        self.q_cross = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_cross = nn.Linear(config.text_embed_dim, config.hidden_size, bias=False)
        self.v_cross = nn.Linear(config.text_embed_dim, config.hidden_size, bias=False)
        
        # Output projections
        self.out_self = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_cross = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Optional QK normalization
        if config.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            
        # RoPE for positional encoding
        if config.use_rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None
            
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = x.shape
        
        # Self-attention
        q_self = self.q_self(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_self = self.k_self(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_self = self.v_self(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.rope is not None:
            q_self = self.rope.apply_rope(q_self)
            k_self = self.rope.apply_rope(k_self)
            
        # Apply QK normalization
        q_self = self.q_norm(q_self)
        k_self = self.k_norm(k_self)
        
        # Self-attention computation
        attn_self = (q_self @ k_self.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_self = attn_self.masked_fill(mask[:, None, None, :] == 0, -1e9)
        attn_self = F.softmax(attn_self, dim=-1)
        attn_self = self.dropout(attn_self)
        
        out_self = (attn_self @ v_self).transpose(1, 2).reshape(B, L, C)
        out_self = self.out_self(out_self)
        
        # Cross-attention
        q_cross = self.q_cross(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_cross = self.k_cross(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_cross = self.v_cross(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK normalization
        q_cross = self.q_norm(q_cross)
        k_cross = self.k_norm(k_cross)
        
        # Cross-attention computation
        attn_cross = (q_cross @ k_cross.transpose(-2, -1)) * self.scale
        attn_cross = F.softmax(attn_cross, dim=-1)
        attn_cross = self.dropout(attn_cross)
        
        out_cross = (attn_cross @ v_cross).transpose(1, 2).reshape(B, L, C)
        out_cross = self.out_cross(out_cross)
        
        return out_self, out_cross

class FluxTransformerBlock(nn.Module):
    """Transformer block with parallel attention and modulation."""
    
    def __init__(self, config: FluxConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        
        self.parallel_attn = ParallelAttention(config)
        
        # MLP
        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Modulation parameters (for conditioning)
        self.adaLN_modulation = nn.Linear(config.vector_embed_dim, 6 * config.hidden_size, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        vec: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(vec).chunk(6, dim=-1)
            
        # Self-attention and cross-attention (parallel)
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        attn_self, attn_cross = self.parallel_attn(x_norm, context, mask)
        x = x + gate_msa.unsqueeze(1) * (attn_self + attn_cross)
        
        # MLP
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x

class FluxTransformer(nn.Module):
    """FLUX.1 Transformer model for image generation."""
    
    def __init__(self, config: FluxConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.vector_embed_dim),
            nn.SiLU(),
            nn.Linear(config.vector_embed_dim, config.vector_embed_dim)
        )
        
        # Vector embeddings for additional conditioning
        self.vector_embed = nn.Linear(config.vector_embed_dim, config.vector_embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FluxTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output layers
        self.norm_out = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.ada_norm_out = nn.Linear(config.vector_embed_dim, 2 * config.hidden_size, bias=False)
        
        # Unpatchify
        self.out_conv = nn.Conv2d(
            config.hidden_size,
            config.out_channels * config.patch_size ** 2,
            kernel_size=1
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                
        self.apply(_basic_init)
        
        # Zero-init adaLN modulation
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation.weight)
            
    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert from patch tokens back to image."""
        B = x.shape[0]
        x = x.reshape(B, H, W, self.config.hidden_size)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        x = self.out_conv(x)
        x = x.reshape(B, self.config.out_channels, self.config.patch_size, self.config.patch_size, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)  # (B, C, H, p, W, p)
        x = x.reshape(B, self.config.out_channels, H * self.config.patch_size, W * self.config.patch_size)
        
        return x
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through FLUX transformer.
        
        Args:
            x: Input image/latent [B, C, H, W]
            timesteps: Timesteps [B]
            context: Text embeddings [B, seq_len, embed_dim]
            y: Optional additional conditioning [B, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Output prediction [B, C, H, W]
        """
        B, C, H_orig, W_orig = x.shape
        
        # Patchify
        x = self.patch_embed(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Time and vector conditioning
        t_emb = self.time_embed(timestep_embedding(timesteps, self.config.time_embed_dim))
        
        if y is not None:
            vec = t_emb + self.vector_embed(y)
        else:
            vec = t_emb
            
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, context, vec, mask)
            
        # Output norm
        x = self.norm_out(x)
        shift, scale = self.ada_norm_out(vec).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Unpatchify
        x = self.unpatchify(x, H, W)
        
        return x

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Flux1(nn.Module):
    """FLUX.1 complete model with flow matching."""
    
    def __init__(self, config: Optional[FluxConfig] = None):
        super().__init__()
        self.config = config or FluxConfig()
        self.transformer = FluxTransformer(self.config)
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for flow matching."""
        return self.transformer(x, timesteps, context, **kwargs)
        
    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate images using rectified flow."""
        # This is a simplified generation method
        # Full implementation would include the complete flow matching sampling
        device = prompt_embeds.device
        
        # Initialize latents
        latents = torch.randn(
            (1, self.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=device
        )
        
        # Simple Euler sampling for demonstration
        for i, t in enumerate(torch.linspace(1, 0, num_inference_steps)):
            t_batch = torch.tensor([t], device=device)
            
            # Predict velocity
            v_pred = self.forward(latents, t_batch, prompt_embeds)
            
            # Update latents (simplified)
            if i < num_inference_steps - 1:
                dt = 1.0 / num_inference_steps
                latents = latents - dt * v_pred
                
        return latents