import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class SDXLConfig:
    """Configuration for Stable Diffusion XL model."""
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    attention_resolutions: Tuple[int, ...] = (4, 2)
    num_res_blocks: int = 2
    channel_mult: Tuple[int, ...] = (1, 2, 4)
    num_heads: int = 8
    num_head_channels: int = 64
    use_spatial_transformer: bool = True
    transformer_depth: int = 1
    context_dim: int = 2048
    legacy: bool = False
    use_checkpoint: bool = False
    use_fp16: bool = False
    
    # SDXL specific
    use_linear_in_transformer: bool = True
    adm_in_channels: int = 2816
    dual_attention_layers: List[int] = None
    transformer_depth_middle: Optional[int] = None

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) 
            * (torch.log(torch.tensor(self.max_period)) / half_dim)
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class CrossAttention(nn.Module):
    """Multi-head cross attention module."""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.heads
        
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], h, -1).transpose(1, 2), (q, k, v))
        
        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if mask is not None:
            mask = mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)
            
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1] * h)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    """Transformer block with self and cross attention."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        use_linear: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout
        )
        
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.attn1(self.norm1(x), context=None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class ResnetBlock(nn.Module):
    """ResNet block with optional time embedding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_conv_shortcut: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_emb_proj = None
            
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if self.in_channels != self.out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = torch.nn.functional.silu(h)
        h = self.conv1(h)
        
        if time_emb is not None and self.time_emb_proj is not None:
            h = h + self.time_emb_proj(torch.nn.functional.silu(time_emb))[:, :, None, None]
            
        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.shortcut(x) + h

class SDXLUNet(nn.Module):
    """SDXL U-Net architecture with dual conditioning."""
    
    def __init__(self, config: SDXLConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        time_embed_dim = config.model_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(config.model_channels),
            nn.Linear(config.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Additional conditioning embedding (for size/crop conditioning)
        if config.adm_in_channels > 0:
            self.label_emb = nn.Sequential(
                nn.Linear(config.adm_in_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(config.in_channels, config.model_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = config.model_channels
        ds = 1
        
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                layers = []
                ch_in = ch
                ch = mult * config.model_channels
                
                layers.append(ResnetBlock(ch_in, ch, time_emb_dim=time_embed_dim))
                
                if ds in config.attention_resolutions:
                    if config.use_spatial_transformer:
                        layers.append(TransformerBlock(
                            ch, config.num_heads, config.num_head_channels,
                            context_dim=config.context_dim,
                            use_linear=config.use_linear_in_transformer
                        ))
                        
                self.down_blocks.append(nn.ModuleList(layers))
                
            if level != len(config.channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
                ]))
                ds *= 2
                
        # Middle
        self.middle_block = nn.ModuleList([
            ResnetBlock(ch, ch, time_emb_dim=time_embed_dim),
            TransformerBlock(
                ch, config.num_heads, config.num_head_channels,
                context_dim=config.context_dim,
                use_linear=config.use_linear_in_transformer
            ),
            ResnetBlock(ch, ch, time_emb_dim=time_embed_dim)
        ])
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(config.channel_mult))):
            for i in range(config.num_res_blocks + 1):
                layers = []
                ch_in = ch
                ch = config.model_channels * mult
                
                layers.append(ResnetBlock(ch_in, ch, time_emb_dim=time_embed_dim))
                
                if ds in config.attention_resolutions:
                    if config.use_spatial_transformer:
                        layers.append(TransformerBlock(
                            ch, config.num_heads, config.num_head_channels,
                            context_dim=config.context_dim,
                            use_linear=config.use_linear_in_transformer
                        ))
                        
                self.up_blocks.append(nn.ModuleList(layers))
                
            if level != 0:
                self.up_blocks.append(nn.ModuleList([
                    nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
                ]))
                ds //= 2
                
        # Output
        self.norm_out = nn.GroupNorm(32, config.model_channels)
        self.conv_out = nn.Conv2d(config.model_channels, config.out_channels, kernel_size=3, padding=1)
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Add conditioning if provided
        if y is not None and hasattr(self, 'label_emb'):
            t_emb = t_emb + self.label_emb(y)
            
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder
        hs = []
        for module in self.down_blocks:
            for layer in module:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, TransformerBlock):
                    h = layer(h, context)
                else:
                    hs.append(h)
                    h = layer(h)
                    
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResnetBlock):
                h = layer(h, t_emb)
            elif isinstance(layer, TransformerBlock):
                h = layer(h, context)
                
        # Decoder
        for module in self.up_blocks:
            for layer in module:
                if isinstance(layer, ResnetBlock):
                    if len(hs) > 0:
                        h = torch.cat([h, hs.pop()], dim=1)
                    h = layer(h, t_emb)
                elif isinstance(layer, TransformerBlock):
                    h = layer(h, context)
                else:
                    h = layer(h)
                    
        # Output
        h = self.norm_out(h)
        h = torch.nn.functional.silu(h)
        h = self.conv_out(h)
        
        return h

class StableDiffusionXL(nn.Module):
    """Complete Stable Diffusion XL model."""
    
    def __init__(self, config: SDXLConfig):
        super().__init__()
        self.config = config
        self.unet = SDXLUNet(config)
        
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        """Forward pass through SDXL model.
        
        Args:
            latents: Noisy latents [B, C, H, W]
            timesteps: Timesteps [B]
            encoder_hidden_states: Text embeddings [B, seq_len, embed_dim]
            added_cond_kwargs: Additional conditioning (size/crop info)
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        y = None
        if added_cond_kwargs is not None:
            y = added_cond_kwargs.get("text_embeds", None)
            
        return self.unet(latents, timesteps, encoder_hidden_states, y)
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        """Load pretrained SDXL model."""
        # This would load from HuggingFace or local checkpoint
        # For now, return initialized model
        config = SDXLConfig(**kwargs)
        return cls(config)