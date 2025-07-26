import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import itertools
from dataclasses import dataclass

@dataclass
class CycleGANConfig:
    """Configuration for CycleGAN."""
    # Model architecture
    input_channels: int = 3
    output_channels: int = 3
    ngf: int = 64  # Generator filters
    ndf: int = 64  # Discriminator filters
    n_blocks: int = 9  # Number of ResNet blocks
    
    # Training
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cycle: float = 10.0  # Cycle consistency loss weight
    lambda_identity: float = 0.5  # Identity loss weight
    
    # Others
    pool_size: int = 50  # Image pool size for discriminator
    use_dropout: bool = False
    norm_type: str = 'instance'  # 'instance' or 'batch'
    init_type: str = 'normal'  # 'normal', 'xavier', 'kaiming'
    init_gain: float = 0.02

class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    
    def __init__(self, channels: int, use_dropout: bool = False, norm_type: str = 'instance'):
        super().__init__()
        
        # Choose normalization
        if norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
            
        # Build block
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            norm_layer(channels),
            nn.ReLU(True)
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            norm_layer(channels)
        ])
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class Generator(nn.Module):
    """CycleGAN Generator with ResNet architecture."""
    
    def __init__(self, config: CycleGANConfig):
        super().__init__()
        
        # Choose normalization
        if config.norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
            
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(config.input_channels, config.ngf, kernel_size=7, padding=0, bias=True),
            norm_layer(config.ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling layers
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model.extend([
                nn.Conv2d(config.ngf * mult, config.ngf * mult * 2, 
                         kernel_size=3, stride=2, padding=1, bias=True),
                norm_layer(config.ngf * mult * 2),
                nn.ReLU(True)
            ])
            
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(config.n_blocks):
            model.append(
                ResidualBlock(
                    config.ngf * mult,
                    use_dropout=config.use_dropout,
                    norm_type=config.norm_type
                )
            )
            
        # Upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.extend([
                nn.ConvTranspose2d(config.ngf * mult, config.ngf * mult // 2,
                                  kernel_size=3, stride=2, padding=1, 
                                  output_padding=1, bias=True),
                norm_layer(config.ngf * mult // 2),
                nn.ReLU(True)
            ])
            
        # Output layer
        model.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(config.ngf, config.output_channels, kernel_size=7, padding=0),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*model)
        self._initialize_weights(config.init_type, config.init_gain)
        
    def _initialize_weights(self, init_type: str, init_gain: float):
        """Initialize network weights."""
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                    
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
                    
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)
                
        self.apply(init_func)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Discriminator(nn.Module):
    """PatchGAN discriminator."""
    
    def __init__(self, config: CycleGANConfig):
        super().__init__()
        
        # Choose normalization
        if config.norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
            
        # Build discriminator
        model = [
            nn.Conv2d(config.input_channels, config.ndf, kernel_size=4, 
                     stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Hidden layers
        n_layers = 3
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model.extend([
                nn.Conv2d(config.ndf * nf_mult_prev, config.ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1, bias=True),
                norm_layer(config.ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ])
            
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model.extend([
            nn.Conv2d(config.ndf * nf_mult_prev, config.ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(config.ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ])
        
        # Output layer
        model.append(
            nn.Conv2d(config.ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.model = nn.Sequential(*model)
        self._initialize_weights(config.init_type, config.init_gain)
        
    def _initialize_weights(self, init_type: str, init_gain: float):
        """Initialize network weights."""
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                    
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
                    
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)
                
        self.apply(init_func)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ImagePool:
    """Image pool to store previously generated images."""
    
    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images = []
        
    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return images from pool with 50% chance."""
        if self.pool_size == 0:
            return images
            
        return_images = []
        
        for image in images:
            if len(self.images) < self.pool_size:
                self.images.append(image.unsqueeze(0))
                return_images.append(image.unsqueeze(0))
            else:
                p = torch.rand(1).item()
                if p > 0.5:
                    # Return image from pool
                    random_id = torch.randint(0, self.pool_size, (1,)).item()
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image.unsqueeze(0)
                    return_images.append(tmp)
                else:
                    # Return current image
                    return_images.append(image.unsqueeze(0))
                    
        return torch.cat(return_images, dim=0)

class CycleGAN(nn.Module):
    """Complete CycleGAN model."""
    
    def __init__(self, config: Optional[CycleGANConfig] = None):
        super().__init__()
        self.config = config or CycleGANConfig()
        
        # Generators
        self.G_A2B = Generator(self.config)  # A -> B
        self.G_B2A = Generator(self.config)  # B -> A
        
        # Discriminators
        self.D_A = Discriminator(self.config)  # For domain A
        self.D_B = Discriminator(self.config)  # For domain B
        
        # Image pools
        self.fake_A_pool = ImagePool(self.config.pool_size)
        self.fake_B_pool = ImagePool(self.config.pool_size)
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
    def forward(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> dict:
        """Forward pass for training.
        
        Args:
            real_A: Images from domain A
            real_B: Images from domain B
            
        Returns:
            Dictionary with all outputs and losses
        """
        # Generate fake images
        fake_B = self.G_A2B(real_A)
        fake_A = self.G_B2A(real_B)
        
        # Reconstruct images
        rec_A = self.G_B2A(fake_B)
        rec_B = self.G_A2B(fake_A)
        
        # Identity mapping (optional)
        idt_A = self.G_B2A(real_A)
        idt_B = self.G_A2B(real_B)
        
        return {
            'fake_A': fake_A,
            'fake_B': fake_B,
            'rec_A': rec_A,
            'rec_B': rec_B,
            'idt_A': idt_A,
            'idt_B': idt_B
        }
        
    def compute_losses(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        outputs: dict
    ) -> dict:
        """Compute all losses."""
        device = real_A.device
        
        # Adversarial losses for generators
        pred_fake_B = self.D_B(outputs['fake_B'])
        loss_GAN_A2B = self.criterion_GAN(
            pred_fake_B,
            torch.ones_like(pred_fake_B)
        )
        
        pred_fake_A = self.D_A(outputs['fake_A'])
        loss_GAN_B2A = self.criterion_GAN(
            pred_fake_A,
            torch.ones_like(pred_fake_A)
        )
        
        # Cycle consistency losses
        loss_cycle_A = self.criterion_cycle(outputs['rec_A'], real_A)
        loss_cycle_B = self.criterion_cycle(outputs['rec_B'], real_B)
        
        # Identity losses
        loss_idt_A = self.criterion_identity(outputs['idt_A'], real_A)
        loss_idt_B = self.criterion_identity(outputs['idt_B'], real_B)
        
        # Total generator losses
        loss_G = (
            loss_GAN_A2B + loss_GAN_B2A +
            self.config.lambda_cycle * (loss_cycle_A + loss_cycle_B) +
            self.config.lambda_identity * (loss_idt_A + loss_idt_B)
        )
        
        # Discriminator losses
        # D_A
        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(self.fake_A_pool.query(outputs['fake_A'].detach()))
        loss_D_A = 0.5 * (
            self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A)) +
            self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        )
        
        # D_B
        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(self.fake_B_pool.query(outputs['fake_B'].detach()))
        loss_D_B = 0.5 * (
            self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B)) +
            self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        )
        
        return {
            'loss_G': loss_G,
            'loss_D_A': loss_D_A,
            'loss_D_B': loss_D_B,
            'loss_GAN_A2B': loss_GAN_A2B,
            'loss_GAN_B2A': loss_GAN_B2A,
            'loss_cycle_A': loss_cycle_A,
            'loss_cycle_B': loss_cycle_B,
            'loss_idt_A': loss_idt_A,
            'loss_idt_B': loss_idt_B
        }
        
    @torch.no_grad()
    def translate(
        self,
        image: torch.Tensor,
        direction: str = 'A2B'
    ) -> torch.Tensor:
        """Translate image from one domain to another.
        
        Args:
            image: Input image tensor
            direction: 'A2B' or 'B2A'
            
        Returns:
            Translated image
        """
        if direction == 'A2B':
            return self.G_A2B(image)
        else:
            return self.G_B2A(image)
            
    def get_optimizers(self, lr: Optional[float] = None):
        """Get optimizers for training."""
        if lr is None:
            lr = self.config.lr
            
        # Generator optimizer (both G_A2B and G_B2A)
        opt_G = torch.optim.Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=lr,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        # Discriminator optimizers
        opt_D_A = torch.optim.Adam(
            self.D_A.parameters(),
            lr=lr,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        opt_D_B = torch.optim.Adam(
            self.D_B.parameters(),
            lr=lr,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        return opt_G, opt_D_A, opt_D_B