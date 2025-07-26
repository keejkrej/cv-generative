import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
from dataclasses import dataclass

@dataclass
class GaussianConfig:
    """Configuration for 3D Gaussian Splatting."""
    # Gaussian parameters
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.001
    rotation_lr: float = 0.001
    
    # Densification
    percent_dense: float = 0.01
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    
    # Rendering
    sh_degree: int = 3
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

class GaussianModel(nn.Module):
    """3D Gaussian representation for real-time rendering."""
    
    def __init__(self, config: GaussianConfig, sh_degree: int = 3):
        super().__init__()
        self.config = config
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        
        # Gaussian parameters
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        # Optimization helpers
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        
        # Setup functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = self.build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = lambda x: torch.log(x / (1 - x))
        self.rotation_activation = torch.nn.functional.normalize
        
    def build_covariance_from_scaling_rotation(
        self,
        scaling: torch.Tensor,
        scaling_modifier: float,
        rotation: torch.Tensor
    ) -> torch.Tensor:
        """Build 3D covariance matrix from scaling and rotation."""
        L = self.build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        
        # Add small epsilon for numerical stability
        symm = (actual_covariance + actual_covariance.transpose(1, 2)) / 2
        symm += 1e-7 * torch.eye(3, device=symm.device).unsqueeze(0)
        
        return symm
        
    def build_scaling_rotation(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Build transformation matrix from scaling and rotation quaternion."""
        L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
        
        # Normalize quaternion
        r = self.rotation_activation(r)
        
        # Convert quaternion to rotation matrix
        x, y, z, w = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        
        # Build rotation matrix R
        R = torch.zeros((r.shape[0], 3, 3), device=r.device)
        
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)
        
        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - w*x)
        
        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        
        # Apply scaling
        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]
        
        L = R @ L
        
        return L
        
    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz
        
    @property
    def get_features(self) -> torch.Tensor:
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
        
    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)
        
    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)
        
    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return self.covariance_activation(
            self.get_scaling,
            scaling_modifier,
            self._rotation
        )
        
    def create_from_pcd(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        spatial_lr_scale: float = 1.0
    ):
        """Initialize Gaussians from point cloud."""
        self.spatial_lr_scale = spatial_lr_scale
        
        # Compute initial scales based on nearest neighbors
        dist2 = torch.clamp_min(
            self.distCUDA2(points),
            0.0000001
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        
        # Initialize 3D positions
        self._xyz = nn.Parameter(points.requires_grad_(True))
        
        # Initialize colors (SH coefficients)
        fused_color = colors
        features = torch.zeros((colors.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        
        # Initialize scales
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        
        # Initialize rotations (quaternions)
        rots = torch.zeros((points.shape[0], 4))
        rots[:, 0] = 1  # w component
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        
        # Initialize opacity
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((points.shape[0], 1), dtype=torch.float)
        )
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # Initialize optimization helpers
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1))
        self.denom = torch.zeros((self.get_xyz.shape[0], 1))
        
    def distCUDA2(self, points: torch.Tensor) -> torch.Tensor:
        """Compute squared distances to nearest neighbors."""
        # Simple implementation - in practice would use CUDA kernel
        xx = (points**2).sum(dim=1, keepdim=True)
        yy = xx.t()
        xy = torch.mm(points, points.t())
        dists = xx + yy - 2*xy
        
        # Set diagonal to infinity
        dists.diagonal().fill_(float('inf'))
        
        # Get minimum distance for each point
        return dists.min(dim=1)[0]
        
    def densify_and_prune(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        size_threshold: float = None
    ):
        """Densification and pruning of Gaussians."""
        if size_threshold is None:
            size_threshold = 20 if scene_extent > 1 else scene_extent / 50
            
        # Accumulate gradients
        selected_pts_mask = grads >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_scaling.max(dim=1).values <= size_threshold
        )
        
        # Clone selected Gaussians
        new_xyz = self._xyz[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(2, 1)
        new_scaling = self._scaling[selected_pts_mask].repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        
        # Add noise to positions
        new_xyz += torch.randn_like(new_xyz) * self.get_scaling[selected_pts_mask].repeat(2, 1) * 0.1
        
        # Reduce scale
        new_scaling = self.scaling_inverse_activation(
            self.scaling_activation(new_scaling) / 1.6
        )
        
        # Concatenate with existing
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacities], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        
        # Prune Gaussians
        prune_mask = (self.get_opacity < 0.005).squeeze()
        
        if prune_mask.sum() > 0:
            self.prune_points(prune_mask)
            
    def prune_points(self, mask: torch.Tensor):
        """Remove Gaussians based on mask."""
        valid_points_mask = ~mask
        
        self._xyz = nn.Parameter(self._xyz[valid_points_mask])
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask])
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask])
        self._opacity = nn.Parameter(self._opacity[valid_points_mask])
        self._scaling = nn.Parameter(self._scaling[valid_points_mask])
        self._rotation = nn.Parameter(self._rotation[valid_points_mask])
        
        # Update optimization helpers
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

class GaussianRenderer(nn.Module):
    """Differentiable renderer for 3D Gaussians."""
    
    def __init__(self, config: GaussianConfig):
        super().__init__()
        self.config = config
        
    def render(
        self,
        viewpoint_camera: Dict,
        pc: GaussianModel,
        bg_color: torch.Tensor = None,
        scaling_modifier: float = 1.0,
        override_color: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Render Gaussians from a viewpoint.
        
        Args:
            viewpoint_camera: Camera parameters dict
            pc: GaussianModel instance
            bg_color: Background color
            scaling_modifier: Scale modifier
            override_color: Optional color override
            
        Returns:
            Dict containing rendered image and auxiliary outputs
        """
        # Get camera parameters
        tanfovx = viewpoint_camera["tanfovx"]
        tanfovy = viewpoint_camera["tanfovy"]
        viewmatrix = viewpoint_camera["viewmatrix"]
        projmatrix = viewpoint_camera["projmatrix"]
        campos = viewpoint_camera["campos"]
        
        # Set background
        if bg_color is None:
            bg_color = torch.zeros(3, device=pc.get_xyz.device)
            
        # Get Gaussian parameters
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features if override_color is None else override_color
        
        # Transform to camera space
        means_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=1)
        means_cam = (viewmatrix @ means_homo.T).T[:, :3]
        
        # Project to screen space
        means_proj = (projmatrix @ torch.cat([means_cam, torch.ones_like(means_cam[:, :1])], dim=1).T).T
        means_ndc = means_proj[:, :3] / means_proj[:, 3:4]
        
        # Simple rasterization (placeholder - real implementation would use CUDA)
        # This is a simplified version for demonstration
        H, W = viewpoint_camera["image_height"], viewpoint_camera["image_width"]
        rendered = torch.zeros((3, H, W), device=means3D.device)
        
        # Convert NDC to pixel coordinates
        px = ((means_ndc[:, 0] + 1) * W / 2).long()
        py = ((means_ndc[:, 1] + 1) * H / 2).long()
        
        # Filter visible points
        valid_mask = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (means_cam[:, 2] > 0)
        
        # Simple splat (real implementation would use 2D Gaussians)
        for i in torch.where(valid_mask)[0]:
            if opacity[i] > 0.01:
                rendered[:, py[i], px[i]] = shs[i, 0, :] * opacity[i]
                
        return {
            "render": rendered,
            "viewspace_points": means_cam,
            "visibility_filter": valid_mask,
            "radii": torch.ones_like(px)  # Placeholder
        }

class GaussianSplatting(nn.Module):
    """Complete 3D Gaussian Splatting system."""
    
    def __init__(self, config: Optional[GaussianConfig] = None):
        super().__init__()
        self.config = config or GaussianConfig()
        self.gaussians = GaussianModel(self.config)
        self.renderer = GaussianRenderer(self.config)
        
    def forward(
        self,
        viewpoint_camera: Dict,
        bg_color: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Forward rendering pass."""
        return self.renderer.render(
            viewpoint_camera,
            self.gaussians,
            bg_color
        )
        
    def reconstruct(
        self,
        images: List[torch.Tensor],
        camera_poses: List[Dict],
        num_iterations: int = 30000
    ) -> GaussianModel:
        """Reconstruct 3D scene from multi-view images.
        
        This is a simplified training loop - real implementation would include:
        - Proper optimization with Adam
        - Learning rate scheduling
        - Densification and pruning
        - Loss computation (L1 + SSIM)
        """
        # Initialize from sparse points (simplified)
        points = torch.randn(1000, 3) * 2
        colors = torch.rand(1000, 3)
        self.gaussians.create_from_pcd(points, colors)
        
        # Training loop would go here
        print(f"Training for {num_iterations} iterations...")
        
        return self.gaussians