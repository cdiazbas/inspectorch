"""
Velocity field models for flow matching.

This module contains neural network architectures for learning
velocity fields in continuous normalizing flows and flow matching.

All flow matching backends should import from this shared module
to avoid code duplication and ensure consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Optional dependency for ResNetFlow architecture
try:
    from nflows.nn import nets
    NFLOWS_AVAILABLE = True
except ImportError:
    NFLOWS_AVAILABLE = False


# =============================================================================
# Time Embeddings
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding as used in Vaswani et al. (2017)."""
    
    def __init__(self, embed_dim: int = 16, max_freq: float = 1000.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embedding dimension must be even")
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(max_freq) / embed_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
            
        time_embedding = torch.zeros(t.shape[:-1] + (self.embed_dim,), device=t.device)
        time_embedding[..., 0::2] = torch.sin(t * self.div_term)
        time_embedding[..., 1::2] = torch.cos(t * self.div_term)
        return time_embedding


class RandomFourierTimeEmbedding(nn.Module):
    """Gaussian random features for encoding time steps."""
    
    def __init__(self, embed_dim: int = 100, scale: float = 30.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        
        times_proj = t * self.W[None, :] * 2 * math.pi
        embedding = torch.cat([torch.sin(times_proj), torch.cos(times_proj)], dim=-1)
        return embedding


# =============================================================================
# AdaMLP Components
# =============================================================================

class AdaMLPBlock(nn.Module):
    """Residual MLP block with adaptive layer norm for conditioning."""
    
    def __init__(
        self,
        hidden_features: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        activation: type[nn.Module] = nn.GELU
    ):
        super().__init__()
        self.ada_ln = nn.Sequential(
            nn.Linear(cond_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 3 * hidden_features),
        )
        # Initialize last layer to zero for identity start
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

        self.block = nn.Sequential(
            nn.Linear(hidden_features, hidden_features * mlp_ratio),
            activation(),
            nn.Linear(hidden_features * mlp_ratio, hidden_features),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond shape: (batch, cond_dim)
        shift, scale, gate = self.ada_ln(cond).chunk(3, dim=-1)
        gate = gate + 1.0
        y = (scale + 1) * x + shift
        y = self.block(y)
        return x + gate * y


class GlobalEmbeddingMLP(nn.Module):
    """Computes global embedding from time and context."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        time_embedding_dim: int = 32,
        hidden_features: int = 100,
        time_emb_type: str = "sinusoidal",
        activation: type[nn.Module] = nn.GELU
    ):
        super().__init__()
        if time_emb_type == "sinusoidal":
            self.time_emb = SinusoidalTimeEmbedding(embed_dim=time_embedding_dim)
        else:
            self.time_emb = RandomFourierTimeEmbedding(embed_dim=time_embedding_dim)

        # Input to MLP is concatenated time_emb + context
        self.input_layer = nn.Linear(time_embedding_dim + input_dim, hidden_features)
        
        self.mlp = nn.Sequential(
            activation(),
            nn.Linear(hidden_features, hidden_features),
            activation(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.output_layer = nn.Linear(hidden_features, output_dim)

    def forward(self, t: torch.Tensor, x_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_emb(t)  # (batch, time_embedding_dim)
        
        if x_emb is not None:
            # Flatten or ensure shape compatibility
            if x_emb.ndim > 2:
                x_emb = x_emb.view(x_emb.shape[0], -1) 
            cond_emb = torch.cat([t_emb, x_emb], dim=-1)
        else:
            cond_emb = t_emb
            
        h = self.input_layer(cond_emb)
        h = self.mlp(h)
        return self.output_layer(h)


# =============================================================================
# Main Velocity Field Architectures
# =============================================================================

class VectorFieldAdaMLP(nn.Module):
    """
    Adaptive MLP Vector Field Network (Recommended).
    
    Uses AdaMLP blocks where time/context modulate features via AdaLN.
    This is the default architecture for all flow matching backends.
    """
    
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_features: int = 128,
        num_layers: int = 5,
        time_embedding_dim: int = 32,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        
        # Global embedding network (Time + Context -> Embedding)
        self.cond_emb_dim = hidden_features 
        
        self.global_mlp = GlobalEmbeddingMLP(
            input_dim=context_dim,
            output_dim=self.cond_emb_dim,
            time_embedding_dim=time_embedding_dim,
            hidden_features=hidden_features,
            activation=activation
        )

        # Main Network
        self.layers = nn.ModuleList()
        
        # Input projection
        self.layers.append(nn.Linear(input_dim, hidden_features))

        # AdaMLP Blocks
        for _ in range(num_layers):
            self.layers.append(
                AdaMLPBlock(
                    hidden_features=hidden_features,
                    cond_dim=self.cond_emb_dim,
                    activation=activation
                )
            )

        # Output projection
        self.final_layer = nn.Linear(hidden_features, input_dim) 
        # Initialize output to zero for stability
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            t: Time (batch,) or (batch, 1)
            x: State (batch, input_dim)
            context: Context (batch, context_dim)
        """
        # Get conditioning embedding
        cond_emb = self.global_mlp(t, x_emb=context)  # (batch, cond_emb_dim)
        
        # Forward pass
        h = x
        h = self.layers[0](h)  # Input projection
        
        for layer in self.layers[1:]:
            h = layer(h, cond_emb)
            
        return self.final_layer(h)


class VelocityMLPLegacy(nn.Module):
    """
    Legacy simple MLP velocity network (for backward compatibility).
    
    Note: VectorFieldAdaMLP is recommended over this architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embedding_dim: int = 32
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
        )

        # Main velocity network
        layers = []
        layers.append(nn.Linear(input_dim + time_embedding_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        """
        Args:
            t: Time tensor of shape (batch_size,) or (batch_size, 1) or scalar
            x: Data tensor of shape (batch_size, input_dim)
        Returns:
            Velocity vector of shape (batch_size, input_dim)
        """
        # Handle different time input formats
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        if t.dim() == 0:  # Scalar
            t = t.unsqueeze(0).expand(x.shape[0])

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Embed time
        t_emb = self.time_mlp(t)

        # Concatenate and pass through network
        tx = torch.cat([x, t_emb], dim=-1)
        return self.net(tx)


class VelocityResNet(nn.Module):
    """ResNet-style velocity network for flow matching."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        time_embedding_dim: int = 32
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim + time_embedding_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, t, x):
        """
        Args:
            t: Time tensor of shape (batch_size,) or (batch_size, 1) or scalar
            x: Data tensor of shape (batch_size, input_dim)
        Returns:
            Velocity vector of shape (batch_size, input_dim)
        """
        # Handle different time input formats
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        if t.dim() == 0:  # Scalar
            t = t.unsqueeze(0).expand(x.shape[0])

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Embed time
        t_emb = self.time_mlp(t)

        # Concatenate and project
        tx = torch.cat([x, t_emb], dim=-1)
        h = F.silu(self.input_proj(tx))

        # Apply residual blocks
        for block in self.blocks:
            h = block(h)

        # Output projection
        return self.output_proj(h)


class ResidualBlock(nn.Module):
    """Residual block for VelocityResNet."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.net(x)


class VelocityResNetFlow(nn.Module):
    """
    Velocity network using nflows.nn.nets.ResidualNet.
    
    This uses the same ResidualNet architecture from nflows that is used
    in normalizing flows, adapted for flow matching with time conditioning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        time_embedding_dim: int = 32,
        activation=F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        if not NFLOWS_AVAILABLE:
            raise ImportError(
                "nflows is required for ResNetFlow architecture. Install with:\n"
                "  pip install nflows"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.time_embedding_dim = time_embedding_dim

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
        )

        # Use nflows ResidualNet architecture
        self.resnet = nets.ResidualNet(
            in_features=input_dim + time_embedding_dim,
            out_features=input_dim,
            hidden_features=hidden_dim,
            num_blocks=num_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, t, x):
        """
        Args:
            t: Time tensor of shape (batch_size,) or (batch_size, 1) or scalar
            x: Data tensor of shape (batch_size, input_dim)
        Returns:
            Velocity vector of shape (batch_size, input_dim)
        """
        # Handle different time input formats
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        if t.dim() == 0:  # Scalar
            t = t.unsqueeze(0).expand(x.shape[0])

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Embed time
        t_emb = self.time_mlp(t)

        # Concatenate input and time embedding
        tx = torch.cat([x, t_emb], dim=-1)

        # Pass through nflows ResidualNet
        v = self.resnet(tx)

        return v


class FourierMLP(nn.Module):
    """MLP with Fourier features for velocity field."""
    
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_resnet_blocks: int = 3,
        num_layers_per_block: int = 2,
        dim_hidden: int = 50,
        activation=nn.GELU(),
        fourier_features: bool = False,
        m_freqs: int = 100,
        sigma: float = 10,
        tune_beta: bool = False,
        time_embedding_dim: int = 32,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            activation,
            nn.Linear(time_embedding_dim, time_embedding_dim),
            activation,
        )
        self.input_dim = dim_in
        self.fourier_features = fourier_features
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.activation = activation
        self.tune_beta = tune_beta
        self.sigma = sigma
        num_neurons = dim_hidden
        
        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(
                torch.ones(self.num_resnet_blocks, self.num_layers_per_block)
            )
        else:
            self.beta0 = torch.ones(1, 1)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block)

        input_dim = dim_in + time_embedding_dim
        if fourier_features:
            input_dim = 2 * m_freqs + dim_in + time_embedding_dim
            n_param = dim_in
            self.B = nn.Parameter(
                torch.randn(n_param, m_freqs) * sigma, requires_grad=False
            )

        self.first = nn.Linear(input_dim, num_neurons)
        self.resblocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(num_neurons, num_neurons)
                        for _ in range(num_layers_per_block)
                    ]
                )
                for _ in range(num_resnet_blocks)
            ]
        )
        self.last = nn.Linear(num_neurons, dim_out)

    def forward(self, t, x):
        # Embed time and concatenate
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t)
        x = torch.cat([x, t_emb], dim=-1)

        xx = x.clone()
        if x.device != self.beta0.device:
            self.beta0 = self.beta0.to(x.device)
        if self.fourier_features and self.B.device != x.device:
            self.B = self.B.to(x.device)

        if self.fourier_features:
            cosx = torch.cos(torch.matmul(xx[:, : self.B.shape[0]], self.B))
            sinx = torch.sin(torch.matmul(xx[:, : self.B.shape[0]], self.B))
            x = torch.cat((cosx, sinx, xx), dim=1)
            x = self.activation(self.beta0 * self.first(x))
        else:
            x = self.activation(self.beta0 * self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](x))
            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))
            x = z + x
        out = self.last(x)
        return out
