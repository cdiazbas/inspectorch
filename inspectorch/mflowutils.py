import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from einops import rearrange
import scipy.optimize
from . import genutils as gu

# Flow Matching imports (Facebook's library)
try:
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import PolynomialConvexScheduler
    from flow_matching.solver import ODESolver
    FLOW_MATCHING_AVAILABLE = True
except ImportError:
    FLOW_MATCHING_AVAILABLE = False
    print("Warning: flow_matching not installed. Install with:")
    print("  pip install flow_matching")

# Import nflows ResidualNet for ResNetFlow architecture
try:
    from nflows.nn import nets
    NFLOWS_AVAILABLE = True
except ImportError:
    NFLOWS_AVAILABLE = False
    print("Warning: nflows not installed. ResNetFlow architecture will not be available.")


# =============================================================================
# Utility Functions (Reused from flowutils.py)
# =============================================================================

dot_dict = gu.dot_dict
nanstd = gu.nanstd
nanvar = gu.nanvar
nume2string = gu.nume2string
GeneralizedPatchedDataset = gu.GeneralizedPatchedDataset


# =============================================================================
# Velocity Network Architectures
# =============================================================================

class VelocityMLP(nn.Module):
    """
    MLP velocity network for flow matching: v_theta(t, x)
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, time_embedding_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU()
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
    """
    ResNet-style velocity network for flow matching: v_theta(t, x)
    """
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, time_embedding_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU()
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim + time_embedding_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
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
    """Residual block for VelocityResNet"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return x + self.net(x)



class VelocityResNetFlow(nn.Module):
    """
    Velocity network using nflows.nn.nets.ResidualNet for flow matching.
    
    This directly uses the same ResidualNet architecture from nflows that is used
    in normalizing flows, but adapted for flow matching with time conditioning.
    This ensures a fair comparison between normalizing flows and flow matching.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_blocks=2,
        time_embedding_dim=32,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
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
            nn.SiLU()
        )
        
        # Use nflows ResidualNet architecture
        # The ResidualNet takes (input + time_embedding) and outputs to hidden_dim
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
    def __init__(
        self,
        dim_in,
        dim_out,
        num_resnet_blocks=3,
        num_layers_per_block=2,
        dim_hidden=50,
        activation=nn.GELU(),
        fourier_features=False,
        m_freqs=100,
        sigma=10,
        tune_beta=False,
        time_embedding_dim=32,
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
            cosx = torch.cos(torch.matmul(xx[:, :self.B.shape[0]], self.B))
            sinx = torch.sin(torch.matmul(xx[:, :self.B.shape[0]], self.B))
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


# =============================================================================
# Velocity Model Wrapper for DataParallel
# =============================================================================

class VelocityModelWrapper(nn.Module):
    """
    Wrapper for velocity models to be compatible with DataParallel.
    Similar to FlowLogProbWrapper in flowutils.py
    """
    def __init__(self, velocity_model):
        super().__init__()
        self.velocity_model = velocity_model
    
    def forward(self, t, x):
        return self.velocity_model(t, x)


# =============================================================================
# Device Configuration (from flowutils.py)
# =============================================================================

configure_device = gu.configure_device


# =============================================================================
# Flow Matching Density Estimator
# =============================================================================

class FlowMatching_Density_estimator(nn.Module):
    """
    Flow Matching based density estimator for anomaly detection.
    
    This class provides the same interface as Density_estimator from flowutils.py
    but uses Flow Matching (Facebook's library) instead of Normalizing Flows.
    
    Attributes:
        velocity_model (nn.Module): The velocity network v_theta(t, x)
        prob_path (ProbPath): Probability path for flow matching
        y_mean (torch.Tensor): Mean for data normalization
        y_std (torch.Tensor): Std for data normalization
        training_loss (list): Training loss history
    """
    
    def __init__(self):
        super().__init__()
        if not FLOW_MATCHING_AVAILABLE:
            raise ImportError(
                "flow_matching is required. Install with:\n"
                "  pip install flow_matching"
            )
        
        self.velocity_model = None
        self.prob_path = None
        self.y_mean = None
        self.y_std = None
        self.training_loss = []
    
    def create_flow(
        self,
        input_size,
        num_layers=5,
        hidden_features=128,
        scheduler_n=None,
        architecture="MLP",
        time_embedding_dim=32,
        num_bins=None,  # For API compatibility with flowutils, not used
        activation=nn.GELU(),
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        """
        Creates a flow matching model for density estimation.
        
        Args:
            input_size (int): Dimensionality of the input data
            num_layers (int): Number of layers/blocks in the velocity network
            hidden_features (int): Number of hidden units
            scheduler_n (float): Polynomial scheduler exponent (3.0 recommended)
            architecture (str): "MLP", "ResNet", or "ResNetFlow" (uses nflows.nn.nets.ResidualNet)
            time_embedding_dim (int): Dimension of time embedding
            num_bins (int): Unused, for API compatibility with flowutils
            activation: Activation function (for ResNetFlow, default: F.relu to match nflows)
            dropout_probability (float): Dropout probability (for ResNetFlow)
            use_batch_norm (bool): Whether to use batch normalization (for ResNetFlow)
        """
        print(f"Creating Flow Matching model...")
        
        # Create velocity network
        if architecture.lower() == "mlp":
            self.velocity_model = VelocityMLP(
                input_dim=input_size,
                hidden_dim=hidden_features,
                num_layers=num_layers,
                time_embedding_dim=time_embedding_dim
            )
        elif architecture.lower() == "resnet":
            self.velocity_model = VelocityResNet(
                input_dim=input_size,
                hidden_dim=hidden_features,
                num_blocks=num_layers,
                time_embedding_dim=time_embedding_dim
            )
        elif architecture.lower() == "resnetflow":
            self.velocity_model = VelocityResNetFlow(
                input_dim=input_size,
                hidden_dim=hidden_features,
                num_blocks=num_layers,
                time_embedding_dim=time_embedding_dim,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            )
            print(f"  Using nflows.nn.nets.ResidualNet architecture")
            if use_batch_norm:
                print(f"  Batch normalization: enabled")
            if dropout_probability > 0:
                print(f"  Dropout probability: {dropout_probability}")
        elif architecture.lower() == "fouriermlp":
            self.velocity_model = FourierMLP(
                dim_in=input_size,
                dim_out=input_size,
                num_resnet_blocks=num_layers,
                num_layers_per_block=2,  # or whatever you want
                dim_hidden=hidden_features,
                activation=activation,
                fourier_features=True,   # or False
                m_freqs=64,
                sigma=1.0,
                tune_beta=False,
                time_embedding_dim=time_embedding_dim,
            )
            print(f"  Using FourierMLP architecture")
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Choose from 'MLP', 'ResNet', or 'ResNetFlow'")
        
        # Create probability path with polynomial convex scheduler
        from flow_matching.path.scheduler import CondOTScheduler
        self.prob_path = AffineProbPath(scheduler=CondOTScheduler())
        

    def print_summary(self):
        """Prints a summary of the velocity network."""
        if self.velocity_model is None:
            print("No model created yet.")
            return
        
        total_params = sum(p.numel() for p in self.velocity_model.parameters() if p.requires_grad)
        print(f"Total params to optimize: {total_params:,}")
    
    def train_flow(
        self,
        train_loader,
        learning_rate=1e-3,
        num_epochs=100,
        device="cpu",
        output_model=None,
        save_model=False,
        load_existing=False,
        extra_noise=0.0,
        training_mode="both"  # Options: "forward", "backward", "both"
    ):
        """
        Trains the flow matching model.
        
        Args:
            train_loader: PyTorch DataLoader
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            device: Device to train on ('cpu', 'cuda', 'cuda:0', 'cuda:0,1,2', etc.)
            output_model: Path to save model
            save_model: Whether to save the model
            load_existing: Whether to load existing model
            extra_noise: Additional noise to add during training (for regularization)
            training_mode: "forward" (x0->x1), "backward" (x1->x0), "both" (average)
        """
        if train_loader is not None:
            self.y_mean = train_loader.dataset.y_mean
            self.y_std = train_loader.dataset.y_std
        
        # Load existing model if requested
        if num_epochs == 0 and load_existing and output_model is not None:
            try:
                self.velocity_model.load_state_dict(
                    torch.load(output_model, map_location="cpu")
                )
                print(f"Loaded existing model from {output_model}")
                return
            except FileNotFoundError:
                print(f"No existing model found at {output_model}")
                return
        
        # Wrap model and configure device
        model_wrapper = VelocityModelWrapper(self.velocity_model)
        active_model, effective_primary_device = configure_device(model_wrapper, device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(active_model.parameters(), lr=learning_rate)
        
        # Normalization tensors
        y_mean = self.y_mean.to(effective_primary_device)
        y_std = self.y_std.to(effective_primary_device)
        
        time0 = time.time()
        
        active_model.train()
        
        for epoch in range(1, num_epochs + 1):
            train_loss = []
            t = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{num_epochs}",
            )
            
            for batch_idx, batch in t:
                if not isinstance(batch, torch.Tensor):
                    batch = torch.from_numpy(batch).float()
                else:
                    batch = batch.float()
                
                # Remove NaN samples
                batch = batch[~torch.isnan(batch).any(dim=1)]
                if batch.shape[0] == 0:
                    continue
                
                # Normalize and move to device
                x1 = (batch.to(effective_primary_device) - y_mean) / y_std
                
                # Add extra noise if specified
                if extra_noise is not None and extra_noise > 0:
                    noise = torch.normal(
                        0, extra_noise, size=x1.shape, device=effective_primary_device
                    )
                    x1 = x1 + noise
                
                # Sample noise (x0) - standard Gaussian
                x0 = torch.randn_like(x1)
                
                # Sample time uniformly in [0, 1]
                t_sample = torch.rand(x1.shape[0], device=effective_primary_device)
                
                # Get path sample: x_t and target velocity dx_t
                loss = 0.0

                # --- FORWARD DIRECTION (Noise to Data) ---
                if training_mode in ["forward", "both"]:
                    # ProbPath typically assumes x0=noise, x1=data
                    path_f = self.prob_path.sample(x_0=x0, x_1=x1, t=t_sample)
                    vt_pred_f = active_model(path_f.t, path_f.x_t)
                    loss_f = torch.mean(((vt_pred_f - path_f.dx_t) / y_std) ** 2)
                    loss += loss_f

                # --- BACKWARD DIRECTION (Data to Noise) ---
                if training_mode in ["backward", "both"]:
                    # We invert relationships: x0=data, x1=noise
                    # t_sample flows same way, but direction is opposite
                    path_b = self.prob_path.sample(x_0=x1, x_1=x0, t=t_sample)
                    
                    # Model v_theta is trained to be the FORWARD velocity (Noise -> Data).
                    # When sampling backward (Data -> Noise path), the physical velocity path_b.dx_t points to Noise.
                    # The forward velocity should point to Data, so it is -path_b.dx_t.
                    # Also, the time parameter for the forward model should be reversed w.r.t the backward path time.
                    # Backward path at t corresponds to Forward path at 1-t.
                    
                    t_fwd = 1.0 - path_b.t
                    vt_target = -path_b.dx_t
                    
                    vt_pred_b = active_model(t_fwd, path_b.x_t)
                    loss_b = torch.mean(((vt_pred_b - vt_target) / y_std) ** 2)
                    
                    if training_mode == "both":
                        loss = (loss + loss_b) / 2
                    else:
                        loss += loss_b

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
                t.set_postfix_str("Loss: {0:2.2f}".format(train_loss[-1]))
            
            current_epoch_avg_loss = np.mean(np.array(train_loss))
            self.training_loss.append(current_epoch_avg_loss)
            
            # Update tqdm postfix with average loss
            t.set_postfix_str(f"Avg: {current_epoch_avg_loss:.4f}")
            
            # Save intermediate model
            if save_model and output_model is not None:
                torch.save(self.velocity_model.state_dict(), output_model)
        
        print(f"Completed training in {(time.time() - time0) / 60.0:2.2f} minutes.")
        
        # Move back to CPU
        self.velocity_model.to("cpu")
        
        # Final save
        if save_model and output_model is not None:
            torch.save(self.velocity_model.state_dict(), output_model)
            print(f"Model saved to {output_model}")
    
    def log_prob(
        self,
        inputs,
        dataset_normalization=True,
        batch_size=1000,
        device="cpu",
        solver_method="dopri5",
        atol=1e-5,
        rtol=1e-5,
        exact_divergence=False,
        step_size=0.1,
    ):
        """
        Computes log-probability of inputs using ODE integration with
        instantaneous change of variables.
        
        Args:
            inputs: Dataset or tensor
            dataset_normalization: Whether to apply dataset normalization
            batch_size: Batch size for processing
            device: Device to use
            solver_method: ODE solver ('dopri5', 'rk4', 'euler', 'midpoint')
            atol: Absolute tolerance for ODE solver
            rtol: Relative tolerance for ODE solver
            exact_divergence: If True, use exact divergence (slow). If False, use Hutchinson estimator (fast)
        
        Returns:
            numpy array of log probabilities
        """
        self.velocity_model.eval()
        self.velocity_model.to(device)
        
        # Get data
        if hasattr(inputs, 'normalized_patches'):
            if dataset_normalization:
                data = inputs.normalized_patches()
            else:
                data = inputs.patches
        elif hasattr(inputs, 'patches'):
            if dataset_normalization:
                data = (inputs.patches - self.y_mean) / self.y_std
            else:
                data = inputs.patches
        else:
            # inputs is a tensor
            if dataset_normalization and self.y_mean is not None:
                data = (inputs - self.y_mean) / self.y_std
            else:
                data = inputs
        
        log_probs = []
        
        print(f"Using {device} for log_prob computation.")
        
        # Create ODE solver
        solver = ODESolver(velocity_model=self.velocity_model)
        
        # Prior log probability (standard Gaussian)
        def log_p0(x):
            return -0.5 * (x.shape[1] * np.log(2 * np.pi) + torch.sum(x ** 2, dim=1))
        
        with torch.no_grad():
            for i in tqdm(range(0, data.shape[0], batch_size), desc="Computing log-prob"):
                batch = data[i:i+batch_size].to(device)
                
                # Remove any remaining NaNs
                nan_mask = torch.isnan(batch).any(dim=1)
                if nan_mask.any():
                    batch = batch[~nan_mask]
                
                if batch.shape[0] == 0:
                    continue
                
                # Compute likelihood using flow_matching's built-in method
                x_0, log_likelihood = solver.compute_likelihood(
                    x_1=batch,
                    log_p0=log_p0,
                    step_size=step_size,
                    method=solver_method,
                    atol=atol,
                    rtol=rtol,
                    exact_divergence=exact_divergence,
                    enable_grad=False
                )
                
                log_probs.append(log_likelihood.cpu().numpy())
        
        self.velocity_model.to("cpu")
        
        return np.concatenate(log_probs)
    
    def sample(
        self,
        num_samples,
        device="cpu",
        solver_method="dopri5",
        atol=1e-5,
        rtol=1e-5,
        return_trajectory=False,
        num_steps=100
    ):
        """
        Generates samples by solving the ODE forward from noise to data.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to use
            solver_method: ODE solver
            atol: Absolute tolerance
            rtol: Relative tolerance
            return_trajectory: Whether to return full trajectory
            num_steps: Number of time steps for trajectory
        
        Returns:
            Generated samples (and optionally trajectory)
        """
        self.velocity_model.eval()
        self.velocity_model.to(device)
        
        # Sample from base distribution (standard Gaussian)
        x0 = torch.randn(num_samples, self.velocity_model.input_dim, device=device)
        
        # Create ODE solver
        solver = ODESolver(velocity_model=self.velocity_model)
        
        with torch.no_grad():
            if return_trajectory:
                # Sample with intermediate steps
                trajectory = solver.sample(
                    x_init=x0,
                    step_size=1.0/num_steps,
                    method=solver_method,
                    atol=atol,
                    rtol=rtol,
                    return_intermediates=True
                )
                x1 = trajectory[-1]
            else:
                # Just get final sample
                x1 = solver.sample(
                    x_init=x0,
                    step_size=0.01,
                    method=solver_method,
                    atol=atol,
                    rtol=rtol
                )
        
        # Denormalize
        if self.y_mean is not None and self.y_std is not None:
            x1_denorm = x1 * self.y_std.to(device) + self.y_mean.to(device)
        else:
            x1_denorm = x1
        
        self.velocity_model.to("cpu")
        
        if return_trajectory:
            # Denormalize entire trajectory
            if self.y_mean is not None and self.y_std is not None:
                traj_denorm = [x * self.y_std.to(device) + self.y_mean.to(device) for x in trajectory]
            else:
                traj_denorm = trajectory
            return x1_denorm.cpu(), [x.cpu() for x in traj_denorm]
        
        return x1_denorm.cpu()
    
    def plot_train_loss(self, show_plot=False, save_path=None):
        """Plots the training loss over epochs."""
        plot_train_loss(self.training_loss, show_plot, save_path)
    
    def check_variables(
        self,
        train_loader,
        plot_variables=[0, 1],
        figsize=(8, 4),
        device="cpu",
        batch_size=50000,
        rel_size=0.1,
    ):
        """
        Checks latent space by transforming samples backwards to noise.
        """
        check_latent_space(
            self.velocity_model,
            train_loader,
            plot_variables,
            figsize,
            device,
            batch_size,
            rel_size,
        )


# =============================================================================
# Helper Functions
# =============================================================================

def plot_train_loss(train_loss_avg, show_plot=False, save_path=None):
    """
    Plots the training loss over the epochs.
    """
    plt.figure()  # Create a new figure
    plt.plot(train_loss_avg, ".-")
    if len(train_loss_avg) > 1:
        output_title_latex = r"${:}$".format(nume2string(train_loss_avg[-1]))
        plt.title("Final loss: " + output_title_latex)
    plt.minorticks_on()
    plt.xlabel("Epoch")
    plt.ylabel("Flow Matching Loss")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    if show_plot:
        plt.show()


def check_latent_space(
    velocity_model,
    train_loader,
    plot_variables=[0, 1],
    figsize=(8, 4),
    device="cpu",
    batch_size=50000,
    rel_size=0.1,
):
    """
    Checks the latent space by transforming data towards noise (N(0,1)) using direct integration.
    Integrates from t=1 (data) to t=0 (noise) using the flow_matching solver.
    If your ODESolver API allows t_span, use sample with t_span=[1.0, 0.0].
    Otherwise, use compute_likelihood (which internally integrates backwards).
    """
    velocity_model.eval()
    velocity_model.to(device)
    
    y_mean = train_loader.dataset.y_mean.to(device)
    y_std = train_loader.dataset.y_std.to(device)
    
    dataset_size = len(train_loader.dataset)
    total_samples = int(rel_size * dataset_size)
    
    # Sample random indices
    random_indices = np.random.randint(0, dataset_size, size=total_samples)
    
    z_batches = []
    
    # Try to import torchdiffeq for direct ODE integration
    try:
        from torchdiffeq import odeint
        _has_odeint = True
    except ImportError:
        _has_odeint = False

    solver = ODESolver(velocity_model=velocity_model)

    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size), desc="Transforming to noise"):
            end_idx = min(i + batch_size, total_samples)
            batch_indices = random_indices[i:end_idx]
            batch = train_loader.dataset.patches[batch_indices]
            nan_mask = torch.isnan(batch).any(dim=1)
            batch = batch[~nan_mask]
            if batch.shape[0] == 0:
                continue
            batch = batch.to(device)
            x1 = (batch - y_mean) / y_std

            if _has_odeint:
                # Use torchdiffeq.odeint for direct ODE integration from t=1 to t=0
                def velocity_fn(t, x):
                    # t: scalar tensor, x: (batch, dim)
                    return velocity_model(t.expand(x.shape[0]), x)
                t_span = torch.tensor([1.0, 0.0], device=x1.device)
                x1_ = x1
                # odeint expects shape (batch, dim), returns (2, batch, dim)
                x_traj = odeint(velocity_fn, x1_, t_span, method='dopri5', atol=1e-5, rtol=1e-5)
                x0 = x_traj[-1]
            else:
                # Fallback: use solver.sample with time_grid (not t_span)
                time_grid = [1.0, 0.0]
                x_traj = solver.sample(
                    x_init=x1,
                    time_grid=time_grid,
                    method='dopri5',
                    step_size=0.01
                )
                if hasattr(x_traj, 'shape') and x_traj.shape[0] > x1.shape[0]:
                    x0 = x_traj[-1]
                else:
                    x0 = x_traj

            z_batches.append(x0.cpu().numpy())
    
    if not z_batches:
        print("Warning: No valid samples to plot")
        velocity_model.to("cpu")
        return
    
    z = np.concatenate(z_batches)
    
    # Validate that the inverted distribution is close to N(0,1)
    z_mean = np.mean(z, axis=0)
    z_std = np.std(z, axis=0)
    print(f"\n=== Latent Space Statistics ===")
    print(f"Mean (should be ≈ 0.0): {z_mean[:min(5, len(z_mean))]}")
    print(f"Std  (should be ≈ 1.0): {z_std[:min(5, len(z_std))]}")
    
    # Check for large deviations
    mean_deviation = np.abs(z_mean).max()
    std_deviation = np.abs(z_std - 1.0).max()
    if mean_deviation > 0.5 or std_deviation > 0.5:
        print(f"⚠ Warning: Latent distribution deviates significantly from N(0,1)")
        print(f"  Max |mean|: {mean_deviation:.3f}, Max |std-1|: {std_deviation:.3f}")
        print(f"  This may indicate: 1) model not fully trained, 2) ODE integration errors")
    else:
        print(f"✓ Latent distribution is close to N(0,1)")
    
    # Plot
    if len(plot_variables) > z.shape[1]:
        plot_variables = list(range(z.shape[1]))
    elif len(plot_variables) == 0:
        plot_variables = list(range(z.shape[1]))
    
    fig, axes = plt.subplots(1, len(plot_variables), figsize=figsize, sharey=True)
    if len(plot_variables) == 1:
        axes = [axes]
    
    for i, var in enumerate(plot_variables):
        ax = axes[i]
        ax.hist(z[:, var], bins=100, density=True, alpha=0.5, label='Output')
        ax.set_title(f"Latent Variable {var}")
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Density")
        
        # Plot standard normal reference
        x_range = np.linspace(z[:, var].min(), z[:, var].max(), 100)
        ax.plot(x_range, norm.pdf(x_range, 0, 1), 'k--', linewidth=2, label=r'$\mathcal{N}(0,1)$')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    velocity_model.to("cpu")