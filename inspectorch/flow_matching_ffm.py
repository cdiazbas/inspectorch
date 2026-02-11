import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from typing import Optional, List, Union
from . import utils
from . import datasets

# Flow Matching imports (Facebook's library) - REQUIRED
try:
    from flow_matching.path import AffineProbPath
    from flow_matching.solver import ODESolver

    FLOW_MATCHING_AVAILABLE = True
except ImportError:
    raise ImportError(
        "\n" + "=" * 70 + "\n"
        "ERROR: flow_matching library is required for FlowMatchingBackend\n"
        "\n"
        "Install with:\n"
        "  pip install flow_matching\n"
        "\n"
        "Or install from source:\n"
        "  git clone https://github.com/facebookresearch/flow_matching\n"
        "  cd flow_matching && pip install -e .\n" + "=" * 70
    )

# Optional dependency: torchdiffeq (Used for manual log_prob)
try:
    from torchdiffeq import odeint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False

# Import nflows ResidualNet for ResNetFlow architecture
try:
    pass

    NFLOWS_AVAILABLE = True
except ImportError:
    NFLOWS_AVAILABLE = False
    print(
        "Warning: nflows not installed. ResNetFlow architecture will not be available."
    )


# =============================================================================
# Utility Functions (Reused from flowutils.py)
# =============================================================================

dot_dict = utils.dot_dict
nanstd = utils.nanstd
nanvar = utils.nanvar
nume2string = utils.nume2string
GeneralizedPatchedDataset = datasets.GeneralizedPatchedDataset


# =============================================================================
# Velocity Network Architectures
# =============================================================================

# Import all velocity models from shared module
from .velocity_models import (
    VectorFieldAdaMLP,
    VelocityMLPLegacy,
    VelocityResNet,
    VelocityResNetFlow,
    FourierMLP,
)


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

configure_device = utils.configure_device


# =============================================================================
# Flow Matching Density Estimator
# =============================================================================


class FlowMatchingBackend(nn.Module):
    """
    Flow Matching based density estimator for anomaly detection.

    This class provides the same interface as NormalizingFlowBackend from normalizing_flow.py
    but uses Flow Matching (Facebook's library) instead of Normalizing Flows.

    Attributes:
        velocity_model (nn.Module): The velocity network v_theta(t, x)
        prob_path (ProbPath): Probability path for flow matching
        y_mean (torch.Tensor): Mean for data normalization
        y_std (torch.Tensor): Std for data normalization
        training_loss (List[float]): Training loss history
    """

    def __init__(self):
        super().__init__()
        if not FLOW_MATCHING_AVAILABLE:
            raise ImportError(
                "flow_matching is required. Install with:\n  pip install flow_matching"
            )

        self.velocity_model: Optional[nn.Module] = None
        self.prob_path = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None
        self.training_loss: List[float] = []

    def create_flow(
        self,
        input_size: int,
        num_layers: int = 5,
        hidden_features: int = 128,
        scheduler_n: Optional[float] = None,
        architecture: str = "AdaMLP",
        time_embedding_dim: int = 32,
        num_bins: Optional[
            int
        ] = None,  # For API compatibility with flowutils, not used
        activation: nn.Module = nn.GELU(),
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        """
        Creates a flow matching model for density estimation.

        Args:
            input_size (int): Dimensionality of the input data
            num_layers (int): Number of layers/blocks in the velocity network
            hidden_features (int): Number of hidden units
            scheduler_n (float): Polynomial scheduler exponent (3.0 recommended)
            architecture (str): "AdaMLP" (default), "ResNet", "ResNetFlow", or "FourierMLP"
            time_embedding_dim (int): Dimension of time embedding
            num_bins (int): Unused, for API compatibility with flowutils
            activation: Activation function (for ResNetFlow, default: F.relu to match nflows)
            dropout_probability (float): Dropout probability (for ResNetFlow)
            use_batch_norm (bool): Whether to use batch normalization (for ResNetFlow)
        """
        print("Creating Flow Matching model...")

        # Create velocity network
        if architecture.lower() in ["adamlp", "mlp"]:
            # Backward compatibility warning
            if architecture.lower() == "mlp":
                import warnings

                warnings.warn(
                    "architecture='MLP' is deprecated. Use 'AdaMLP' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            # Handle activation type vs instance
            act_type = (
                type(activation) if isinstance(activation, nn.Module) else activation
            )

            self.velocity_model = VectorFieldAdaMLP(
                input_dim=input_size,
                context_dim=0,  # Unconditional by default
                hidden_features=hidden_features,
                num_layers=num_layers,
                time_embedding_dim=time_embedding_dim,
                activation=act_type,
            )
            print("  Using VectorFieldAdaMLP (AdaMLP) architecture")

        elif architecture.lower() == "mlplegacy":
            # Legacy support
            self.velocity_model = VelocityMLPLegacy(
                input_dim=input_size,
                hidden_dim=hidden_features,
                num_layers=num_layers,
                time_embedding_dim=time_embedding_dim,
            )
            print("  Using Legacy MLP architecture")

        elif architecture.lower() == "resnet":
            self.velocity_model = VelocityResNet(
                input_dim=input_size,
                hidden_dim=hidden_features,
                num_blocks=num_layers,
                time_embedding_dim=time_embedding_dim,
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
            print("  Using nflows.nn.nets.ResidualNet architecture")
            if use_batch_norm:
                print("  Batch normalization: enabled")
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
                fourier_features=True,  # or False
                m_freqs=64,
                sigma=1.0,
                tune_beta=False,
                time_embedding_dim=time_embedding_dim,
            )
            print("  Using FourierMLP architecture")
        else:
            raise ValueError(
                f"Unknown architecture: {architecture}. Choose from 'MLP', 'ResNet', or 'ResNetFlow'"
            )

        # Create probability path with polynomial convex scheduler
        from flow_matching.path.scheduler import CondOTScheduler

        self.prob_path = AffineProbPath(scheduler=CondOTScheduler())

    def print_summary(self):
        """
        Prints a summary of the velocity network.
        """
        if self.velocity_model is None:
            print("No model created yet.")
            return

        total_params = sum(
            p.numel() for p in self.velocity_model.parameters() if p.requires_grad
        )
        print(f"Total params to optimize: {total_params:,}")

    def train_flow(
        self,
        train_loader: torch.utils.data.DataLoader,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cpu",
        output_model: Optional[str] = None,
        save_model: bool = False,
        load_existing: bool = False,
        extra_noise: float = 0.0,
        training_mode: str = "forward",  # Options: "forward", "backward", "both"
    ) -> None:
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
                if training_mode == "forward_SBI":
                    # Explicit SBI-style implementation to bypass library abstractions
                    # x0 = noise, x1 = data
                    # t_sample is (batch,)

                    t_expand = t_sample.view(-1, *([1] * (x1.ndim - 1)))

                    # 1. Path: linear interpolation
                    x_t = (1 - t_expand) * x0 + t_expand * x1

                    # 2. Target Velocity: x1 - x0
                    v_target = x1 - x0

                    # 3. Predict
                    # Model takes t with shape (batch,) or (batch, 1) usually
                    t_in = t_sample.unsqueeze(1)  # (batch, 1)
                    vt_pred = active_model(t_in, x_t)

                    # 4. Loss (Standard MSE)
                    loss_f = torch.mean((vt_pred - v_target) ** 2)
                    loss += loss_f

                elif training_mode in ["forward", "both"]:
                    # ProbPath typically assumes x0=noise, x1=data
                    path_f = self.prob_path.sample(x_0=x0, x_1=x1, t=t_sample)
                    vt_pred_f = active_model(path_f.t, path_f.x_t)
                    loss_f = torch.mean((vt_pred_f - path_f.dx_t) ** 2)
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
                    loss_b = torch.mean((vt_pred_b - vt_target) ** 2)

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
        inputs: Union[torch.Tensor, np.ndarray, object],
        dataset_normalization: bool = True,
        batch_size: int = 1000,
        device: str = "cpu",
        solver_method: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        exact_divergence: bool = False,
        step_size: float = 0.01,  # Default to 0.01 (100 steps)
        steps: Optional[int] = None,  # Accept explicit steps count
    ) -> np.ndarray:
        """
        Computes log-probability of inputs using ODE integration with
        instantaneous change of variables. Uses manual torchdiffeq
        implementation for consistency with SBI backend.

        Args:
           steps: Optional override for step_size (step_size = 1/steps)
        """
        if not TORCHDIFFEQ_AVAILABLE:
            print(
                "Warning: torchdiffeq not installed. Falling back to flow_matching ODESolver (if available)."
            )
            # Fallback logic could go here, but for now we enforce consistent behavior or error
            raise ImportError(
                "torchdiffeq is required for the updated log_prob implementation."
            )

        self.velocity_model.eval()
        utils.configure_device(self, device)
        self.velocity_model.to(device)

        # Handle step size logic
        if steps is not None:
            step_size = 1.0 / steps

        # Get data
        apply_mean_std_adjustment = False
        if hasattr(inputs, "normalized_patches"):
            if dataset_normalization:
                data = inputs.normalized_patches()
            else:
                data = inputs.patches
        elif hasattr(inputs, "patches"):
            data = inputs.patches
            apply_mean_std_adjustment = dataset_normalization
        elif isinstance(inputs, (torch.Tensor, np.ndarray)):
            if isinstance(inputs, np.ndarray):
                data = torch.from_numpy(inputs).float()
            else:
                data = inputs
            apply_mean_std_adjustment = dataset_normalization
        else:
            raise ValueError(f"Unknown input type: {type(inputs)}")

        # Data Loading
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        log_probs = []

        # Integration times: 1.0 -> 0.0 (Data -> Noise)
        t_span = torch.tensor([1.0, 0.0], device=device)

        # Handle fixed step solvers or manual steps
        # If solver is fixed step (euler, rk4) we need grid.
        # Even adaptive solvers can benefit from evaluation points if 'steps' is strictly required?
        # Standard ODESolver in library handles "step_size" for fixed.
        # We will use common logic:
        if solver_method in ["euler", "rk4", "midpoint"] or steps is not None:
            num_steps = int(1.0 / step_size) if steps is None else steps
            t_span = torch.linspace(1, 0, num_steps, device=device)

        # Define fused ODE function (Divergence + Velocity)
        class ODEFuncLogProb(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, t, states):
                x = states[0]

                # Compute gradients inside forward to avoid double evaluation
                with torch.set_grad_enabled(True):
                    x.requires_grad_(True)
                    t_expand = t * torch.ones(x.shape[0], 1, device=x.device)
                    # We don't have 'context' in this generic backend usually, but AdaMLP might supports it.
                    # Current FlowMatchingBackend doesn't support generic context passing in log_prob yet.
                    # VectorFieldAdaMLP supports context=None.
                    v = self.net(t_expand, x)

                    if not exact_divergence:  # Hutchinson
                        epsilon = torch.randn_like(x)
                        v_eps = torch.sum(v * epsilon)
                        grad_v_eps = torch.autograd.grad(v_eps, x, create_graph=False)[
                            0
                        ]
                        div = torch.sum(grad_v_eps * epsilon, dim=-1)
                    else:  # Exact
                        div = 0.0
                        for i in range(x.shape[1]):
                            grad_v_i = torch.autograd.grad(
                                v[:, i].sum(), x, create_graph=False, retain_graph=True
                            )[0]
                            div += grad_v_i[:, i]

                return v, -div

        print(f"Computing log-probs on {device} using {solver_method}...")

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing log prob"):
                x_batch = batch[0]

                # Apply normalization if necessary
                if apply_mean_std_adjustment and self.y_mean is not None:
                    # Ensure stats are on device
                    if self.y_mean.device != device:
                        self.y_mean = self.y_mean.to(device)
                        self.y_std = self.y_std.to(device)
                    x_batch = (x_batch.to(device) - self.y_mean) / self.y_std
                else:
                    x_batch = x_batch.to(device)

                batch_n = x_batch.shape[0]
                zeros = torch.zeros(batch_n, device=device)

                func = ODEFuncLogProb(self.velocity_model)

                # Integrate
                state = odeint(
                    func,
                    (x_batch, zeros),
                    t_span,
                    method=solver_method,
                    atol=atol,
                    rtol=rtol,
                )

                x_final = state[0][-1]
                delta_log_p = state[1][-1]

                # Log p(z) (Standard Normal)
                d = x_final.shape[1]
                log_p_z = -0.5 * d * np.log(2 * np.pi) - 0.5 * torch.sum(
                    x_final**2, dim=1
                )

                # log p(x) = log p(z) - delta_log_p
                # (Note: Jacobian trace calculation: div = tr(dv/dx).
                # ODE for log_p: d/dt logp = - div.
                # Integrated: logp(0) - logp(1) = int_1^0 -div dt = - int_0^1 -div dt = int div.
                # logp(1) = logp(0) - int div.
                # Our code returns 'state[1]' which integrates '-div'.
                # So state[1] = int_{1}^{0} -div dt.
                # log p(x) (at t=1) -> transformed to z (at t=0).
                # Formula: log p(x) = log p(z) + int_0^1 div v_t(x_t) dt
                # Our integration is 1->0.
                # Let's trust the SBI implementation which was verified.)

                batch_log_prob = log_p_z - delta_log_p
                log_probs.append(batch_log_prob.cpu().numpy())

        self.velocity_model.to("cpu")
        return np.concatenate(log_probs)

    # log_prob method using original API produces a lot of noise if Hutchinson estimation is used.
    def log_prob_WRONG(
        self,
        inputs: Union[torch.Tensor, np.ndarray, object],
        dataset_normalization: bool = True,
        batch_size: int = 1000,
        device: str = "cpu",
        solver_method: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        exact_divergence: bool = False,
        step_size: float = 0.1,
    ) -> np.ndarray:
        """
        Computes log-probability using the ORIGINAL logic (via external
        flow_matching library).

        Use this to verify against the new torchdiffeq-based log_prob.
        """
        if not FLOW_MATCHING_AVAILABLE:
            raise ImportError(
                "flow_matching library is required for log_prob_original."
            )

        self.velocity_model.eval()
        utils.configure_device(self, device)
        self.velocity_model.to(device)

        # Get data
        if hasattr(inputs, "normalized_patches"):
            if dataset_normalization:
                data = inputs.normalized_patches()
            else:
                data = inputs.patches
        elif hasattr(inputs, "patches"):
            if dataset_normalization:
                data = (inputs.patches - self.y_mean) / self.y_std
            else:
                data = inputs.patches
        else:
            if dataset_normalization and self.y_mean is not None:
                # Ensure stats are on device
                if self.y_mean.device != device:
                    self.y_mean = self.y_mean.to(device)
                    self.y_std = self.y_std.to(device)
                data = (inputs.to(device) - self.y_mean) / self.y_std
            else:
                data = inputs

        log_probs = []

        print(f"Using {device} for log_prob_original computation.")

        # Create ODE solver
        solver = ODESolver(velocity_model=self.velocity_model)

        def log_p0(x):
            return -0.5 * (x.shape[1] * np.log(2 * np.pi) + torch.sum(x**2, dim=1))

        with torch.no_grad():
            for i in tqdm(
                range(0, data.shape[0], batch_size),
                desc="Computing log-prob (original)",
            ):
                batch = data[i : i + batch_size].to(device)

                # Remove any remaining NaNs
                nan_mask = torch.isnan(batch).any(dim=1)
                if nan_mask.any():
                    batch = batch[~nan_mask]

                if batch.shape[0] == 0:
                    continue

                x_0, log_likelihood = solver.compute_likelihood(
                    x_1=batch,
                    log_p0=log_p0,
                    step_size=step_size,
                    method=solver_method,
                    atol=atol,
                    rtol=rtol,
                    exact_divergence=exact_divergence,
                    enable_grad=False,
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
        num_steps=100,
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
                    step_size=1.0 / num_steps,
                    method=solver_method,
                    atol=atol,
                    rtol=rtol,
                    return_intermediates=True,
                )
                x1 = trajectory[-1]
            else:
                # Just get final sample
                x1 = solver.sample(
                    x_init=x0,
                    step_size=0.01,
                    method=solver_method,
                    atol=atol,
                    rtol=rtol,
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
                traj_denorm = [
                    x * self.y_std.to(device) + self.y_mean.to(device)
                    for x in trajectory
                ]
            else:
                traj_denorm = trajectory
            return x1_denorm.cpu(), [x.cpu() for x in traj_denorm]

        return x1_denorm.cpu()

    def plot_train_loss(self, show_plot=False, save_path=None):
        """
        Plots the training loss over epochs.
        """
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
    plt.ylabel(r"FM Loss: $\|\mathbf{v}_\theta - \mathbf{u}_t\|^2$")

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
    Checks the latent space by transforming data towards noise (N(0,1)) using
    direct integration.

    Integrates from t=1 (data) to t=0 (noise) using the flow_matching
    solver. If your ODESolver API allows t_span, use sample with
    t_span=[1.0, 0.0]. Otherwise, use compute_likelihood (which
    internally integrates backwards).
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
        for i in tqdm(
            range(0, total_samples, batch_size), desc="Transforming to noise"
        ):
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
                x_traj = odeint(
                    velocity_fn, x1_, t_span, method="dopri5", atol=1e-5, rtol=1e-5
                )
                x0 = x_traj[-1]
            else:
                # Fallback: use solver.sample with time_grid (not t_span)
                time_grid = [1.0, 0.0]
                x_traj = solver.sample(
                    x_init=x1, time_grid=time_grid, method="dopri5", step_size=0.01
                )
                if hasattr(x_traj, "shape") and x_traj.shape[0] > x1.shape[0]:
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
    print("\n=== Latent Space Statistics ===")
    print(f"Mean (should be ≈ 0.0): {z_mean[: min(5, len(z_mean))]}")
    print(f"Std  (should be ≈ 1.0): {z_std[: min(5, len(z_std))]}")

    # Check for large deviations
    mean_deviation = np.abs(z_mean).max()
    std_deviation = np.abs(z_std - 1.0).max()
    if mean_deviation > 0.5 or std_deviation > 0.5:
        print("⚠ Warning: Latent distribution deviates significantly from N(0,1)")
        print(f"  Max |mean|: {mean_deviation:.3f}, Max |std-1|: {std_deviation:.3f}")
        print(
            "  This may indicate: 1) model not fully trained, 2) ODE integration errors"
        )
    else:
        print("✓ Latent distribution is close to N(0,1)")

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
        ax.hist(z[:, var], bins=100, density=True, alpha=0.5, label="Output")
        ax.set_title(f"Latent Variable {var}")
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Density")

        # Plot standard normal reference
        x_range = np.linspace(z[:, var].min(), z[:, var].max(), 100)
        ax.plot(
            x_range,
            norm.pdf(x_range, 0, 1),
            "k--",
            linewidth=2,
            label=r"$\mathcal{N}(0,1)$",
        )
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    velocity_model.to("cpu")
