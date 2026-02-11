import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Tuple
from tqdm import tqdm
from . import utils

# Required dependency: torchdiffeq
try:
    from torchdiffeq import odeint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    raise ImportError(
        "\n" + "=" * 70 + "\n"
        "ERROR: torchdiffeq is required for FlowMatchingSBIBackend\n"
        "\n"
        "Install with:\n"
        "  pip install torchdiffeq\n"
        "\n"
        "Or install from source:\n"
        "  git clone https://github.com/rtqichen/torchdiffeq\n"
        "  cd torchdiffeq && pip install -e .\n" + "=" * 70
    )

# Import velocity models from shared module


# =============================================================================
# Backend Implementation
# =============================================================================


class FlowMatchingSBIBackend(nn.Module):
    """
    Flow Matching Backend using SBI-inspired architecture (AdaMLP) and pure
    PyTorch CFM loss implementation (no external flow_matching dependency).
    """

    def __init__(self):
        super().__init__()
        self.velocity_model: Optional[nn.Module] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None
        self.training_loss: List[float] = []

    def create_flow(
        self,
        input_size: int,
        num_layers: int = 5,
        hidden_features: int = 128,
        time_embedding_dim: int = 32,
        context_dim: int = 0,
        architecture: str = "AdaMLP",
        activation: nn.Module = nn.GELU(),
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        **kwargs,
    ) -> None:
        """
        Creates the flow matching model.
        """
        print(
            f"Creating Flow Matching (SBI-style) model with architecture: {architecture}..."
        )

        # Import all models from shared module
        from .velocity_models import (
            VectorFieldAdaMLP,
            VelocityMLPLegacy,
            VelocityResNet,
            VelocityResNetFlow,
            FourierMLP,
        )

        if architecture.lower() == "adamlp":
            self.velocity_model = VectorFieldAdaMLP(
                input_dim=input_size,
                context_dim=context_dim,
                hidden_features=hidden_features,
                num_layers=num_layers,
                time_embedding_dim=time_embedding_dim,
                activation=type(activation),
            )
            print("  Using VectorFieldAdaMLP architecture")

        elif architecture.lower() == "mlplegacy":
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
            print("  Using VelocityResNet architecture")

        elif architecture.lower() == "resnetflow":
            self.velocity_model = VelocityResNetFlow(
                input_dim=input_size,
                hidden_dim=hidden_features,
                num_blocks=num_layers,
                time_embedding_dim=time_embedding_dim,
                activation=activation if callable(activation) else F.relu,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            )
            print("  Using nflows ResidualNet architecture")

        elif architecture.lower() == "fouriermlp":
            self.velocity_model = FourierMLP(
                dim_in=input_size,
                dim_out=input_size,
                num_resnet_blocks=num_layers,
                dim_hidden=hidden_features,
                activation=activation,
                time_embedding_dim=time_embedding_dim,
                **kwargs,
            )
            print("  Using FourierMLP architecture")

        else:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Choose from: 'AdaMLP', 'ResNet', 'ResNetFlow', 'FourierMLP', 'MLPLegacy'"
            )

    def print_summary(self):
        if self.velocity_model is None:
            print("No model created yet.")
            return
        total_params = sum(
            p.numel() for p in self.velocity_model.parameters() if p.requires_grad
        )
        print(f"Total params to optimize: {total_params:,}")

    def compute_loss(
        self, x1: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes Conditional Flow Matching (CFM) loss.

        Target: x1 (Data). Source: x0 (Normal Noise).
        Path: x_t = (1 - t) * x0 + t * x1
        Velocity Target: v_t = x1 - x0
        """
        batch_size = x1.shape[0]
        device = x1.device

        # 1. Sample x0 (Noise) and Time t
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, 1, device=device)

        # 2. Optimal Transport Path
        # Expanded t for broadcasting: (batch, 1, 1...) matches x1 dimensions
        t_expand = t.view(batch_size, *([1] * (x1.ndim - 1)))

        x_t = (1 - t_expand) * x0 + t_expand * x1
        v_target = x1 - x0

        # 3. Predict Velocity
        # inputs: t, x_t, context
        v_pred = self.velocity_model(t, x_t, context)

        # 4. MSE Loss
        loss = torch.mean((v_pred - v_target) ** 2)
        return loss

    def train_flow(
        self,
        train_loader: torch.utils.data.DataLoader,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cpu",
        output_model: Optional[str] = None,
        save_model: bool = False,
        load_existing: bool = False,
        **kwargs,
    ) -> None:
        """
        Trains the model using CFM loss.
        """

        if train_loader is not None:
            # Check if dataset has normalization stats
            if hasattr(train_loader.dataset, "y_mean"):
                self.y_mean = train_loader.dataset.y_mean
                self.y_std = train_loader.dataset.y_std
            else:
                # Fallback defaults if not present
                self.y_mean = torch.zeros(1)
                self.y_std = torch.ones(1)

        # Load existing
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

        # Setup
        utils.configure_device(self, device)  # Helper to move self to device
        self.velocity_model.to(device)
        optimizer = torch.optim.AdamW(
            self.velocity_model.parameters(), lr=learning_rate
        )

        if self.y_mean is not None:
            self.y_mean = self.y_mean.to(device)
            self.y_std = self.y_std.to(device)

        self.velocity_model.train()
        print(f"Starting training on {device}...")

        for epoch in range(1, num_epochs + 1):
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

            for batch in pbar:
                if isinstance(batch, list):
                    batch = batch[0]  # Handle tuple batches if any
                if not isinstance(batch, torch.Tensor):
                    batch = torch.from_numpy(batch).float()

                batch = batch.to(device)

                # Normalize
                x1 = (batch - self.y_mean) / self.y_std

                # Context handling (if we had labels, we'd extract them here)
                # For now, unconditional flow matching for density estimation
                context = None
                # But AdaMLP expects context vector if context_dim > 0.
                # If we initialized context_dim=0, we pass None/dummy.
                if getattr(self.velocity_model, "global_mlp", None):
                    if (
                        self.velocity_model.global_mlp.input_layer.in_features
                        > self.velocity_model.global_mlp.time_emb.embed_dim
                    ):
                        # Expecting context. Pass zero context for unconditional if needed?
                        # Ideally factory logic handles this.
                        # For this generic implementation, we assume context_dim=0 -> unconditional.
                        pass

                optimizer.zero_grad()
                loss = self.compute_loss(x1, context)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = np.mean(epoch_losses)
            self.training_loss.append(avg_loss)

            if save_model and output_model:
                torch.save(self.velocity_model.state_dict(), output_model)

        print("Training complete.")
        self.velocity_model.to("cpu")  # Move back to CPU to save memory/state

    def sample(
        self,
        num_samples: int,
        device: str = "cpu",
        steps: int = 100,
        method: str = "dopri5",
        context: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Samples from the flow using torchdiffeq.

        Solve ODE from t=0 (Noise) to t=1 (Data).
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is required for sampling.")

        self.velocity_model.eval()
        self.velocity_model.to(device)

        # 1. Sample Noise (x0)
        # Assuming input_dim is known from model output layer
        input_dim = self.velocity_model.final_layer.out_features
        x0 = torch.randn(num_samples, input_dim, device=device)

        # 2. Define ODE function
        # d/dt x_t = v(t, x_t, context)
        class ODEFunc(nn.Module):
            def __init__(self, net, context):
                super().__init__()
                self.net = net
                self.context = context

            def forward(self, t, x):
                # t is scalar during simple integration, expand it
                t_expand = t * torch.ones(x.shape[0], 1, device=x.device)
                return self.net(t_expand, x, self.context)

        func = ODEFunc(self.velocity_model, context)

        # 3. Integrate 0 -> 1
        # Use simple fixed grid for 'euler' or 'midpoint', or adaptive for 'dopri5'
        if method == "euler" or method == "rk4":
            t_span = torch.linspace(0, 1, steps, device=device)
        else:
            t_span = torch.tensor([0.0, 1.0], device=device)

        with torch.no_grad():
            full_traj = odeint(func, x0, t_span, method=method, atol=1e-5, rtol=1e-5)

        # full_traj shape: (time_steps, batch, dim)
        # Final sample is at last time step
        samples = full_traj[-1]

        # Denormalize
        if self.y_mean is not None:
            self.y_mean = self.y_mean.to(device)
            self.y_std = self.y_std.to(device)
            samples = samples * self.y_std + self.y_mean

        return samples.cpu().numpy()

    def log_prob(
        self,
        inputs: Union[torch.Tensor, np.ndarray, object],
        dataset_normalization: bool = True,
        batch_size: int = 1000,
        device: str = "cpu",
        context: Optional[torch.Tensor] = None,
        solver_method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        exact_divergence: bool = False,
        steps: int = 100,  # Kept for compatibility but unused if solver is adaptive
        **kwargs,
    ) -> np.ndarray:
        """
        Computes log-probability of inputs by integrating the ODE backward from
        Data (t=1) to Noise (t=0).

        Δ log p = ∫ -Tr(∂v/∂x) dt
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is required for log_prob.")

        self.velocity_model.eval()
        utils.configure_device(self, device)
        self.velocity_model.to(device)

        # 1. Handle Inputs (Dataset vs Tensor)
        if hasattr(inputs, "normalized_patches"):
            # It's a GeneralizedPatchedDataset or similar
            if dataset_normalization:
                data = inputs.normalized_patches()
                apply_mean_std_adjustment = False
            else:
                data = inputs.patches
                apply_mean_std_adjustment = False
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

        # 2. Normalize and Compute Jacobian Transform LogDet

        # If we need to apply normalization using self.y_mean/std
        if apply_mean_std_adjustment and self.y_mean is not None:
            self.y_mean = self.y_mean.to(device)
            self.y_std = self.y_std.to(device)

            # Calculate normalization log det
            # y = (x - mu) / std
            # log p(x) = log p(y) + log |det dy/dx|
            # dy/dx = 1/std
            # log |det| = sum(log(1/std)) = -sum(log(std))
            if self.y_std.numel() == 1:
                -torch.log(self.y_std) * data.shape[1]
            else:
                -torch.sum(torch.log(self.y_std))

        # Data Loading
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        log_probs = []

        # Integration times: 1.0 -> 0.0 (Data -> Noise)
        # Integration times: 1.0 -> 0.0 (Data -> Noise)
        t_span = torch.tensor([1.0, 0.0], device=device)
        # Handle fixed step solvers which might need explicit grid
        if solver_method in ["euler", "rk4", "midpoint"]:
            # Check if step_size was passed in kwargs (test_backends uses it)
            step_size = kwargs.get("step_size", None)

            if step_size is not None:
                computed_steps = int(1.0 / step_size)
                t_span = torch.linspace(1, 0, computed_steps, device=device)
            else:
                # Fallback to 'steps' argument
                t_span = torch.linspace(1, 0, steps, device=device)

        class ODEFuncLogProb(nn.Module):
            def __init__(self, net, context):
                super().__init__()
                self.net = net
                self.context = context

            def forward(self, t, states):
                x = states[0]

                # Compute gradients inside forward to avoid double evaluation
                with torch.set_grad_enabled(True):
                    x.requires_grad_(True)
                    t_expand = t * torch.ones(x.shape[0], 1, device=x.device)
                    v = self.net(t_expand, x, self.context)

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

                # Apply normalization if flag is True AND we have stats
                if apply_mean_std_adjustment and self.y_mean is not None:
                    x_batch = (x_batch.to(device) - self.y_mean) / self.y_std
                else:
                    x_batch = x_batch.to(device)

                batch_n = x_batch.shape[0]
                zeros = torch.zeros(batch_n, device=device)

                func = ODEFuncLogProb(self.velocity_model, context)

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

                d = x_final.shape[1]
                log_p_z = -0.5 * d * np.log(2 * np.pi) - 0.5 * torch.sum(
                    x_final**2, dim=1
                )

                # log p(x) = log p(z) + delta_log_p

                batch_log_prob = log_p_z - delta_log_p

                log_probs.append(batch_log_prob.cpu().numpy())

        return np.concatenate(log_probs)

    def plot_train_loss(self, show_plot: bool = False, save_path: Optional[str] = None):
        """
        Plots the training loss over epochs.
        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.training_loss, ".-")
        if len(self.training_loss) > 1:
            try:
                # Helper to format title if utils available, else simple
                output_title_latex = r"${:}$".format(
                    utils.nume2string(self.training_loss[-1])
                )
                plt.title("Final loss: " + output_title_latex)
            except:
                pass
        plt.minorticks_on()
        plt.xlabel("Epoch")
        plt.ylabel(r"FM Loss: $\|\mathbf{v}_\theta - \mathbf{u}_t\|^2$")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        if show_plot:
            plt.show()

    def check_variables(
        self,
        train_loader: torch.utils.data.DataLoader,
        plot_variables: List[int] = [0, 1],
        figsize: Tuple[int, int] = (8, 4),
        device: str = "cpu",
        batch_size: int = 50000,  # Unused, kept for API compatibility
        rel_size: float = 0.1,
        steps: int = 100,
        method: str = "dopri5",
    ):
        """
        Checks latent space by transforming samples backwards to noise (x_data
        -> x_noise).

        Plots histograms of inverted variables vs Standard Normal.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        if not TORCHDIFFEQ_AVAILABLE:
            print("check_variables requires torchdiffeq.")
            return

        self.velocity_model.eval()
        utils.configure_device(self, device)
        self.velocity_model.to(device)

        # Collect data
        total_samples = len(train_loader.dataset)
        target_samples = int(total_samples * rel_size)

        latents = []
        collected = 0

        # Backward integration: 1 -> 0
        t_span = torch.tensor([1.0, 0.0], device=device)
        if method in ["euler", "rk4"]:
            t_span = torch.linspace(1, 0, steps, device=device)

        class ODEFuncBackward(nn.Module):
            def __init__(self, net, context):
                super().__init__()
                self.net = net
                self.context = context

            def forward(self, t, x):
                # Backwards: dx/dt = v(t, x)
                # We integrate from 1 to 0, which automatically reverses flow.
                t_expand = t * torch.ones(x.shape[0], 1, device=x.device)
                return self.net(t_expand, x, self.context)

        print(
            f"Checking variables (Backward Integration)... Target samples: {target_samples}"
        )

        with torch.no_grad():
            for batch in train_loader:
                if collected >= target_samples:
                    break

                if isinstance(batch, list):
                    batch = batch[0]
                if not isinstance(batch, torch.Tensor):
                    batch = torch.from_numpy(batch).float()

                # Normalize
                batch = batch.to(device)

                # Ensure we have normalization stats if available in dataset
                if self.y_mean is None and hasattr(train_loader.dataset, "y_mean"):
                    self.y_mean = train_loader.dataset.y_mean
                    self.y_std = train_loader.dataset.y_std

                if self.y_mean is not None:
                    # Move to device if needed (safeguard)
                    if self.y_mean.device != device:
                        self.y_mean = self.y_mean.to(device)
                        self.y_std = self.y_std.to(device)

                    batch = (batch - self.y_mean) / self.y_std

                # Integrate 1 -> 0
                func = ODEFuncBackward(self.velocity_model, None)  # Assuming no context
                traj = odeint(func, batch, t_span, method=method, atol=1e-5, rtol=1e-5)
                x0 = traj[-1]  # Latent (Noise)

                latents.append(x0.cpu().numpy())
                collected += batch.shape[0]

        latents = np.concatenate(latents, axis=0)
        latents = latents[:target_samples]

        # Validate that the inverted distribution is close to N(0,1)
        z_mean = np.mean(latents, axis=0)
        z_std = np.std(latents, axis=0)
        print("\n=== Latent Space Statistics ===")
        print(f"Mean (should be ≈ 0.0): {z_mean[: min(5, len(z_mean))]}")
        print(f"Std  (should be ≈ 1.0): {z_std[: min(5, len(z_std))]}")

        # Check for large deviations
        mean_deviation = np.abs(z_mean).max()
        std_deviation = np.abs(z_std - 1.0).max()
        if mean_deviation > 0.5 or std_deviation > 0.5:
            print("⚠ Warning: Latent distribution deviates significantly from N(0,1)")
            print(
                f"  Max |mean|: {mean_deviation:.3f}, Max |std-1|: {std_deviation:.3f}"
            )
            print(
                "  This may indicate: 1) model not fully trained, 2) ODE integration errors"
            )
        else:
            print("✓ Latent distribution is close to N(0,1)")

        # Plot using same style as original
        if len(plot_variables) > latents.shape[1]:
            plot_variables = list(range(latents.shape[1]))
        elif len(plot_variables) == 0:
            plot_variables = list(range(latents.shape[1]))

        fig, axes = plt.subplots(1, len(plot_variables), figsize=figsize, sharey=True)
        if len(plot_variables) == 1:
            axes = [axes]

        for i, var in enumerate(plot_variables):
            ax = axes[i]
            ax.hist(latents[:, var], bins=100, density=True, alpha=0.5, label="Output")
            ax.set_title(f"Latent Variable {var}")
            ax.set_xlabel("Value")
            if i == 0:
                ax.set_ylabel("Density")

            # Plot standard normal reference
            x_range = np.linspace(latents[:, var].min(), latents[:, var].max(), 100)
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
