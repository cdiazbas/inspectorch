
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Tuple, List
from tqdm import tqdm
import math

# TorchCFM imports (pip-installable library) - REQUIRED
try:
    from torchcfm.conditional_flow_matching import (
        ConditionalFlowMatcher,
        ExactOptimalTransportConditionalFlowMatcher,
        TargetConditionalFlowMatcher,
        SchrodingerBridgeConditionalFlowMatcher
    )
    TORCHCFM_AVAILABLE = True
except ImportError:
    raise ImportError(
        "\n" + "="*70 + "\n"
        "ERROR: torchcfm library is required for FlowMatchingCFMBackend\n"
        "\n"
        "Install with:\n"
        "  pip install torchcfm\n"
        "\n"
        "Or install from source:\n"
        "  git clone https://github.com/atong01/conditional-flow-matching\n"
        "  cd conditional-flow-matching && pip install -e .\n"
        "\n"
        "Note: For Optimal Transport methods, also install:\n"
        "  pip install pot\n"
        + "="*70
    )

# Use existing dependencies for inference
from torchdiffeq import odeint

# Import VectorFieldAdaMLP from flow_matching_sbi
try:
    from .flow_matching_sbi import VectorFieldAdaMLP
except ImportError:
    VectorFieldAdaMLP = None
    print("WARNING: Could not import VectorFieldAdaMLP from flow_matching_sbi.")

class CFMMLPWrapper(nn.Module):
    """
    Wrapper for standard MLP to accept (t, x) separate inputs.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        # t: (B,) or (B, 1)
        # x: (B, D)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])
            
        t_emb = t.view(-1, 1)
        x_in = torch.cat([x, t_emb], dim=-1)
        return self.model(x_in)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_features=64, num_layers=3):
        super().__init__()
        layers = []
        # Input: Data dim + 1 (time)
        layers.append(nn.Linear(input_dim + 1, hidden_features))
        layers.append(nn.ELU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ELU())
        layers.append(nn.Linear(hidden_features, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FlowMatchingCFMBackend:
    """
    Backend integrating 'conditional-flow-matching' (torchcfm).
    """

    def __init__(self, sigma: float = 0.0, method: str = "independent", **kwargs):
        """
        Args:
            sigma: Noise level (sigma_min for OT-CFM).
            method: 'independent' (fast, default), 'exact' (OT-CFM, slow), 'target', or 'sb'.
        """
        self.sigma = sigma
        self.method = method
        self.velocity_model = None # Interface: forward(t, x) -> v
        
        # Initialize Matcher
        if method == "exact":
            if ExactOptimalTransportConditionalFlowMatcher is not None:
                self.matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
            else:
                print("WARNING: Exact OT matcher unavailable (missing pot?). Falling back to Independent CFM.")
                self.matcher = ConditionalFlowMatcher(sigma=sigma)
        elif method == "target":
             if TargetConditionalFlowMatcher is not None:
                self.matcher = TargetConditionalFlowMatcher(sigma=sigma)
             else:
                print("WARNING: Target matcher unavailable. Falling back to Independent CFM.")
                self.matcher = ConditionalFlowMatcher(sigma=sigma)
        elif method == "sb":
             if SchrodingerBridgeConditionalFlowMatcher is not None:
                self.matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma if sigma > 0 else 1.0)
             else:
                print("WARNING: SB matcher unavailable. Falling back to Independent CFM.")
                self.matcher = ConditionalFlowMatcher(sigma=sigma)
        else:
            # Default independent
            self.matcher = ConditionalFlowMatcher(sigma=sigma)
            
    def create_flow(
        self, 
        input_size: int, 
        hidden_features: int = 64, 
        num_layers: int = 3, 
        architecture: str = "MLP", 
        **kwargs
    ):
        """
        Initialize the neural network vector field.
        Args:
            architecture: "MLP" (Standard) or "AdaMLP" (Adaptive)
        """
        self.input_dim = input_size
        
        if architecture == "AdaMLP" and VectorFieldAdaMLP is not None:
             # VectorFieldAdaMLP(input_dim, context_dim, ...)
             self.velocity_model = VectorFieldAdaMLP(
                 input_dim=input_size,
                 context_dim=0,
                 hidden_features=hidden_features,
                 num_layers=num_layers,
                 time_emb_dim=kwargs.get("time_embedding_dim", 32)
             )
        else:
            if architecture == "AdaMLP":
                print("WARNING: AdaMLP requested but not available. Falling back to MLP.")
                
            # Standard MLP
            net = MLP(input_size, hidden_features, num_layers)
            self.velocity_model = CFMMLPWrapper(net)
        
    def train_flow(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        num_epochs: int = 100, 
        learning_rate: float = 1e-3, 
        save_model: bool = False,
        **kwargs
    ):
        """
        Train using the Conditional Flow Matching objective.
        Loss = ||v_theta(t, x_t) - u_t(x|z)||^2
        """
        if self.velocity_model is None:
            raise ValueError("Model not created. Call create_flow first.")

        optimizer = torch.optim.Adam(self.velocity_model.parameters(), lr=learning_rate)
        
        self.velocity_model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.velocity_model.to(device)

        print(f"Device: Using {device} for training.")
        
        self.training_loss = []
        
        for epoch in range(1, num_epochs + 1):
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
            
            for batch in pbar:
                # Handle batch: expects Tensor or list
                if isinstance(batch, list):
                    x1 = batch[0]
                else:
                    x1 = batch
                
                x1 = x1.to(device)
                
                # Normalize (same as other backends)
                if hasattr(train_loader.dataset, "y_mean"):
                     y_mean = train_loader.dataset.y_mean.to(device)
                     y_std = train_loader.dataset.y_std.to(device)
                     x1 = (x1 - y_mean) / y_std
                
                # 1. Sample Source Distribution x0 (Gaussian Noise)
                x0 = torch.randn_like(x1)
                
                # 2. Sample Conditional Flow (t, xt, ut)
                # t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
                if self.method in ["exact", "sb"]:
                     # OT methods might return extra args but typically signature matches
                     t, xt, ut = self.matcher.sample_location_and_conditional_flow(x0, x1)
                else:
                     t, xt, ut = self.matcher.sample_location_and_conditional_flow(x0, x1)

                # 3. Model Prediction v_theta(t, xt)
                # velocity_model takes (t, x)
                vt = self.velocity_model(t, xt)
                
                # 4. Loss MSE
                loss = torch.mean((vt - ut) ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_loss = np.mean(epoch_losses)
            self.training_loss.append(avg_loss)
        
        print("Training complete.")

    def print_summary(self):
        """
        Prints a summary of the model and matching method.
        """
        print(f"CFM Method: {self.method}")
        print(f"Sigma: {self.sigma}")
        
        if self.velocity_model is None:
            print("No model created yet.")
            return

        total_params = sum(
            p.numel() for p in self.velocity_model.parameters() if p.requires_grad
        )
        print(f"Total params to optimize: {total_params:,}")

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
        Checks latent space by transforming samples backwards to noise from t=1 (Data) to t=0 (Noise).
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        from tqdm import tqdm
        
        self.velocity_model.eval()
        self.velocity_model.to(device)

        # Get stats if available
        if hasattr(train_loader.dataset, "y_mean"):
             y_mean = train_loader.dataset.y_mean.to(device)
             y_std = train_loader.dataset.y_std.to(device)
        else:
             y_mean = torch.zeros(self.input_dim, device=device)
             y_std = torch.ones(self.input_dim, device=device) # Assuming normalized

        dataset_size = len(train_loader.dataset)
        total_samples = int(rel_size * dataset_size)

        # Sample random indices
        # If dataset is tensor-like
        if hasattr(train_loader.dataset, "patches"):
             data_source = train_loader.dataset.patches
        elif hasattr(train_loader.dataset, "tensors"):
             data_source = train_loader.dataset.tensors[0]
        else:
             print("Warning: Dataset format not recognized for random sampling. Using sequential loader.")
             data_source = None
             
        z_batches = []
        
        # Integration setup
        # Backward integration: t from 1.0 to 0.0
        t_span = torch.tensor([1.0, 0.0], device=device)
        
        def velocity_fn(t, x):
             # t: scalar tensor (during solve) -> expand for model
             t_batch = t.expand(x.shape[0])
             return self.velocity_model(t_batch, x)

        with torch.no_grad():
            if data_source is not None:
                random_indices = np.random.randint(0, len(data_source), size=total_samples)
                
                for i in tqdm(range(0, total_samples, batch_size), desc="Transforming to noise"):
                    end_idx = min(i + batch_size, total_samples)
                    batch_indices = random_indices[i:end_idx]
                    batch = data_source[batch_indices]
                    
                    if isinstance(batch, torch.Tensor):
                         batch = batch.to(device)
                    
                    # Normalize
                    x1 = (batch - y_mean) / y_std
                    
                    # Integrate 1 -> 0
                    x_traj = odeint(
                        velocity_fn, x1, t_span, method="dopri5", atol=1e-5, rtol=1e-5
                    )
                    x0 = x_traj[-1]
                    z_batches.append(x0.cpu().numpy())
            else:
                 # Sequential fallback
                 count = 0
                 for batch in tqdm(train_loader, desc="Transforming to noise (seq)"):
                      if count >= total_samples: break
                      x = batch[0] if isinstance(batch, list) else batch
                      x = x.to(device)
                      x1 = (x - y_mean) / y_std
                      
                      x_traj = odeint(
                        velocity_fn, x1, t_span, method="dopri5", atol=1e-5, rtol=1e-5
                      )
                      x0 = x_traj[-1]
                      z_batches.append(x0.cpu().numpy())
                      count += x.shape[0]

        if not z_batches:
            print("Warning: No valid samples to plot")
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

    def plot_train_loss(self, show_plot=False, save_path=None):
        """
        Plots the training loss over epochs.
        """
        # Use simple matplotlib if utils not available/compatible, 
        # or import standard utility.
        # FlowMatchingBackend uses: plot_train_loss(self.training_loss, ...)
        # We can implement directly or import.
        # Let's import the one from inspectorch.utils typically used?
        # Actually FlowMatchingBackend calls a global function `plot_train_loss` which is likely imported?
        # No, it looks like it might be a utility.
        # Let's just implement a simple one here or use inspectorch.utils if we knew where it was.
        # To remain independent:
        
        if not hasattr(self, "training_loss") or not self.training_loss:
            print("No training loss to plot.")
            return

        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss, ".-")
        if len(self.training_loss) > 1:
            try:
                from . import utils
                output_title_latex = r"${:}$".format(utils.nume2string(self.training_loss[-1]))
                plt.title("Final loss: " + output_title_latex)
            except:
                plt.title(f"Final loss: {self.training_loss[-1]:.4f}")
        plt.minorticks_on()
        plt.xlabel("Epoch")
        plt.ylabel(r"FM Loss: $\|\mathbf{v}_\theta - \mathbf{u}_t\|^2$")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
            
        if show_plot:
            plt.show() # In notebooks this is needed or just plt.figure is enough
        else:
            # If not show_plot, usually we still show in notebooks? 
            # The API says show_plot=False default? 
            # Actually standard behavior:
            plt.show()

    def _ode_func(self, t, x):
        """
        Wrapper for torchdiffeq.
        """
        # x: (B, D)
        # t: scalar
        # model expects (B, D+1) with t concatenated
        B = x.shape[0]
        # Allow t to be tensor scalar or float
        if isinstance(t, float):
             t_tens = torch.tensor(t).to(x.device).float()
        else:
             t_tens = t.to(x.device).float()
             
        # t_tens is scalar. 
        # wrapper/AdaMLP expects t to be broadcastable or (B, )?
        # wrapper expects (B,) or (B,1). AdaMLP expects (B,) or (B,1).
        # Expand t to (B,)
        t_batch = t_tens.expand(B)
        return self.velocity_model(t_batch, x)
    
    def log_prob(
        self,
        inputs: Union[torch.Tensor, np.ndarray, object],
        solver_method: str = "dopri5",
        step_size: float = 0.1,  # Used if method is fixed step (e.g. euler)
        device: str = "cpu",
        batch_size: int = 100,
        use_naive: bool = False,  # False: API (CFM-style), True: SBI-style
        **kwargs
    ) -> np.ndarray:
        """
        Compute log-probability using Continuous Normalizing Flow trace estimation.
        
        Args:
            use_naive: If True, use naive implementation (like SBI). 
                      If False, use torchcfm API-based implementation.
        """
        if use_naive:
            return self._log_prob_naive(inputs, solver_method, step_size, device, batch_size, **kwargs)
        else:
            return self._log_prob_api(inputs, solver_method, step_size, device, batch_size, **kwargs)
    
    def _log_prob_naive(
        self,
        inputs: Union[torch.Tensor, np.ndarray, object],
        solver_method: str = "euler",
        step_size: float = 0.1,
        device: str = "cpu",
        batch_size: int = 1000,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        exact_divergence: bool = False,
        steps: int = 100,
        **kwargs
    ) -> np.ndarray:
        """
        Direct copy of SBI backend log_prob implementation.
        Computes log-probability by integrating ODE backward from Data (t=1) to Noise (t=0).
        """
        from tqdm import tqdm
        
        self.velocity_model.eval()
        self.velocity_model.to(device)

        # Handle inputs (simplified from SBI - assume normalized_patches)
        if hasattr(inputs, "normalized_patches"):
            data = inputs.normalized_patches()
        elif hasattr(inputs, "patches"):
             data = inputs.patches
        elif isinstance(inputs, np.ndarray):
            data = torch.from_numpy(inputs).float()
        else:
            data = inputs
            
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        log_probs = []
        
        # Integration times: 1.0 -> 0.0 (Data -> Noise)
        t_span = torch.tensor([1.0, 0.0], device=device)
        if solver_method in ['euler', 'rk4', 'midpoint']:
            t_span = torch.linspace(1, 0, steps, device=device)

        # ODE function matching SBI exactly
        class ODEFuncLogProb(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, t, states):
                x = states[0]
                
                with torch.set_grad_enabled(True):
                    x.requires_grad_(True)
                    t_expand = t * torch.ones(x.shape[0], 1, device=x.device)
                    # CFM model expects (t_batch, x), SBI expects (t_expand, x, context)
                    # Adapt: expand t to match batch
                    t_batch = t_expand.squeeze(-1)  # (B,)
                    v = self.net(t_batch, x)
                    
                    if not exact_divergence:  # Hutchinson
                        epsilon = torch.randn_like(x)
                        v_eps = torch.sum(v * epsilon)
                        grad_v_eps = torch.autograd.grad(v_eps, x, create_graph=False)[0]
                        div = torch.sum(grad_v_eps * epsilon, dim=-1)
                    else:  # Exact
                        div = 0.0
                        for i in range(x.shape[1]):
                            grad_v_i = torch.autograd.grad(v[:, i].sum(), x, create_graph=False, retain_graph=True)[0]
                            div += grad_v_i[:, i]
                
                return v, -div

        print(f"Computing log-probs on {device} using {solver_method}...")
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Log Prob"):
                x_batch = batch[0].to(device)

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
                    rtol=rtol
                )
                
                x_final = state[0][-1]
                delta_log_p = state[1][-1]
                
                d = x_final.shape[1]
                log_p_z = -0.5 * d * np.log(2 * np.pi) - 0.5 * torch.sum(x_final**2, dim=1)
                
                batch_log_prob = log_p_z - delta_log_p
                
                log_probs.append(batch_log_prob.cpu().numpy())

        return np.concatenate(log_probs)

    def _log_prob_api(
        self,
        inputs: Union[torch.Tensor, np.ndarray, object],
        solver_method: str = "euler",
        step_size: float = 0.1,
        device: str = "cpu",
        batch_size: int = 1000,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        exact_divergence: bool = False,
        steps: int = 100,
        **kwargs
    ) -> np.ndarray:
        """
        Original CFM implementation using combined state vector approach.
        This was the initial implementation before copying SBI.
        """
        self.velocity_model.eval()
        self.velocity_model.to(device)
        
        # Input handling
        if hasattr(inputs, "normalized_patches"):
            data = inputs.normalized_patches()
        elif hasattr(inputs, "patches"):
             data = inputs.patches
        elif isinstance(inputs, np.ndarray):
            data = torch.from_numpy(inputs).float()
        else:
            data = inputs
            
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_log_probs = []
        
        # Integration times: t=1 (Data) to t=0 (Noise)
        t_span = torch.tensor([1.0, 0.0]).to(device)
        # Handle fixed step solvers (must match naive implementation!)
        if solver_method in ["euler", "rk4", "midpoint"]:
            t_span = torch.linspace(1, 0, steps, device=device)
        
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                
                # Combined state approach (original CFM implementation)
                class ODEFunc(nn.Module):
                    def __init__(self, net):
                        super().__init__()
                        self.net = net
                    def forward(self, t, state):
                        # state: (B, D+1) - combined [x, logp]
                        x_curr = state[:, :-1]
                        
                        with torch.enable_grad():
                            x_curr = x_curr.detach().requires_grad_(True)
                            
                            # t is scalar
                            t_batch = t.expand(x_curr.shape[0])
                            dx = self.net(t_batch, x_curr)
                            
                            # Hutchinson's Trace Estimator
                            epsilon = torch.randn_like(x_curr)
                            v_eps = torch.sum(dx * epsilon)
                            grad_v_eps = torch.autograd.grad(v_eps, x_curr, create_graph=False)[0]
                            div = torch.sum(grad_v_eps * epsilon, dim=-1)
                        
                        # Output: concat(dx, -div) to match SBI sign convention
                        return torch.cat([dx, (-div).view(-1, 1)], dim=-1)

                state0 = torch.cat([x, torch.zeros(x.shape[0], 1, device=device)], dim=-1)
                ode = ODEFunc(self.velocity_model)
                
                # Integrate (no options needed when using linspace t_span)
                traj = odeint(
                    ode, 
                    state0, 
                    t_span, 
                    method=solver_method,
                    atol=atol, 
                    rtol=rtol
                )
                
                final_state = traj[-1] # at t=0
                z0 = final_state[:, :-1]
                delta_logp = final_state[:, -1] # Integral 1->0 div(v) dt
                
                # log p(x0) (Prior at t=0 is Normal(0,1))
                log_p_z0 = -0.5 * (math.log(2 * math.pi) + z0.pow(2)).sum(dim=-1)
                
                log_prob = log_p_z0 - delta_logp
                all_log_probs.append(log_prob.cpu().numpy())
                
        return np.concatenate(all_log_probs)

    def sample(
        self, 
        num_samples: int, 
        device: str = "cpu", 
        solver_method: str = "dopri5", 
        step_size=0.1,
        **kwargs
    ):
        """
        Generate samples by integrating from t=0 (Noise) to t=1 (Data).
        """
        self.velocity_model.eval()
        self.velocity_model.to(device)
        
        # 1. Sample Noise x0
        x0 = torch.randn(num_samples, self.input_dim, device=device)
        
        # 2. Integrate 0 -> 1
        t_span = torch.tensor([0.0, 1.0]).to(device)
        
        # Wrapper for odeint that matches (t, x) sig
        def func(t, x):
             return self._ode_func(t, x)
             
        options = {}
        if solver_method in ["euler", "rk4"]:
                     options = {"step_size": step_size}
        
        traj = odeint(
            func, 
            x0, 
            t_span, 
            method=solver_method, 
            options=options,
            atol=1e-5, rtol=1e-5
        )
        
        x1 = traj[-1]
        return x1.detach().cpu().numpy()
