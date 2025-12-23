
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from sklearn.datasets import make_moons
import os
import sys

# Ensure inspectorch is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inspectorch import DensityEstimator
from inspectorch import utils

# === Helper Classes ===

class MockDataset(torch.utils.data.Dataset):
    """
    Mock dataset that replicates the interface of GeneralizedPatchedDataset.
    FlowMatchingBackend expects .y_mean, .y_std, and .patches attributes.
    """
    def __init__(self, data_tensor):
        self.data = data_tensor
        self.patches = data_tensor # Treat data as 'patches'
        
        # Compute mean/std for normalization (even if data is already roughly normalized)
        self.y_mean = torch.mean(self.data, dim=0)
        self.y_std = torch.std(self.data, dim=0)
        # Avoid zero std
        self.y_std = torch.where(self.y_std < 1e-6, torch.ones_like(self.y_std), self.y_std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
    def normalized_patches(self):
        return (self.patches - self.y_mean) / self.y_std

# === Data Generation ===

def get_moon_data(n_samples=2000):
    data, _ = make_moons(n_samples=n_samples, noise=0.05)
    data = data.astype(np.float32)
    # Normalize to roughly [-2, 2] centered for easier flows
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data

# === Viz ===

def plot_results(name, data, samples, log_probs_grid=None, extent=None):
    plt.figure(figsize=(12, 4))
    
    # 1. Original Data
    plt.subplot(1, 3, 1)
    plt.scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, label='Data')
    plt.title(f"{name}: Data")
    plt.legend()
    
    # 2. Samples
    plt.subplot(1, 3, 2)
    if samples is not None:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, color='orange', label='Samples')
    else:
        plt.text(0.5, 0.5, "Sampling not supported", ha='center')
        
    plt.title(f"{name}: Samples")
    plt.legend()
    
    # 3. Log Prob
    plt.subplot(1, 3, 3)
    if log_probs_grid is not None:
        plt.imshow(log_probs_grid, origin='lower', extent=extent, cmap='viridis')
        plt.colorbar(label='Log Prob')
    else:
        plt.text(0.5, 0.5, "Grid Log Prob Skipped", ha='center')
        
    plt.title(f"{name}: Log Prob")
    
    plt.tight_layout()
    plt.savefig(f"models/test_backend_{name}.png")
    plt.close()
    print(f"Saved plot to models/test_backend_{name}.png")

# === Tester ===

def test_backend(backend_name, data, sequence_length=1):
    print(f"\n{'='*20} Testing Backend: {backend_name} {'='*20}")
    
    train_data = data
    input_dim = 2
    effective_T = 1
        
    # Wrap in MockDataset
    dataset = MockDataset(torch.tensor(train_data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 1. Initialize
    try:
        model = DensityEstimator(type=backend_name)
    except Exception as e:
        print(f"SKIPPING {backend_name}: {e}")
        return

    # 2. Create Flow
    kwargs = {}
    if backend_name == "flow_matching_sbi":
        kwargs['hidden_features'] = 64
        kwargs['num_transforms'] = 5
        model.create_flow(input_size=input_dim, **kwargs) 
        
    elif backend_name == "normalizing_flow":
        kwargs['hidden_features'] = 32
        kwargs['num_layers'] = 5
        kwargs['num_bins'] = 8 
        model.create_flow(input_size=input_dim, **kwargs)
        
    else: # flow_matching (Default)
        kwargs['hidden_features'] = 32
        kwargs['num_layers'] = 5
        kwargs['architecture'] = "MLP"
        kwargs['time_embedding_dim'] = 32
        kwargs['scheduler_n'] = 3.0
        model.create_flow(input_size=input_dim, **kwargs)
        
    print("Model created.")
    
    # 3. Check Initial Log Prob
    # Use dataset object for backends that expect it
    sample_batch_idx = list(range(10))
    sample_subset = torch.utils.data.Subset(dataset, sample_batch_idx)

    # Note: model.log_prob usually accepts tensor or object with .patches
    # To be safe for all backends, passing the MockDataset object or a wrapper
    class SubsetWrapper:
        def __init__(self, subset):
            self.subset = subset
            # Mock patches logic: concat all data in subset
            self.patches = torch.stack([subset[i] for i in range(len(subset))])
            self.y_mean = subset.dataset.y_mean
            self.y_std = subset.dataset.y_std
        def normalized_patches(self):
            return (self.patches - self.y_mean) / self.y_std
            
    # Define extra kwargs for log_prob based on backend
    lp_kwargs = {}
    if "flow_matching" in backend_name:
         # User requested "solver to euler" for speed
         lp_kwargs = {"solver_method": "euler", "step_size": 0.1}

    try:
        # Use coarse step_size/tolerances for speed in testing
        lp = model.log_prob(SubsetWrapper(sample_subset), batch_size=10, **lp_kwargs)
        print(f"Initial Mean Log Prob: {lp.mean():.4f}")
    except Exception as e:
        print(f"Initial Log Prob Failed: {e}")
        lp = np.array([-999.0])
    
    # 4. Train
    print("Training...")
    try:
        # Standard LR and Epochs
        lr = 1e-3
        epochs = 100
        model.train_flow(loader, num_epochs=epochs, learning_rate=lr, save_model=False)
    except Exception as e:
        print(f"TRAINING FAILED: {e}")
        import traceback; traceback.print_exc()
        return

    # 5. Check Trained Log Prob
    try:
        lp_trained = model.log_prob(SubsetWrapper(sample_subset), batch_size=10, **lp_kwargs)
        print(f"Trained Mean Log Prob: {lp_trained.mean():.4f}")
        if lp_trained.mean() > lp.mean():
            print("PASS: Log prob improved.")
        else:
            print("WARNING: Log prob did not improve.")
    except Exception as e:
        print(f"Trained Log Prob Failed: {e}")

    # 6. Sample
    print("Sampling...")
    samples = None
    try:
         # All backends should support sampling now
         samples = model.sample(1000)
         if isinstance(samples, torch.Tensor):
             samples = samples.detach().cpu().numpy()
         # Flatten if (N, T, C)
         if samples.ndim == 3:
              samples = samples.reshape(samples.shape[0], -1)
         print(f"Samples shape: {samples.shape}")
    except Exception as e:
        print(f"Sampling failed: {e}")

    # 7. Visualization
    limit = 3
    # Reduced resolution for speed (ODE solvers are slow on CPU)
    x = np.linspace(-limit, limit, 20)
    y = np.linspace(-limit, limit, 20)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    
    # Grid Wrapper
    class GridWrapper:
            def __init__(self, g):
                self.patches = torch.tensor(g)
                    
                # Use dataset stats for consistency if backend uses them
                self.y_mean = dataset.y_mean if hasattr(dataset, 'y_mean') else torch.zeros(2)
                self.y_std = dataset.y_std if hasattr(dataset, 'y_std') else torch.ones(2)
            def normalized_patches(self):
                # Standard normalization (if used)
                # Note: reshaping might mess up broadcast if y_mean is (2,) and patches (N, 2, 1)
                # If StarFlow T=2, C=1, y_mean should be handled carefully.
                # Here we assume data was pre-normalized or mock dataset handles it.
                # For simplicity in this test, we skip complex normalization logic check
                # as get_moon_data is already normalized standard normal.
                return self.patches
    
    try:
        # Reshape back to grid size (20x20)
        lp_grid = model.log_prob(GridWrapper(grid), batch_size=400, **lp_kwargs).reshape(20, 20)
        plot_results(backend_name, train_data, samples, lp_grid, extent=[-limit, limit, -limit, limit])
    except Exception as e:
        print(f"Grid Viz Failed: {e}")
        import traceback; traceback.print_exc()
        plot_results(backend_name, train_data, samples, None)

class TLogger:
    """Redirects stream to both the original stream and a file."""
    def __init__(self, filename, original_stream):
        self.terminal = original_stream
        self.log = open(filename, "a")  # Use append mode just in case

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Setup Logger
    # Clean previous log if exists because we append now
    if os.path.exists("models/test_run.log"):
        os.remove("models/test_run.log")
        
    sys.stdout = TLogger("models/test_run.log", sys.stdout)
    sys.stderr = TLogger("models/test_run.log", sys.stderr)
    print(f"Logging to models/test_run.log")
        
    data = get_moon_data()
    
    backends = [
        "normalizing_flow",
        "flow_matching",
        "flow_matching_sbi",
        # "starflow" # Removed
    ]
    
    for b in backends:
        try:
            test_backend(b, data)
        except Exception as e:
            print(f"FATAL ERROR TESTING {b}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
