
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from sklearn.datasets import make_moons
import os
import sys
import argparse
import time
import pandas as pd

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
    # Save to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f"test_backend_{name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

# === Tester ===

def test_backend(backend_name, data, sequence_length=1):
    print(f"\n{'='*20} Testing Backend: {backend_name} {'='*20}")
    
    metrics = {
        "backend": backend_name,
        "train_time": 0.0,
        "log_prob_time": 0.0,
        "final_loss": None,
        "mean_log_prob": None
    }
    
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
        return metrics

    # 2. Create Flow
    # Standardized hyperparameters for flow matching
    standard_fm_args = {
        'hidden_features': 64,
        'num_layers': 2,
        'architecture': "AdaMLP",
        'time_embedding_dim': 32
    }
    
    kwargs = {}
    if backend_name == "flow_matching_sbi":
        kwargs.update(standard_fm_args)
        model.create_flow(input_size=input_dim, **kwargs) 
        
    elif backend_name == "normalizing_flow":
        # Keep NF settings decent but not necessarily same architecture as FM
        kwargs['hidden_features'] = 64
        kwargs['num_layers'] = 5
        kwargs['num_bins'] = 8 
        model.create_flow(input_size=input_dim, **kwargs)
        
    elif backend_name == "flow_matching_cfm":
        kwargs.update(standard_fm_args)
        kwargs['method'] = "exact" # OT-CFM
        model.create_flow(input_size=input_dim, **kwargs)

    else: # flow_matching_ffm (Default)
        kwargs.update(standard_fm_args)
        # flow_matching might interpret num_layers differently or use hidden_features differently? 
        # But we unified models, so it should be fine if it passes them to velocity model.
        # flow_matching.py create_flow accepts num_layers and hidden_features.
        model.create_flow(input_size=input_dim, **kwargs)
        
    print("Model created.")
    model.print_summary()
    
    # 3. Check Initial Log Prob
    # Use dataset object for backends that expect it
    sample_batch_idx = list(range(10))
    sample_subset = torch.utils.data.Subset(dataset, sample_batch_idx)

    class SubsetWrapper:
        def __init__(self, subset):
            self.subset = subset
            # Mock patches: concat all data in subset
            self.patches = torch.stack([subset[i] for i in range(len(subset))])
            self.y_mean = subset.dataset.y_mean
            self.y_std = subset.dataset.y_std
        def normalized_patches(self):
            return (self.patches - self.y_mean) / self.y_std
            
    # Define extra kwargs for log_prob based on backend
    lp_kwargs = {}
    if "flow_matching" in backend_name:
         # User requested "solver to euler" for speed
         lp_kwargs = {"solver_method": "euler", "step_size": 0.01}

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
        
        start_time = time.time()
        model.train_flow(train_loader=loader, num_epochs=epochs, learning_rate=lr, save_model=False)
        end_time = time.time()
        metrics["train_time"] = end_time - start_time
        print(f"Training Time: {metrics['train_time']:.4f}s")
        
        # Capture training loss
        if hasattr(model.backend, "training_loss") and model.backend.training_loss:
            metrics["final_loss"] = model.backend.training_loss[-1]
        elif hasattr(model, "training_loss") and model.training_loss: # Fallback
            metrics["final_loss"] = model.training_loss[-1]
        
    except Exception as e:
        print(f"TRAINING FAILED: {e}")
        import traceback; traceback.print_exc()
        return metrics

    # 5. Check Trained Log Prob
    try:
        start_time = time.time()
        lp_trained = model.log_prob(SubsetWrapper(sample_subset), batch_size=10, **lp_kwargs)
        end_time = time.time()
        metrics["log_prob_time"] = end_time - start_time
        metrics["mean_log_prob"] = float(lp_trained.mean())
        
        print(f"Trained Mean Log Prob: {lp_trained.mean():.4f}")
        print(f"Log Prob Calculation Time: {metrics['log_prob_time']:.4f}s")
        
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
                return self.patches
    
    try:
        # Reshape back to grid size (20x20)
        lp_grid = model.log_prob(GridWrapper(grid), batch_size=400, **lp_kwargs).reshape(20, 20)
        plot_results(backend_name, train_data, samples, lp_grid, extent=[-limit, limit, -limit, limit])
    except Exception as e:
        print(f"Grid Viz Failed: {e}")
        import traceback; traceback.print_exc()
        plot_results(backend_name, train_data, samples, None)
        
    return metrics

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test INSPECTORCH backends')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific backend to test (e.g., flow_matching_cfm). If not specified, tests all backends.')
    args = parser.parse_args()
    
    # Output directory: Same as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir # "inspectorch/test/" implicitly
    
    # Setup Logger
    # Clean previous log if exists because we append now
    log_file = os.path.join(output_dir, "test_run.log")
    if os.path.exists(log_file):
        os.remove(log_file)
        
    sys.stdout = TLogger(log_file, sys.stdout)
    sys.stderr = TLogger(log_file, sys.stderr)
    print(f"Logging to {log_file}")
    
    data = get_moon_data()
    
    # All available backends
    all_backends = [
        "normalizing_flow",
        "flow_matching_ffm",
        "flow_matching_sbi",
        "flow_matching_cfm"
    ]
    
    # Determine which backends to test
    if args.model:
        if args.model in all_backends:
            backends = [args.model]
            print(f"Testing single backend: {args.model}")
        else:
            print(f"ERROR: Unknown backend '{args.model}'")
            print(f"Available backends: {', '.join(all_backends)}")
            return
    else:
        backends = all_backends
        print(f"Testing all backends: {', '.join(backends)}")
    
    results = []
    for b in backends:
        try:
            m = test_backend(b, data)
            results.append(m)
        except Exception as e:
            print(f"FATAL ERROR TESTING {b}: {e}")
            import traceback
            traceback.print_exc()
            
    # Print Summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    df = pd.DataFrame(results)
    if not df.empty:
        # Standardize NaN
        df = df.fillna("N/A")
        # Print table
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join(script_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()
