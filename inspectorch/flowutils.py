import torch
import torch.nn as nn  # Added for torch.nn.DataParallel
import nflows
from nflows import transforms
from nflows.transforms import CompositeTransform
from nflows import utils
from nflows.nn import nets
import torch.nn.functional as F
import numpy as np
from nflows.flows.base import Flow
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # Ensure tqdm is imported
from scipy.stats import norm
from einops import rearrange
import torch.nn.functional as F # <-- Import F
import warnings # <-- Import warnings

# =============================================================================
class dot_dict(dict):
    """
    A dictionary subclass that allows for attribute-style access.

    This class extends the built-in dictionary to allow accessing keys as attributes.
    It overrides the __getattr__, __setattr__, and __delattr__ methods to provide
    this functionality.

    Example:
        d = DotDict({'key': 'value'})
        print(d.key)  # Outputs: value
        d.new_key = 'new_value'
        print(d['new_key'])  # Outputs: new_value
        del d.key
        print(d)  # Outputs: {'new_key': 'new_value'}
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# =============================================================================
class GeneralizedPatchedDataset(torch.utils.data.Dataset):
    """
    A highly flexible PyTorch Dataset that uses named dimensions and einops
    to extract patches and prepare data for a normalizing flow.
    Includes automatic padding to preserve spatial/temporal dimensions.
    """
    def __init__(self, data, dimension_string, primary_variables, patch_config=None,
                 padding=True, padding_mode='reflect'):
        """
        Initializes the dataset, performs patch extraction and reshaping.

        Args:
            data (torch.Tensor or np.ndarray): The input data tensor.
            dimension_string (str): A space-separated string of dimension names.
            primary_variables (list): List of dimension names for the flow's features.
            patch_config (dict, optional): Configures patching on primary variables.
            padding (bool, optional): If True, applies padding to preserve the
                                      number of patches. Defaults to True.
            padding_mode (str, optional): The padding mode ('reflect', 'replicate', etc.).
                                          Defaults to 'reflect'.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()

        patch_config = patch_config or {}
        all_dims = dimension_string.split()

        if data.dim() != len(all_dims):
            raise ValueError(f"Data tensor has {data.dim()} dims, but "
                             f"dimension_string defines {len(all_dims)}.")

        # 1. Determine the role of each dimension
        patch_dims = [d for d in primary_variables if d in patch_config]
        feature_dims = [d for d in primary_variables if d not in patch_config]
        sample_dims = [d for d in all_dims if d not in primary_variables]

        # 2. Permute data to group dimensions by role (others, then patchable)
        # This makes the unfolding process predictable.
        other_dims = sample_dims + feature_dims
        permute_pattern = f"{' '.join(all_dims)} -> {' '.join(other_dims)} {' '.join(patch_dims)}"
        permuted_data = rearrange(data, permute_pattern)


        # 3. Iteratively unfold the patchable dimensions
        unfolded_data = permuted_data
        num_other_dims = len(other_dims)
        for i, dim_name in enumerate(patch_dims):
            axis_to_unfold = num_other_dims + i
            size = patch_config[dim_name]['size']
            stride = patch_config[dim_name].get('stride', 1)
            unfolded_data = unfolded_data.unfold(axis_to_unfold, size, stride)
            
        # 4. Construct the final rearrangement pattern to create the patches
        # The shape of unfolded_data is now: (*other_dims, *num_patches, *patch_sizes)
        sample_str = ' '.join(sample_dims)
        feature_str = ' '.join(feature_dims)
        num_patches_str = ' '.join([f'n_{d}' for d in patch_dims])
        patch_size_str = ' '.join([f'p_{d}' for d in patch_dims])
        
        input_pattern = f"{sample_str} {feature_str} {num_patches_str} {patch_size_str}"
        output_pattern = f"-> ({sample_str} {num_patches_str}) ({feature_str} {patch_size_str})"
        
        self.patches = rearrange(unfolded_data, f"{input_pattern} {output_pattern}")

        print(f"Dataset initialized with {self.patches.shape[0]} samples.")
        print(f"Each sample is a flattened vector of size {self.patches.shape[1]}.")

        # Compute mean and std for normalization
        self.y_mean = self.patches.mean(dim=0)
        self.y_std = self.patches.std(dim=0)
        if (self.y_std == 0).any():
            self.y_std[self.y_std == 0] = 1.0
            
        self.shape = self.patches.shape
        self.flow_dim = self.patches.shape[1]

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, index):
        return self.patches[index]

    def get_normalization_stats(self):
        return {'mean': self.y_mean, 'std': self.y_std}

    def set_normalization_stats(self, stats):
        self.y_mean = stats['mean']
        self.y_std = stats['std']
    
    def normalized_patches(self):
        """
        Returns the normalized patches.
        """
        return (self.patches - self.y_mean) / self.y_std


# =============================================================================
def piecewise_rational_quadratic_coupling_transform(
    iflow, input_size, hidden_size, num_blocks=1, activation=F.elu, num_bins=8
):
    """
    Creates a Piecewise Rational Quadratic Coupling Transform for use in
    normalizing flows.
    """
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(input_size, even=(iflow % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_size,
            num_blocks=num_blocks,
            activation=activation,
        ),
        num_bins=num_bins,
        tails="linear",
        tail_bound=5,
        apply_unconditional_transform=False,
    )


# =============================================================================
def masked_piecewise_rational_quadratic_autoregressive_transform(
    input_size, hidden_size, num_blocks=1, activation=F.elu, num_bins=8
):
    return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        features=input_size,
        hidden_features=hidden_size,
        num_bins=num_bins,
        tails="linear",
        tail_bound=6,
        use_residual_blocks=True,
        random_mask=False,
        activation=activation,
        num_blocks=num_blocks,
    )


# =============================================================================
def masked_umnn_autoregressive_transform(
    input_size, hidden_size, num_blocks=1, activation=F.elu
):
    """
    An unconstrained monotonic neural networks autoregressive layer that transforms the variables.
    """
    from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform

    return MaskedUMNNAutoregressiveTransform(
        features=input_size,
        hidden_features=hidden_size,
        use_residual_blocks=True,
        random_mask=False,
        activation=activation,
        num_blocks=num_blocks,
    )


# =============================================================================
def create_linear_transform(param_dim):
    """
    Creates a composite simple linear transformation with a random permutation.
    """
    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


# =============================================================================
def create_linear_transform_withActNorm(input_size):
    """Creates a linear transform with ActNorm, permutation, LU factorization, and a learnable bias."""
    return transforms.CompositeTransform([
        transforms.ActNorm(features=input_size),
        transforms.RandomPermutation(features=input_size),
        transforms.LULinear(input_size, identity_init=True)
        # Optionally, you can add a learnable bias here if desired
    ])


# =============================================================================
def create_flow(
    input_size=1, num_layers=5, hidden_features=32, num_bins=8, flow_type="PRQCT"
):
    """
    Creates a flow model.
    """
    base_dist = nflows.distributions.StandardNormal((input_size,))
    transformsi = []
    for i in range(num_layers):
        transformsi.append(create_linear_transform(param_dim=input_size))
        transformsi.append(
            piecewise_rational_quadratic_coupling_transform(
                i, input_size, hidden_features, num_bins=num_bins
            )
        )
    transformsi.append(create_linear_transform(param_dim=input_size))
    transformflow = CompositeTransform(transformsi)
    return Flow(transformflow, base_dist)


# =============================================================================
def create_flow_autoregressive(
    input_size=1, num_layers=5, hidden_features=32, num_bins=8
):
    """
    Creates an autoregressive flow model.
    """
    base_dist = nflows.distributions.StandardNormal((input_size,))
    transformsi = []
    for i in range(num_layers):
        transformsi.append(create_linear_transform(param_dim=input_size))
        transformsi.append(
            masked_umnn_autoregressive_transform(input_size, hidden_features)
        )
    transformsi.append(create_linear_transform(param_dim=input_size))
    transformflow = CompositeTransform(transformsi)
    return Flow(transformflow, base_dist)


# =============================================================================
def print_summary(model):
    """
    Prints a summary of the model.
    """
    pytorch_total_params_grad = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Total params to optimize:", pytorch_total_params_grad)


# =============================================================================
class FlowLogProbWrapper(nn.Module):
    """
    A wrapper for nflows.Flow models to make log_prob callable via the forward
    method, compatible with torch.nn.DataParallel.
    """

    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = (
            flow_model  # This ensures flow_model is registered as a child module
        )

    def forward(self, inputs, context=None):
        # This method is called by DataParallel, which will handle the inputs
        if context is not None:
            return self.flow_model.log_prob(inputs=inputs, context=context)
        return self.flow_model.log_prob(inputs=inputs)

    # Other methods of the original flow_model (e.g., sample, sample_and_log_prob)
    # can be exposed here if needed, or accessed via an unwrapped model.
    # For training with DataParallel focusing on log_prob, this is sufficient.


# =============================================================================
def configure_device(flow_wrapper, device, active_model):
    """
    Configures the device placement for a PyTorch model based on the specified
    device string. This function determines whether to use CPU or CUDA (GPU)
    devices for model training or inference.

    It supports flexible device string specifications, including:
        - 'cpu': Use CPU only.
        - 'cuda': Use all available GPUs.
        - 'cuda:X': Use a specific GPU (e.g., 'cuda:0').
        - 'cuda:X,Y,Z': Use multiple specified GPUs (e.g., 'cuda:0,2,3').
        - 'cuda:' or 'cuda: ': Treated as 'cuda', i.e., use all GPUs.
    The function validates GPU availability and indices, falls back to CPU if CUDA is unavailable or invalid indices are provided,
    and wraps the model in `nn.DataParallel` if multiple GPUs are selected.
    Args:
        model (torch.nn.Module): The PyTorch model to be placed on the selected device(s).
        device (str): Device specification string (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:0,1').
    Returns:
        Tuple[torch.nn.Module, torch.device]:
            - active_model: The model moved to the selected device(s), possibly wrapped in `nn.DataParallel`.
            - effective_primary_device: The primary `torch.device` used for computation and output.
    Notes:
        - Prints informative messages about device selection and any fallbacks.
        - Ensures only valid and available GPU indices are used.
        - Maintains user-specified GPU order for primary device selection.
        - If an invalid device string is provided, defaults to CPU.
    """
    if device == "cpu":
        effective_primary_device = torch.device("cpu")
        print(f"Device: Using CPU for training ({effective_primary_device}).")
        active_model = flow_wrapper.to(effective_primary_device)
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            effective_primary_device = torch.device("cpu")
            print(
                "Device: CUDA specified, but torch.cuda.is_available() is False. Falling back to CPU."
            )
            active_model = flow_wrapper.to(effective_primary_device)
        else:
            num_gpus_available = torch.cuda.device_count()
            if num_gpus_available == 0:
                effective_primary_device = torch.device("cpu")
                print(
                    "Device: CUDA available, but no GPUs detected (torch.cuda.device_count() == 0). Falling back to CPU."
                )
                active_model = flow_wrapper.to(effective_primary_device)
            else:  # CUDA and GPUs are available
                parts = device.split(":", 1)
                # gpu_selection_str is None for 'cuda', or '0' or '0,2,3' for 'cuda:0' or 'cuda:0,2,3'
                # .strip() handles cases like 'cuda: ' which should default to all GPUs
                gpu_selection_str = (
                    parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
                )

                target_gpu_ids_for_dp = []  # Integer GPU IDs for DataParallel's device_ids

                if (
                    gpu_selection_str is None
                ):  # Case: 'cuda' (or 'cuda:' or 'cuda: ') -> use all available
                    target_gpu_ids_for_dp = list(range(num_gpus_available))
                    print(
                        f"Device: '{device}' interpreted as using all {num_gpus_available} available GPUs: {target_gpu_ids_for_dp}."
                    )
                else:  # Case: 'cuda:X' or 'cuda:X,Y,Z'
                    try:
                        parsed_indices_str = gpu_selection_str.split(",")
                        temp_parsed_gpu_ids = []
                        for s_idx_str in parsed_indices_str:
                            s_idx_clean = s_idx_str.strip()
                            if not s_idx_clean:
                                continue  # Handles 'cuda:0,', 'cuda:,1' or 'cuda:,'
                            gpu_id = int(s_idx_clean)
                            if 0 <= gpu_id < num_gpus_available:
                                if (
                                    gpu_id not in temp_parsed_gpu_ids
                                ):  # Keep order, ensure uniqueness
                                    temp_parsed_gpu_ids.append(gpu_id)
                            else:
                                print(
                                    f"Warning: Specified GPU ID {gpu_id} is out of range (0-{num_gpus_available - 1}). Ignoring."
                                )

                        if (
                            not temp_parsed_gpu_ids
                        ):  # If all specified were invalid or string was e.g. 'cuda:,'
                            raise ValueError(
                                "No valid GPU indices derived from specification."
                            )
                        target_gpu_ids_for_dp = temp_parsed_gpu_ids
                        print(
                            f"Device: '{device}'. Validated target GPU IDs (maintaining user order for primary): {target_gpu_ids_for_dp}."
                        )
                    except ValueError as e:
                        print(
                            f"Warning: Could not parse GPU IDs from '{gpu_selection_str}' (Error: {e}). Falling back to CPU."
                        )
                        effective_primary_device = torch.device("cpu")
                        active_model = flow_wrapper.to(
                            effective_primary_device
                        )  # Mark for CPU

                # If GPU path is still viable (active_model not set to CPU fallback yet)
                if active_model is None:
                    if not target_gpu_ids_for_dp:
                        print(
                            "Warning: No target GPU indices selected despite CUDA availability. Falling back to CPU."
                        )
                        effective_primary_device = torch.device("cpu")
                        active_model = flow_wrapper.to(effective_primary_device)
                    else:
                        primary_gpu_id = target_gpu_ids_for_dp[
                            0
                        ]  # First valid GPU is primary
                        effective_primary_device = torch.device(
                            f"cuda:{primary_gpu_id}"
                        )

                        flow_wrapper.to(
                            effective_primary_device
                        )  # Move base wrapper to primary device

                        if len(target_gpu_ids_for_dp) > 1:
                            print(
                                f"Device: Using DataParallel for GPUs {target_gpu_ids_for_dp}. Primary/Output device: {effective_primary_device}."
                            )
                            active_model = nn.DataParallel(
                                flow_wrapper, device_ids=target_gpu_ids_for_dp
                            )
                            # DataParallel's output_device defaults to device_ids[0], matching our effective_primary_device
                        else:  # Single specified GPU
                            print(
                                f"Device: Using single specified GPU: {effective_primary_device}."
                            )
                            active_model = (
                                flow_wrapper  # Already on effective_primary_device
                            )
    else:  # Unrecognized device string format
        effective_primary_device = torch.device("cpu")
        print(
            f"Device: String '{device}' not recognized. Falling back to CPU ({effective_primary_device})."
        )
        active_model = flow_wrapper.to(effective_primary_device)
    return active_model, effective_primary_device


# =============================================================================
def train_flow(
    model,
    train_loader,
    learning_rate=1e-3,
    num_epochs=100,
    device="cpu",
    output_model=None,
    save_model=False,
    load_existing=False,
    extra_noise=1e-4,
):
    """
    Trains a normalizing flow model with flexible device selection.

    - 'cpu': Uses CPU.
    - 'cuda': Uses all available GPUs with DataParallel (primary cuda:0).
    - 'cuda:X': Uses only GPU X (e.g., 'cuda:0').
    - 'cuda:X,Y,Z': Uses specified GPUs X,Y,Z with DataParallel (primary X).
    If num_epochs=0 and load_existing=True, loads the existing model from output_model.
    """
    original_nflow_model = model

    # If requested, load existing model and skip training
    if num_epochs == 0 and load_existing and output_model is not None:
        try:
            original_nflow_model.load_state_dict(
                torch.load(output_model, map_location="cpu")
            )
            print(f"Loaded existing model from {output_model}")
            dict_info = {"model": original_nflow_model, "train_loss_avg": []}
            return dict_info
        except FileNotFoundError:
            print(f"No existing model found at {output_model}.")
            dict_info = {"model": original_nflow_model, "train_loss_avg": []}
            return dict_info

    # 1. Prepare the base model wrapper
    flow_wrapper = FlowLogProbWrapper(original_nflow_model)
    active_model = None  # Model (possibly wrapped) for the training loop
    effective_primary_device = (
        None  # torch.device for data loading and single-GPU model placement
    )

    # 2. Parse device string and configure devices
    active_model, effective_primary_device = configure_device(
        flow_wrapper, device, active_model
    )

    # 3. Optimizer, Data Normalization, Training Loop
    # Parameters of original_nflow_model are accessed through active_model
    optimizer = torch.optim.Adam(active_model.parameters(), lr=learning_rate)

    # Ensure dataset_lines is correctly handled (assuming it's tensor or numpy)
    y_std_val = train_loader.dataset.y_std.numpy()
    y_mean_val = train_loader.dataset.y_mean.numpy()

    y_std = torch.tensor(
        y_std_val, device=effective_primary_device, dtype=torch.float32
    )
    y_mean = torch.tensor(
        y_mean_val, device=effective_primary_device, dtype=torch.float32
    )

    train_loss_avg = []
    time0 = time.time()
    active_model.train()  # Set model to training mode

    for epoch in range(1, num_epochs + 1):
        train_loss = []
        t = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{num_epochs}",
        )
        for batch_idx, (splines_batch) in t:
            optimizer.zero_grad()
            if not isinstance(splines_batch, torch.Tensor):  # Ensure data is tensor
                splines_tensor = torch.from_numpy(splines_batch).float()
            else:
                splines_tensor = splines_batch.float()

            # Move data to the primary device; DataParallel will scatter if active
            y = (splines_tensor.to(effective_primary_device) - y_mean) / y_std

            if extra_noise is not None and extra_noise > 0:
                # Add small noise to the data to avoid numerical issues
                noise = torch.normal(
                    0, extra_noise, size=y.shape, device=effective_primary_device
                )
                y += noise

            log_probs = active_model(
                inputs=y
            )  # Call invokes FlowLogProbWrapper.forward
            loss = -log_probs.mean()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            t.set_postfix_str("Loss: {0:2.2f}".format(train_loss[-1]))

        current_epoch_avg_loss = np.mean(np.array(train_loss))
        train_loss_avg.append(current_epoch_avg_loss)
        # Update tqdm postfix with average loss and device info
        t.set_postfix_str(f"Avg: {current_epoch_avg_loss:.4f}")

    print(f"Completed training in {(time.time() - time0) / 60.0:2.2f} minutes.")

    # 4. Unwrap model and return
    original_nflow_model.to("cpu")  # Move the *original* model state to CPU

    dict_info = {"model": original_nflow_model, "train_loss_avg": train_loss_avg}

    # If save_model is True, save the model state:
    if save_model is True and output_model is not None:
        torch.save(original_nflow_model.state_dict(), output_model)
        print(f"Model saved to {output_model}")

    return dict_info


# =================================================================
def nume2string(num):
    """
    Convert number to scientific latex mode.
    """
    mantissa, exp = f"{num:.2e}".split("e")
    return mantissa + " \\times 10^{" + str(int(exp)) + "}"


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
    plt.ylabel("Loss: -log prob")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    if show_plot:
        plt.show()


# =============================================================================
def check_variables(
    model, train_loader, plot_variables=[0, 1], figsize=(8, 4), device="cpu"
):
    """
    Plots histograms of selected latent variables after transforming the input data with the model.

    Args:
        model: The trained model with a `transform_to_noise` method.
        train_loader: DataLoader containing the training dataset.
        plot_variables: List of variable indices to plot.
        figsize: Size of the plot.
        device: Device to use ('cpu', 'cuda', etc.).

    Shows the distribution of each selected variable compared to a standard normal.
    """
    # Move model to device
    model = model.to(device)
    model.eval()

    y_mean = train_loader.dataset.patches.mean(dim=0)
    y_std = train_loader.dataset.patches.std(dim=0)
    inputs = (train_loader.dataset.patches - y_mean) / y_std
    inputs = inputs.to(device)

    with torch.no_grad():
        zz = model.transform_to_noise(inputs).cpu().numpy()

    # If plot_variables is larger than the number of features, adjust it
    if len(plot_variables) > zz.shape[1]:
        print(
            f"Warning: plot_variables length {len(plot_variables)} exceeds number of features {zz.shape[1]}. Adjusting to available features."
        )
        plot_variables = list(range(zz.shape[1]))
    fig, axes = plt.subplots(1, len(plot_variables), figsize=figsize, sharey=True)
    if len(plot_variables) == 1:
        axes = [axes]
    for i, var in enumerate(plot_variables):
        ax = axes[i]
        ax.hist(zz[:, var], bins=100, density=True, alpha=0.5, label=f"Variable {var}")
        ax.set_title(f"Variable {var} distribution")
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Density")
        x = np.linspace(zz[:, var].min(), zz[:, var].max(), 100)
        ax.plot(x, norm.pdf(x, 0, 1), "k--", label="Normal Gaussian")
        ax.locator_params(axis="y", nbins=6)
    plt.tight_layout()
