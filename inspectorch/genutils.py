import platform
import socket
import torch
import torch.nn as nn
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange


# =============================================================================
def device_info():
    """
    Prints information about the current machine and available CUDA devices.

    If CUDA is available, this function prints the machine name, operating system,
    the number of available GPUs, and the name of each GPU. If CUDA is not available,
    it notifies the user that the CPU will be used instead.

    Requires:
        - torch
        - socket
        - platform
    """
    if torch.cuda.is_available():
        print(f"Machine name: {socket.gethostname()}")
        print(f"Operating system: {platform.system()} {platform.release()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(
                f"GPU {i}: {torch.cuda.get_device_name(i)} -",
                f"Memory: {allocated:.2f}/{total:.2f} GB",
            )
    else:
        print("CUDA is not available. Using CPU.")


# =============================================================================
def save_json(args, filename):
    """
    Saves the given arguments as a JSON file.

    Args:
        args (dict or iterable): The arguments to save. Should be convertible to a dictionary.
        filename (str): The path to the JSON file where the arguments will be saved.

    Notes:
        - If the directory for the given filename does not exist, it will be created.
        - The JSON file will be formatted with an indentation of 4 spaces.
    """
    import json

    args_dict = dict(args)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        json.dump(args_dict, f, indent=4)


# =============================================================================
def create_spectral_to_white_cmap():
    """
    Creates a colormap that uses the Red-Orange-Yellow transition from the
    'Spectral' colormap and then smoothly transitions from that yellow to
    white.
    """
    # 1. Get the original 'Spectral' colormap
    # We get it with 256 discrete colors
    spectral_map = plt.cm.get_cmap("Spectral", 256)

    # 2. Isolate the first half (Red -> Yellow)
    # The 'Spectral' map goes from Red to Yellow in its first half.
    # We take the first 128 colors to capture this transition.
    spectral_first_half = spectral_map(np.linspace(0, 0.45, 128))

    # 3. The endpoint of our first segment is the specific 'Spectral' yellow
    end_yellow = spectral_first_half[-1]

    # 4. Create the second half of the colormap: from our specific yellow to white
    # We create a small colormap segment for this transition, also with 128 steps.
    yellow_to_white_segment = LinearSegmentedColormap.from_list(
        "y2w_custom",
        [end_yellow, (1, 1, 1, 1)],  # (1,1,1,1) is RGBA for white
        N=128,
    )
    # Get the color values from this new segment
    yellow_to_white_colors = yellow_to_white_segment(np.arange(128))

    # 5. Combine the two lists of colors
    # We stack the first half on top of the second half.
    # We use np.vstack to join the two color arrays.
    final_colors = np.vstack((spectral_first_half, yellow_to_white_colors))

    # 6. Create the final, combined colormap
    custom_cmap = LinearSegmentedColormap.from_list("SpectralToWhite", final_colors)

    return custom_cmap


# Create our new colormap
spectral_to_white = create_spectral_to_white_cmap()


# =============================================================================
# Utility Functions
# =============================================================================


class dot_dict(dict):
    """
    A dictionary subclass that allows for attribute-style access.
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def nanstd(tensor, dim=None, keepdim=False):
    """
    Computes the standard deviation of a tensor, ignoring NaN values.
    """
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


def nanvar(tensor, dim=None, keepdim=False):
    """
    Computes the variance of a tensor, ignoring NaN values.
    """
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nume2string(num):
    """
    Convert number to scientific latex mode.
    """
    mantissa, exp = f"{num:.2e}".split("e")
    return mantissa + " \\times 10^{" + str(int(exp)) + "}"


# =============================================================================
class GeneralizedPatchedDataset(torch.utils.data.Dataset):
    """
    A highly flexible PyTorch Dataset that uses named dimensions and einops to
    extract patches and prepare data for a normalizing flow.

    Includes automatic padding to preserve spatial/temporal dimensions.
    """

    def __init__(
        self,
        data,
        dim_names,
        feature_dims,
        patch_config=None,
        dim_reduction=None,
    ):
        """
        Initializes the dataset, performs patch extraction and reshaping.

        Args:
            data (torch.Tensor or np.ndarray): The input data tensor.
            dim_names (str): A space-separated string of dimension names.
            feature_dims (list): List of dimension names for the flow's features.
            patch_config (dict, optional): Configures patching on primary variables, e.g, {'x': {'size': 5, 'stride': 1}, 'y': {'size': 5, 'stride': 1}} or in the temporal case  {'t': {'size': 11, 'stride': 1}}
            dim_reduction (dict, optional): Dimensionality reduction config, e.g. {'method': 'pca', 'n_components': 10}
        """
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data.astype(np.float32, copy=False)).float()

        patch_config = patch_config or {}
        all_dims = dim_names.split()

        if data.dim() != len(all_dims):
            raise ValueError(
                f"Data tensor has {data.dim()} dims, but "
                f"dim_names defines {len(all_dims)}."
            )
        # 0. Save the dimension of each variable:
        self.data_dims = {f"n{d}": data.shape[i] for i, d in enumerate(all_dims)}

        # 1. Determine the role of each dimension
        sample_dims = [d for d in all_dims if d not in feature_dims]
        patch_dims = [d for d in feature_dims if d in patch_config]
        non_patch_feature_dims = [d for d in feature_dims if d not in patch_config]

        # 2. Permute data to group dimensions by role (others, then patchable)
        # This makes the unfolding process predictable.
        other_dims = sample_dims + non_patch_feature_dims
        permute_pattern = (
            f"{' '.join(all_dims)} -> {' '.join(other_dims)} {' '.join(patch_dims)}"
        )
        permuted_data = rearrange(data, permute_pattern)

        # 3. Iteratively unfold the patchable dimensions
        unfolded_data = permuted_data
        num_other_dims = len(other_dims)
        for i, dim_name in enumerate(patch_dims):
            axis_to_unfold = num_other_dims + i
            size = patch_config[dim_name]["size"]
            stride = patch_config[dim_name].get("stride", 1)
            unfolded_data = unfolded_data.unfold(axis_to_unfold, size, stride)

        # 4. Construct the final rearrangement pattern to create the patches
        # The shape of unfolded_data is now: (*other_dims, *num_patches, *patch_sizes)
        sample_str = " ".join(sample_dims)
        feature_str = " ".join(non_patch_feature_dims)
        num_patches_str = " ".join([f"n_{d}" for d in patch_dims])
        patch_size_str = " ".join([f"p_{d}" for d in patch_dims])

        input_pattern = f"{sample_str} {feature_str} {num_patches_str} {patch_size_str}"
        output_pattern = (
            f"-> ({sample_str} {num_patches_str}) ({feature_str} {patch_size_str})"
        )

        self.input_pattern = input_pattern
        self.output_pattern = output_pattern
        self.patches = rearrange(unfolded_data, f"{input_pattern} {output_pattern}")

        print(f"Dataset initialized with {self.patches.shape[0]} samples.")
        print(f"Each sample is a flattened vector of size {self.patches.shape[1]}.")

        print(f"Input pattern: {self.input_pattern} -> {self.output_pattern}")

        if dim_reduction is not None:
            # Local import to prevent circular dependencies if dimreduction imports genutils
            # Assuming inspectorch package structure
            try:
                import inspectorch.dimreduction as dr
            except ImportError:
                import dimreduction as dr

            self.patches = dr.apply_dim_reduction(self.patches, dim_reduction)

        # Compute mean and std for normalization
        self.y_mean = torch.nanmean(self.patches, dim=0)
        self.y_std = nanstd(self.patches, dim=0)
        if (self.y_std == 0).any():
            self.y_std[self.y_std == 0] = 1.0

        self.shape = self.patches.shape
        self.flow_dim = self.patches.shape[1]

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, index):
        return self.patches[index]

    def get_normalization_stats(self):
        return {"mean": self.y_mean, "std": self.y_std}

    def set_normalization_stats(self, stats):
        self.y_mean = stats["mean"]
        self.y_std = stats["std"]

    def normalized_patches(self):
        """
        Returns the normalized patches.
        """
        return (self.patches - self.y_mean) / self.y_std


# =============================================================================
def configure_device(model_wrapper, device):
    """
    Configures device placement for training with support for multi-GPU.

    Args:
        model_wrapper: Wrapped model
        device: Device string ('cpu', 'cuda', 'cuda:0', 'cuda:0,1,2', etc.)

    Returns:
        Tuple of (active_model, effective_primary_device)
    """
    if device == "cpu":
        effective_primary_device = torch.device("cpu")
        print(f"Device: Using CPU for training ({effective_primary_device}).")
        active_model = model_wrapper.to(effective_primary_device)

    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            effective_primary_device = torch.device("cpu")
            print(
                "Device: CUDA specified, but torch.cuda.is_available() is False. Falling back to CPU."
            )
            active_model = model_wrapper.to(effective_primary_device)
        else:
            num_gpus_available = torch.cuda.device_count()
            if num_gpus_available == 0:
                effective_primary_device = torch.device("cpu")
                print(
                    "Device: CUDA specified, but no GPUs detected. Falling back to CPU."
                )
                active_model = model_wrapper.to(effective_primary_device)
            else:
                # Parse device string
                device_clean = device.strip()

                if device_clean in ["cuda", "cuda:"]:
                    # Use all available GPUs
                    gpu_ids = list(range(num_gpus_available))
                    primary_gpu = 0

                elif "," in device_clean:
                    # Multiple specific GPUs: 'cuda:0,1,2'
                    try:
                        device_part = (
                            device_clean.split(":", 1)[1]
                            if ":" in device_clean
                            else device_clean
                        )
                        gpu_ids = [int(g.strip()) for g in device_part.split(",")]

                        # Validate GPU IDs
                        valid_gpu_ids = [
                            g for g in gpu_ids if 0 <= g < num_gpus_available
                        ]
                        if not valid_gpu_ids:
                            print(
                                f"Device: No valid GPU IDs in {gpu_ids}. Falling back to CPU."
                            )
                            effective_primary_device = torch.device("cpu")
                            active_model = model_wrapper.to(effective_primary_device)
                            return active_model, effective_primary_device

                        gpu_ids = valid_gpu_ids
                        primary_gpu = gpu_ids[0]
                    except ValueError:
                        print(
                            f"Device: Could not parse GPU IDs from '{device_clean}'. Using all GPUs."
                        )
                        gpu_ids = list(range(num_gpus_available))
                        primary_gpu = 0

                else:
                    # Single GPU: 'cuda:0'
                    try:
                        if ":" in device_clean:
                            primary_gpu = int(device_clean.split(":")[1])
                        else:
                            primary_gpu = 0

                        if primary_gpu >= num_gpus_available or primary_gpu < 0:
                            print(
                                f"Device: GPU {primary_gpu} not available (only {num_gpus_available} GPUs). Using GPU 0."
                            )
                            primary_gpu = 0

                        gpu_ids = [primary_gpu]
                    except (ValueError, IndexError):
                        print(f"Device: Could not parse '{device_clean}'. Using GPU 0.")
                        gpu_ids = [0]
                        primary_gpu = 0

                effective_primary_device = torch.device(f"cuda:{primary_gpu}")

                # Move model to primary device first
                model_wrapper = model_wrapper.to(effective_primary_device)

                # Use DataParallel if multiple GPUs
                if len(gpu_ids) > 1:
                    print(
                        f"Device: Using DataParallel on GPUs {gpu_ids} (primary: cuda:{primary_gpu})"
                    )
                    active_model = nn.DataParallel(model_wrapper, device_ids=gpu_ids)
                else:
                    print(f"Device: Using single GPU cuda:{primary_gpu}")
                    active_model = model_wrapper
    else:
        print(f"Device: Unknown device string '{device}'. Falling back to CPU.")
        effective_primary_device = torch.device("cpu")
        active_model = model_wrapper.to(effective_primary_device)

    return active_model, effective_primary_device
