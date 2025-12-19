import torch
import torch.utils.data
import numpy as np
from einops import rearrange


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
            # Local import to prevent circular dependencies
            from inspectorch import dim_reduction as dr

            self.patches = dr.apply_dim_reduction(self.patches, dim_reduction)

        # Compute mean and std for normalization
        # Local import to prevent circular dependencies
        from inspectorch.utils import nanstd

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
