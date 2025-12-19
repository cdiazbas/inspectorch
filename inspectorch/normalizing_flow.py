import torch
import torch.nn as nn
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
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import norm
from typing import Optional, List, Union, Tuple, Dict
from . import utils as iu
from . import datasets


# =============================================================================
# Utility Functions (Reused from flowutils.py)
# =============================================================================
dot_dict = iu.dot_dict
nanstd = iu.nanstd
nanvar = iu.nanvar
nume2string = iu.nume2string
GeneralizedPatchedDataset = datasets.GeneralizedPatchedDataset


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
    An unconstrained monotonic neural networks autoregressive layer that
    transforms the variables.
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
    """
    Creates a linear transform with ActNorm, permutation, LU factorization, and
    a learnable bias.
    """
    return transforms.CompositeTransform(
        [
            transforms.ActNorm(features=input_size),
            transforms.RandomPermutation(features=input_size),
            transforms.LULinear(input_size, identity_init=True),
            # Optionally, you can add a learnable bias here if desired
        ]
    )


# =============================================================================
class NormalizingFlowBackend(nn.Module):
    """
    NormalizingFlowBackend is a PyTorch module for density estimation using
    normalizing flows.

    Attributes:
        flow_model (nn.Module): The flow-based model for density estimation.
        y_mean (torch.Tensor or None): Mean of the target variable, used for normalization.
        y_std (torch.Tensor or None): Standard deviation of the target variable, used for normalization.
        training_loss (List[float]): List to store training loss values during flow training.

    Methods:
        create_flow(input_size, num_layers, hidden_features, num_bins):
            Initializes the flow_model with the specified architecture.

        log_prob(inputs, dataset_normalization=True):
            Computes the log-probability of the inputs under the flow model.
            If dataset_normalization is True, normalizes the inputs before evaluation.

        print_summary():
            Prints a summary of the flow model architecture.

        check_variables(train_loader, plot_variables=[0, 1], figsize=(8, 4), device="cpu", batch_size=500000, rel_size=0.1):
            Checks and optionally plots variables from the training data for diagnostics.

        train_flow(train_loader, learning_rate=1e-3, num_epochs=100, device="cpu", output_model=None, save_model=False, load_existing=False, extra_noise=1e-4):
            Trains the flow model using the provided training data loader and hyperparameters.
            Stores training loss in self.training_loss.

        plot_train_loss(show_plot=False, save_path=None):
            Plots the training loss curve. Optionally displays or saves the plot.
    """

    def __init__(self):
        super(NormalizingFlowBackend, self).__init__()
        # Flow model will be created later
        self.flow_model: Optional[nn.Module] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        # Extra attributes to store information
        self.training_loss: List[float] = []

    def create_flow(
        self, input_size: int, num_layers: int, hidden_features: int, num_bins: int
    ) -> None:
        """
        Creates a flow-based model for density estimation or generative
        modeling.

        Args:
            input_size (int): The dimensionality of the input data.
            num_layers (int): Number of layers in the flow model.
            hidden_features (int): Number of hidden units in each layer.
            num_bins (int): Number of bins used for the flow's transformation (e.g., in rational quadratic splines).

        Returns:
            nn.Module: A flow-based model instance configured with the specified parameters.
        """
        self.flow_model = create_flow(input_size, num_layers, hidden_features, num_bins)

    def log_prob(
        self,
        inputs: Union[torch.Tensor, np.ndarray, object],
        dataset_normalization: bool = True,
        batch_size: int = 100_000,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Computes the log probability of the input data using the flow model.

        Args:
            inputs: The input data to evaluate. If `dataset_normalization` is True,
                it is expected to have a `normalized_patches()` method that returns
                the normalized data.
            dataset_normalization (bool, optional): If True, applies dataset normalization
                to the inputs before computing log probability. Defaults to True.

        Returns:
            numpy.ndarray: The log probabilities of the inputs as a NumPy array.
        """
        if dataset_normalization:
            inputs = inputs.normalized_patches()

        else:
            inputs = inputs.patches

        results = []
        print(f"Using {device} for log_prob computation.")
        # Move model to the specified device
        self.flow_model.to(device)
        self.flow_model.eval()  # Set model to evaluation mode

        from tqdm import tqdm

        for i in tqdm(range(0, inputs.shape[0], batch_size)):
            batch = inputs[i : i + batch_size].to(
                next(self.flow_model.parameters()).device
            )
            results.append(self.flow_model.log_prob(batch).detach().cpu().numpy())

        # Move model back to CPU if needed
        self.flow_model.to("cpu")

        return np.concatenate(results)

    def print_summary(self):
        """
        Prints a summary of the flow model.

        This method calls the `print_summary` function, passing the `flow_model` attribute
        as an argument. The summary typically includes key information about the flow model,
        such as its structure, parameters, and configuration.
        """
        print_summary(self.flow_model)

    def check_variables(
        self,
        train_loader,
        plot_variables=[0, 1],
        figsize=(8, 4),
        device="cpu",
        batch_size=500000,
        rel_size=0.1,
    ):
        check_variables(
            self.flow_model,
            train_loader,
            plot_variables,
            figsize,
            device,
            batch_size,
            rel_size,
        )

    def train_flow(
        self,
        train_loader: torch.utils.data.DataLoader,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        device: str = "cpu",
        output_model: Optional[str] = None,
        save_model: bool = False,
        load_existing: bool = False,
        extra_noise: float = 1e-3,
    ) -> None:
        if train_loader is not None:
            self.y_mean = train_loader.dataset.y_mean
            self.y_std = train_loader.dataset.y_std

        train_flow(
            self.flow_model,
            train_loader,
            learning_rate,
            num_epochs,
            device,
            output_model,
            save_model,
            load_existing,
            extra_noise,
            self.training_loss,
            {"mean": self.y_mean, "std": self.y_std}
            if self.y_mean is not None
            else None,
        )

    def plot_train_loss(self, show_plot=False, save_path=None):
        """
        Plots the training loss over epochs.

        Args:
            show_plot (bool, optional): If True, displays the plot. Defaults to False.
            save_path (str or None, optional): If provided, saves the plot to the specified file path. Defaults to None.
        """
        plot_train_loss(self.training_loss, show_plot, save_path)


# =============================================================================
def create_flow(
    input_size=1, num_layers=5, hidden_features=32, num_bins=8, flow_type="PRQCT"
):
    """
    Creates a flow model.
    """
    if input_size > 1.5:
        return create_flow_prqct(
            input_size=input_size,
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_bins=num_bins,
        )
    else:
        return create_flow_autoregressive(
            input_size=input_size,
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_bins=num_bins,
        )


# =============================================================================
def create_flow_prqct(
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
    print(f"Total params to optimize: {pytorch_total_params_grad:,}")


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
# =============================================================================
configure_device = iu.configure_device


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
    train_loss_avg=[],
    normalization_stats=None,
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
    active_model, effective_primary_device = configure_device(flow_wrapper, device)

    # 3. Optimizer, Data Normalization, Training Loop
    # Parameters of original_nflow_model are accessed through active_model
    optimizer = torch.optim.Adam(active_model.parameters(), lr=learning_rate)

    # Ensure dataset_lines is correctly handled (assuming it's tensor or numpy)
    if train_loader is None:
        y_std_val = train_loader.y_std.numpy()
        y_mean_val = train_loader.y_mean.numpy()
    else:
        if normalization_stats is not None:
            y_std_val = normalization_stats["std"].numpy()
            y_mean_val = normalization_stats["mean"].numpy()
        else:
            raise ValueError(
                "Normalization stats must be provided if train_loader is None."
            )

    y_std = torch.tensor(
        y_std_val, device=effective_primary_device, dtype=torch.float32
    )
    y_mean = torch.tensor(
        y_mean_val, device=effective_primary_device, dtype=torch.float32
    )

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

            # Crop out any NaNs in the batch:
            splines_tensor = splines_tensor[~torch.isnan(splines_tensor).any(dim=1)]

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

        # Save intermediate model every epoch:
        if save_model and output_model is not None:
            torch.save(original_nflow_model.state_dict(), output_model)

    print(f"Completed training in {(time.time() - time0) / 60.0:2.2f} minutes.")

    # 4. Unwrap model and return
    original_nflow_model.to("cpu")  # Move the *original* model state to CPU

    # If save_model is True, save the model state:
    if save_model is True and output_model is not None:
        torch.save(original_nflow_model.state_dict(), output_model)
        print(f"Model saved to {output_model}")

    return


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
    plt.ylabel(r"$- \log p_{\phi}(x)$")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    if show_plot:
        plt.show()


# =============================================================================
def check_variables_loader(
    model,
    train_loader,
    plot_variables=[0, 1],
    figsize=(8, 4),
    device="cpu",
    batch_size=500000,
    rel_size=0.1,
):
    """
    Plots histograms of selected latent variables after transforming the input
    data with the model.

    Args:
        model: The trained model with a `transform_to_noise` method.
        train_loader: DataLoader containing the training dataset.
        plot_variables: List of variable indices to plot.
        figsize: Size of the plot.
        device: Device to use ('cpu', 'cuda', etc.).
        batch_size (int, optional): Unused in this function. Included for compatibility. Defaults to 500000.
        rel_size (float, optional): Fraction of the dataset to use for plotting (between 0 and 1). Defaults to 0.1.

    Shows the distribution of each selected variable compared to a standard normal.
    """
    # Move model to device
    model = model.to(device)
    model.eval()

    y_mean = train_loader.dataset.y_mean.to(device)
    y_std = train_loader.dataset.y_std.to(device)

    # Evaluate only rel_size % of the dataset using the train_loader, with extra shuffle
    # Process in batches - more memory efficient
    zz_batches = []
    total_samples_needed = int(rel_size * len(train_loader.dataset))
    samples_collected = 0

    # DataLoader iterator
    dataloader_iter = iter(train_loader)

    with torch.no_grad():
        while samples_collected < total_samples_needed:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                print("DataLoader exhausted, breaking...")
                break

            if samples_collected >= total_samples_needed:
                break

            if not isinstance(batch, torch.Tensor):
                batch = torch.from_numpy(batch).float()

            # Remove samples with NaNs
            valid_mask = ~torch.isnan(batch).any(dim=1)
            batch_clean = batch[valid_mask]

            if batch_clean.size(0) == 0:
                print("Empty batch after NaN removal, continuing...")
                continue

            # Only take what we need
            samples_to_take = min(
                batch_clean.size(0), total_samples_needed - samples_collected
            )
            batch_subset = batch_clean[:samples_to_take].to(device)

            # Transform this batch
            zz_batch = (
                model.transform_to_noise((batch_subset - y_mean) / y_std)
                .cpu()
                .detach()
                .numpy()
            )
            zz_batches.append(zz_batch)

            samples_collected += samples_to_take

    if not zz_batches:
        raise ValueError("No valid samples found after NaN removal!")

    zz = np.concatenate(zz_batches)

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
        ax.set_title(f"Variable {var}")
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Density")
        x = np.linspace(zz[:, var].min(), zz[:, var].max(), 100)
        ax.plot(x, norm.pdf(x, 0, 1), "k--", label="Normal Gaussian")
        ax.locator_params(axis="y", nbins=6)
    plt.tight_layout()


# =============================================================================
def check_variables(
    model,
    train_loader,
    plot_variables=[0, 1],
    figsize=(8, 4),
    device="cpu",
    batch_size=50000,  # Now this parameter is actually used!
    rel_size=0.1,
):
    """
    Plots histograms of selected latent variables after transforming the input
    data with the model.

    Args:
        model: The trained model with a `transform_to_noise` method.
        train_loader: DataLoader containing the training dataset.
        plot_variables: List of variable indices to plot.
        figsize: Size of the plot.
        device: Device to use ('cpu', 'cuda', etc.).
        batch_size (int): Batch size for processing samples. Defaults to 50000.
        rel_size (float, optional): Fraction of the dataset to use for plotting (between 0 and 1). Defaults to 0.1.

    Shows the distribution of each selected variable compared to a standard normal.
    """
    import time

    # Move model to device
    time.time()
    model = model.to(device)
    model.eval()

    y_mean = train_loader.dataset.y_mean.to(device)
    y_std = train_loader.dataset.y_std.to(device)

    # Calculate total samples needed
    dataset_size = len(train_loader.dataset)
    total_samples_needed = int(rel_size * dataset_size)

    # Step 1: Randomly sample indices
    time.time()
    random_indices = np.random.randint(0, dataset_size, size=total_samples_needed)

    # Step 2: Process in batches
    zz_batches = []
    num_batches = (
        total_samples_needed + batch_size - 1
    ) // batch_size  # Ceiling division

    with torch.no_grad():
        for batch_idx in range(num_batches):
            time.time()

            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples_needed)
            batch_indices = random_indices[start_idx:end_idx]

            # Step 3: Collect samples for this batch - VECTORIZED VERSION

            # Get all samples at once using advanced indexing
            batch_tensor = train_loader.dataset.patches[batch_indices]

            # Vectorized NaN check - check which samples have NaNs
            nan_mask = torch.isnan(batch_tensor).any(dim=1)
            valid_samples = batch_tensor[~nan_mask]

            if valid_samples.size(0) == 0:
                print("No valid samples in this batch, skipping...")
                continue

            # Move to device
            valid_samples = valid_samples.to(device)

            # Step 4: Transform this batch
            zz_batch = (
                model.transform_to_noise((valid_samples - y_mean) / y_std)
                .cpu()
                .detach()
                .numpy()
            )
            zz_batches.append(zz_batch)

    if not zz_batches:
        raise ValueError("No valid samples found after NaN removal!")

    # Step 5: Concatenate results
    zz = np.concatenate(zz_batches)

    # Step 6: Plot results
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
        ax.hist(zz[:, var], bins=100, density=True, alpha=0.5, label="Output")
        ax.set_title(f"Latent variable {var}")
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Density")
        # Plot standard normal reference
        x_range = np.linspace(zz[:, var].min(), zz[:, var].max(), 100)
        ax.plot(
            x_range,
            norm.pdf(x_range, 0, 1),
            "k--",
            linewidth=2,
            label=r"$\mathcal{N}(0,1)$",
        )
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        ax.locator_params(axis="y", nbins=6)
    plt.tight_layout()
