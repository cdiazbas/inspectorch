import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelSummary

import nflows
from nflows import transforms, utils
from nflows.transforms import CompositeTransform
from nflows.nn import nets
from nflows.flows.base import Flow

import numpy as np
import time # Kept for potential direct use, though Trainer handles timing
import matplotlib.pyplot as plt # Kept for plotting

# =============================================================================
# Utility Classes and Functions (Largely Unchanged)
# =============================================================================
class dot_dict(dict):
    """
    A dictionary subclass that allows for attribute-style access.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# =============================================================================
class custom_dataset(Dataset): # Inherits from torch.utils.data.Dataset
    """
    A custom Dataset class for PyTorch.
    """
    def __init__(self, lines):
        'Initialization'
        self.lines = lines

    def __len__(self):
        'Denotes the total number of samples'
        return self.lines.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Ensure data is float32, common requirement for PyTorch models
        return torch.tensor(self.lines[index, :], dtype=torch.float32)

# =============================================================================
# Transform and Flow Creation Functions (Unchanged - these define the model architecture)
# =============================================================================
def piecewise_rational_quadratic_coupling_transform(iflow, input_size, hidden_size, num_blocks=1, activation=F.elu, num_bins=8):
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(input_size, even=(iflow % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_size,
            num_blocks=num_blocks,
            activation=activation
        ),
        num_bins=num_bins,
        tails='linear',
        tail_bound=5,
        apply_unconditional_transform=False
    )

def masked_piecewise_rational_quadratic_autoregressive_transform(input_size, hidden_size, num_blocks=1, activation=F.elu, num_bins=8):
    return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=input_size,
        hidden_features=hidden_size,
        num_bins=num_bins,
        tails='linear',
        tail_bound=6,
        use_residual_blocks=True,
        random_mask=False,
        activation=activation,
        num_blocks=num_blocks,
    )

def masked_umnn_autoregressive_transform(input_size, hidden_size, num_blocks=1, activation=F.elu):
    from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform # Keep import local if specific
    return MaskedUMNNAutoregressiveTransform(features=input_size,
        hidden_features=hidden_size,
        use_residual_blocks=True,
        random_mask=False,
        activation=activation,
        num_blocks=num_blocks,
    )

def create_linear_transform(param_dim):
    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=param_dim),
        transforms.LULinear(param_dim, identity_init=True)
    ])

def create_flow(input_size=1, num_layers=5, hidden_features=32, num_bins=8, flow_type='PRQCT'): # Added flow_type for flexibility
    base_dist = nflows.distributions.StandardNormal((input_size,))
    transformsi = []
    for i in range(num_layers):
        transformsi.append(create_linear_transform(param_dim=input_size))
        if flow_type == 'PRQCT':
            transformsi.append(piecewise_rational_quadratic_coupling_transform(i, input_size, hidden_features, num_bins=num_bins))
        elif flow_type == 'MPRQAT': # Masked Piecewise Rational Quadratic Autoregressive Transform
             transformsi.append(masked_piecewise_rational_quadratic_autoregressive_transform(input_size, hidden_features, num_bins=num_bins))
        elif flow_type == 'MUMNNAT': # Masked UMNN Autoregressive Transform
             transformsi.append(masked_umnn_autoregressive_transform(input_size, hidden_features))
        else:
            raise ValueError(f"Unknown flow_type: {flow_type}")
    transformsi.append(create_linear_transform(param_dim=input_size))
    transformflow = CompositeTransform(transformsi)
    return Flow(transformflow, base_dist)

# Removed create_flow_autoregressive as create_flow now handles different types via flow_type

# =============================================================================
# PyTorch Lightning Module
# =============================================================================
class FlowLightningModule(pl.LightningModule):
    def __init__(self, input_size=1, num_layers=5, hidden_features=32, num_bins=8,
                 flow_type='PRQCT', learning_rate=1e-3, batch_size=64): # Added batch_size for DataModule
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args to self.hparams

        self.model = create_flow(
            input_size=self.hparams.input_size,
            num_layers=self.hparams.num_layers,
            hidden_features=self.hparams.hidden_features,
            num_bins=self.hparams.num_bins,
            flow_type=self.hparams.flow_type
        )
        # For data normalization, calculated in setup()
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

        self.train_loss_avg_epoch = [] # To replicate original plotting if needed

    def setup(self, stage: str):
        if stage == "fit":
            # Calculate mean and std from the training dataset
            # This assumes you have a trainer with a datamodule or train_dataloader attached
            if self.trainer.datamodule:
                train_dataset = self.trainer.datamodule.train_dataloader().dataset.lines
            elif self.trainer.train_dataloader: # Accessing internal attribute, prefer datamodule
                 # This is a bit fragile as it accesses an internal attribute.
                 # It's better to use a DataModule or pass dataset stats during init.
                try:
                    train_dataset_lines = self.trainer.train_dataloader.dataset.lines
                except AttributeError: # If train_dataloader is a list of dataloaders
                    train_dataset_lines = self.trainer.train_dataloader.loaders.dataset.lines

                if isinstance(train_dataset_lines, np.ndarray):
                    train_dataset = torch.from_numpy(train_dataset_lines)
                else: # Assuming it's already a tensor or similar
                    train_dataset = train_dataset_lines

            else:
                # Fallback or error if no way to access training data for stats
                # For simplicity, using placeholder if no dataloader is found during setup.
                # In a real scenario, you'd ensure data is available or handle this error.
                print("Warning: Training data not available at setup phase for normalization. Using default 0/1.")
                print("It is highly recommended to use a LightningDataModule or pass dataset statistics directly.")
                return

            if isinstance(train_dataset, np.ndarray): # Convert to tensor if it's numpy
                 train_dataset_tensor = torch.from_numpy(train_dataset.astype(np.float32))
            else:
                 train_dataset_tensor = train_dataset.float()


            self.y_mean = train_dataset_tensor.mean(dim=0).mean() # Mean across features and samples
            self.y_std = train_dataset_tensor.std(dim=0).mean()   # Std across features and samples
            if self.y_std == 0: self.y_std = torch.tensor(1.0) # Avoid division by zero

            print(f"Calculated normalization constants: y_mean={self.y_mean.item():.4f}, y_std={self.y_std.item():.4f}")


    def forward(self, inputs):
        # The "forward" of a flow for density estimation is typically log_prob
        return self.model.log_prob(inputs)

    def training_step(self, batch, batch_idx):
        # batch is already on the correct device thanks to Lightning
        y_normalized = (batch - self.y_mean) / self.y_std
        loss = -self.model.log_prob(inputs=y_normalized).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        # To replicate the original `train_loss_avg` list for plotting
        # This uses the logged value which is an average over steps in the epoch
        epoch_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        if epoch_loss is not None:
            self.train_loss_avg_epoch.append(epoch_loss.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def sample(self, num_samples):
        """Convenience method to sample from the flow"""
        self.model.eval() # Ensure model is in eval mode for sampling
        with torch.no_grad():
            samples_normalized = self.model.sample(num_samples)
            # Denormalize samples
            samples = samples_normalized * self.y_std.to(samples_normalized.device) + self.y_mean.to(samples_normalized.device)
        return samples

# =============================================================================
# PyTorch Lightning DataModule (Optional but Recommended)
# =============================================================================
class FlowDataModule(pl.LightningDataModule):
    def __init__(self, data_array: np.ndarray, batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.data_array = data_array
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None # Initialized in setup

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = custom_dataset(self.data_array)
            # You could add validation/test splits here if needed
            # For example:
            # num_samples = len(self.data_array)
            # train_size = int(0.8 * num_samples)
            # val_size = num_samples - train_size
            # self.train_data, self.val_data = torch.utils.data.random_split(
            #     custom_dataset(self.data_array), [train_size, val_size]
            # )

    def train_dataloader(self):
        if self.train_dataset is None:
             # This might happen if trainer.fit is called before trainer.datamodule is fully set up
             # or if setup was not called for 'fit' stage.
             self.setup('fit') # Ensure dataset is created
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    # Define val_dataloader() and test_dataloader() if you have validation/test data

# =============================================================================
# Utility Functions (Largely Unchanged, but print_summary might need slight adaptation)
# =============================================================================
def print_lightning_model_summary(model: pl.LightningModule):
    """
    Prints a summary of the Lightning model including the total number of parameters.
    Uses PyTorch Lightning's ModelSummary callback for a detailed view.
    """
    summary = ModelSummary(max_depth=-1) # -1 for full depth
    summary.on_train_start(None, model) # Manually trigger summary generation
    pytorch_total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params to optimize (from manual count): {pytorch_total_params_grad}')


def nume2string(num):
    """ Convert number to scientific latex mode """
    mantissa, exp = f"{num:.2e}".split("e")
    return mantissa+ " \\times 10^{"+str(int(exp))+"}"

def plot_train_loss(train_loss_avg):
    """
    Plots the training loss over the epochs.
    """
    fig = plt.figure()
    plt.plot(train_loss_avg, '.-')
    if len(train_loss_avg) > 1 and train_loss_avg[-1] is not None:
        output_title_latex = r'${:}$'.format(nume2string(train_loss_avg[-1]))
        plt.title('Final loss: '+output_title_latex)
    else:
        plt.title('Training Loss')
    plt.minorticks_on()
    plt.xlabel('Epoch')
    plt.ylabel('Loss: -log prob')
    plt.show() # Added to display the plot

# =============================================================================
# Training Script Example
# =============================================================================
if __name__ == '__main__':
    # Configuration (similar to your original setup or using dot_dict)
    config = dot_dict()
    config.input_size = 2  # Example: 2D data
    config.num_layers = 5
    config.hidden_features = 64
    config.num_bins = 8
    config.flow_type = 'PRQCT'  # or 'MPRQAT', 'MUMNNAT'
    config.learning_rate = 1e-4
    config.num_epochs = 5 # Keep epochs low for quick demo
    config.batch_size = 128

    # 1. Create dummy data (replace with your actual data loading)
    print("Generating dummy data...")
    # Example: 2D data from two Gaussians
    data1 = np.random.multivariate_normal([-2, -2], [[1, 0.5], [0.5, 1]], size=1000)
    data2 = np.random.multivariate_normal([2, 2], [[1, -0.5], [-0.5, 1]], size=1000)
    training_data_np = np.vstack((data1, data2)).astype(np.float32)
    print(f"Dummy data shape: {training_data_np.shape}")

    # 2. Create Lightning DataModule
    print("Creating DataModule...")
    data_module = FlowDataModule(data_array=training_data_np, batch_size=config.batch_size)
    # data_module.setup('fit') # Trainer calls this, but can be called manually for inspection

    # 3. Create Lightning Module
    print("Creating LightningModule...")
    lightning_model = FlowLightningModule(
        input_size=config.input_size,
        num_layers=config.num_layers,
        hidden_features=config.hidden_features,
        num_bins=config.num_bins,
        flow_type=config.flow_type,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size # Pass batch_size if used in init or hparams
    )

    # Print model summary using the new function or Lightning's built-in
    print_lightning_model_summary(lightning_model)
    # Alternatively, use ModelSummary callback in Trainer for more detail during training start

    # 4. Create Trainer
    print("Creating Trainer...")
    # Add RichProgressBar for a nicer training interface
    progress_bar = RichProgressBar()
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="cpu",  # Automatically selects GPU if available, else CPU
        devices="auto",
        callbacks=[progress_bar], # Add callbacks here
        log_every_n_steps=10,
        # deterministic=True # For reproducibility, can slow down training
    )

    # 5. Train the model
    print("Starting training...")
    time0 = time.time()
    trainer.fit(model=lightning_model, datamodule=data_module)
    print(f'Training finished in: {(time.time()-time0)/60.:.2f} min')

    # 6. Plot training loss (using the manually collected list from the module)
    # This is one way to replicate your original plot.
    # Alternatively, you can use TensorBoard or other loggers integrated with Lightning.
    if hasattr(lightning_model, 'train_loss_avg_epoch') and lightning_model.train_loss_avg_epoch:
        print("Plotting training loss...")
        plot_train_loss(lightning_model.train_loss_avg_epoch)
    else:
        print("No epoch losses recorded for plotting in the lightning_model.train_loss_avg_epoch list.")
        print("Ensure on_train_epoch_end is correctly accumulating losses if you need this specific plot.")
        print("Alternatively, check logs via TensorBoard (if configured).")


    # 7. Example: Sample from the trained model
    print("Sampling from the trained model...")
    num_generated_samples = 500
    generated_samples = lightning_model.sample(num_generated_samples)
    generated_samples_np = generated_samples.cpu().numpy() # Move to CPU and convert to NumPy

    print(f"Generated {num_generated_samples} samples of shape: {generated_samples_np.shape}")

    # Plotting the generated samples and original data for comparison (for 2D data)
    if config.input_size == 2:
        print("Plotting original vs. generated samples...")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(training_data_np[:, 0], training_data_np[:, 1], alpha=0.5, label='Original Data')
        ax[0].set_title('Original Data')
        ax[0].set_xlabel('Dimension 1')
        ax[0].set_ylabel('Dimension 2')
        ax[0].legend()
        ax[0].axis('equal')

        ax[1].scatter(generated_samples_np[:, 0], generated_samples_np[:, 1], alpha=0.5, color='red', label='Generated Samples')
        ax[1].set_title('Generated Samples')
        ax[1].set_xlabel('Dimension 1')
        ax[1].set_ylabel('Dimension 2')
        ax[1].legend()
        ax[1].axis('equal') # Use equal axis for better visual comparison of distributions

        plt.tight_layout()
        plt.show()