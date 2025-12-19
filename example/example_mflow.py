#!/usr/bin/env python
# coding: utf-8

# Anomaly detection using Flow Matching

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from inspectorch import mflowutils
from einops import rearrange
from inspectorch.plot_params import set_params
from inspectorch import genutils

import warnings

warnings.filterwarnings("ignore")


# Set matplotlib plotting parameters for consistent style
set_params()


# Print device info for debugging and reproducibility
genutils.device_info()

# --- Data loading and preprocessing ---
# Read the compressed Hinode data from file
with np.load("hinode_data.npz") as npzfile:
    # The notebook example normalizes by a fixed factor (186.0)
    data = npzfile["data"] / 186.0
    wav = npzfile["wav"]
    pixel_size = npzfile["pixel_size"] * 3.0

ny, nwav, nx = data.shape
print(f"Data shape: {data.shape}")

# Data does not need any further normalization here (handled in the notebook example)


# --- Dataset preparation ---

# Wrap the data in a patched dataset for spectral analysis
dataset = mflowutils.GeneralizedPatchedDataset(
    data,
    dim_names="y wav x",  # Define the dimensions of the dataset
    feature_dims=["wav"],  # Specify the feature dimensions for spectral analysis
)

print(f"Data (patched) shape: {dataset.shape}")

# Compute average spectrum for reference
average = np.mean(data, axis=(0, 2))

# --- Visualization of raw data ---

plt.figure(figsize=(8, 8))
extent = np.array([0, data.shape[2], 0, data.shape[0]]) * pixel_size
plt.imshow(
    data[:, 0, :], cmap="gray", vmin=None, vmax=1.3, extent=extent, origin="lower"
)
plt.minorticks_on()
plt.xlabel("X [arcsec]")
plt.ylabel("Y [arcsec]")
cb = plt.colorbar(pad=0.02, shrink=1.0, aspect=40)
cb.set_label("Intensity [a.u.]")
plt.locator_params(axis="x", nbins=3)
plt.locator_params(axis="y", nbins=5)
plt.savefig("models/raw_data_mflow.png", dpi=300, bbox_inches="tight")


# --- Data preparation for training ---

# Create a dot_dict for training arguments
args = mflowutils.dot_dict()
args.batch_size = 10000

# Create a DataLoader for efficient batch training
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True
)


# Build the Flow Matching model for density estimation
model = mflowutils.FlowMatching_Density_estimator()
model.create_flow(
    input_size=dataset.flow_dim,
    num_layers=5,
    hidden_features=32,
    scheduler_n=3.0,  # Polynomial convex scheduler (flow matching specific)
    architecture="MLP",  # Options: "MLP" or "ResNet"
    time_embedding_dim=32,
)

# Print model summary for inspection
model.print_summary()

# Visualize data distribution before training
model.check_variables(
    train_loader,
    plot_variables=[0, 1],  # Which variables to plot from your feature set
    figsize=(8, 4),
    rel_size=0.1,  # It only checks 10% of the data for speed
)
plt.savefig("models/pre_training_variables_mflow.png", dpi=300, bbox_inches="tight")


# ## Training the density estimator

# Set training hyperparameters
args.learning_rate = 1e-3
args.num_epochs = 15
args.device = "cuda:0"  # Supports multi-GPU: "cuda:0,1,2"
args.output_model = "models/mflow_model_hinode_mini.pth"
args.save_model = True
args.load_existing = True
args.extra_noise = 0.0  # Additional noise for regularization (0.0 = no extra noise)

# Save training arguments for reproducibility
genutils.save_json(args, "models/mflow_args_hinode_mini.json")

# Train the Flow Matching model using the FlowMatching_Density_estimator API
model.train_flow(
    train_loader=train_loader,
    learning_rate=args.learning_rate,
    num_epochs=args.num_epochs,
    device=args.device,
    output_model=args.output_model,
    save_model=args.save_model,
    load_existing=args.load_existing,
    extra_noise=args.extra_noise,
)

# Plot training loss to monitor convergence
model.plot_train_loss()
plt.savefig("models/training_loss_mflow.png", dpi=300, bbox_inches="tight")

# Visualize data distribution after training
model.check_variables(
    train_loader,
    plot_variables=[0, 1],  # Which variables to plot from your feature set
    figsize=(8, 4),
    rel_size=0.1,  # It only checks 10% of the data for speed
)
plt.savefig("models/post_training_variables_mflow.png", dpi=300, bbox_inches="tight")

# Compute log-probabilities for all data points
# Note: Flow matching uses ODE integration, so we can choose solver parameters
log_prob = model.log_prob(
    inputs=dataset,
    dataset_normalization=True,
    batch_size=1000,
    device=args.device,
    solver_method="dopri5",  # Options: 'dopri5', 'rk4', 'euler', 'midpoint'
    atol=1e-5,
    rtol=1e-5,
    exact_divergence=False,  # Use Hutchinson estimator for speed (True = exact but slower)
)


# The model evaluated all the dataset, so we need to reshape the log_prob
# to match the original dimensions of the data

# Reshape log_prob to match original spatial dimensions
log_prob_reshaped = rearrange(log_prob, "(y x) -> y x", y=ny, x=nx)


# Plot spatial distribution of log-probabilities
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
spectral_to_white = genutils.create_spectral_to_white_cmap()

im = ax.imshow(
    log_prob_reshaped,
    cmap=spectral_to_white,
    vmin=0.1 * np.min(log_prob_reshaped),
    vmax=0.9 * np.max(log_prob_reshaped),
    origin="lower",
    interpolation="bilinear",
    extent=extent,
)
cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=1.0, aspect=40)
cb.set_label(r"$\log\,p_\phi(\mathbf{x})$ (Flow Matching)")
plt.minorticks_on()
plt.xlabel("X [arcsec]")
plt.ylabel("Y [arcsec]")
plt.locator_params(axis="x", nbins=3)
plt.locator_params(axis="y", nbins=5)
plt.savefig("models/logprob_map_mflow.png", dpi=300, bbox_inches="tight")


# --- Visualization of least common spectra ---

nplots = 5

# Find indices of least probable spectra
idxs = np.argsort(log_prob)[:nplots]
print(f"Least common spectra indices: {idxs}")

fig, ax = plt.subplots(1, 5, figsize=(20, 16 / 4 * 1), sharex=True, sharey=True)
ax = ax.flatten()
for i in range(nplots):
    # Plot normalized spectrum for each least common sample
    ax[i % 5].plot(wav / 10, dataset[idxs[i]] / torch.max(dataset[idxs[i]]), color="C3")
    ax[i % 5].plot(
        wav / 10, average / np.max(average), ls="-", color="black", alpha=0.2
    )

    ax[i % 5].set_title(
        r"$\log\,p_\phi(\mathbf{{x}})$: ${0:.1f}$".format(log_prob[idxs[i]]),
        fontsize=20,
        fontweight="bold",
        loc="center",
        pad=25,
    )
    # Annotate spatial location of each spectrum
    ax[i % 5].text(
        0.5,
        +0.1,
        r"$\mathbf{{[x,y]}}$ = ({0}, {1})".format(idxs[i] // nx, idxs[i] % nx),
        fontsize=16,
        ha="center",
        va="top",
        transform=ax[i % 5].transAxes,
    )

    # Only set ylabel for the first column
    if i % 5 == 0:
        ax[i % 5].set_ylabel("Intensity [a.u.]")
    ax[i % 5].set_xlabel("Wavelength [nm]")

    # Add velocity axis at the top for reference
    cc = 299792.458  # speed of light in km/s
    line = (
        wav[len(wav) // 2] / 10
    )  # reference wavelength in nm (choose center or a specific value)

    # Draw a vertical line at 10 km/s on the secondary axis
    v_line = 10  # km/s
    lambda_v = line + (v_line * line / cc)
    ax[i % 5].axvline(lambda_v, color="k", linestyle="--", linewidth=1.5)
    # Add velocity label at the top of the plot
    if i == 0:
        ax[i % 5].text(
            lambda_v + 0.02,
            0.95,
            f"{v_line} km/s",
            fontsize=12,
            ha="center",
            va="top",
            color="k",
            transform=ax[i % 5].get_xaxis_transform(),
        )

for axes in ax.flat:
    axes.ticklabel_format(style="plain", axis="x", useOffset=False)

# Set y-axis limits for all subplots for consistency
for axes in ax.flat:
    axes.set_ylim(0.2, 1.1)


# Format x-axis tick labels to show a fixed number of decimals
num_decimals = 2
for axes in ax.flat:
    ticks = axes.get_xticks()
    axes.set_xticks(ticks)  # Ensure ticks are fixed
    axes.set_xticklabels([f"{tick:.{num_decimals}f}" for tick in ticks])
plt.savefig("models/least_common_spectra_mflow.png", dpi=300, bbox_inches="tight")


# --- Optional: Generate samples from the learned distribution ---
print("\nGenerating samples from the learned Flow Matching model...")
num_samples = 100
samples = model.sample(
    num_samples=num_samples,
    device=args.device,
    solver_method="dopri5",
    atol=1e-5,
    rtol=1e-5,
)

# Plot some generated samples
fig, ax = plt.subplots(1, 5, figsize=(20, 16 / 4 * 1), sharex=True, sharey=True)
ax = ax.flatten()
for i in range(min(5, num_samples)):
    sample = samples[i].numpy()
    ax[i].plot(wav / 10, sample / np.max(sample), color="C0")
    ax[i].plot(wav / 10, average / np.max(average), ls="-", color="black", alpha=0.2)
    ax[i].set_title(f"Generated Sample {i + 1}", fontsize=16)

    if i == 0:
        ax[i].set_ylabel("Intensity [a.u.]")
    ax[i].set_xlabel("Wavelength [nm]")
    ax[i].set_ylim(0.2, 1.1)

plt.suptitle("Samples Generated by Flow Matching Model", fontsize=20, y=1.02)
plt.savefig("models/generated_samples_mflow.png", dpi=300, bbox_inches="tight")

print("\nFlow Matching analysis complete!")
print(f"Models saved to: {args.output_model}")
print("All plots saved to the models/ directory with '_mflow' suffix")
