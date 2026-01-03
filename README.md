# Inspectorch: Efficient rare event exploration with normalizing flows

This repository contains the code for the paper "Inspectorch: Efficient rare event exploration with normalizing flows" (arXiv:X).

## Abstract
With current and future solar telescopes, the Sun is observed in unprecedented detail, making it possible to study its activity on very small scales. As a result, large volumes of data are collected that cannot be reasonably analyzed with conventional methods. Automatic methods based on machine learning can be optimized to identify general trends from observations. However, these methods tend to overlook unique and rare events due to their low frequency of occurrence. We aim to develop a method to efficiently identify rare events in multidimensional solar observations where existing classic approaches fail. We use normalizing flows, a flexible density estimator, to model the multidimensional distribution of solar spectra. After training, the normalizing flow assigns a probability to each sample, allowing us to identify rare events. We illustrate the potential of this method by applying it to observations from the Hinode Spectro-Polarimeter (Hinode/SP), the CRisp Imaging SpectroPolarimeter (CRISP) at Swedish 1-m Solar Telescope (SST), the Interface Region Imaging Spectrograph (IRIS) and the Microlensed Hyperspectral Imager (MiHI). On those datasets, we identify extreme spectra in the data that occupy a tiny percentage of the total number of samples and are overlooked by traditional clustering methods. The probabilistic nature of the method makes it ideal for implementing denoising algorithms while preserving the rare events and investigating the correlation between different spectral lines.Density estimators like normalizing flows can be a powerful tool to identify rare events in extensive datasets. The method requires minimal parameter tuning and can be applied to large data sets with minimal effort. With these methods we can optimize our resources to apply them to the most interesting events of our observations.



## Installation

Create a conda environment (recommended) if you don't have one already. Important: use Python 3.9 - 3.13, since you will need to install PyTorch (see below). You can force a specific Python version with `python=3.x` in the command below.

```bash
conda create -n inspectorch python
conda activate inspectorch
```

Clone the repository and install the required dependencies (automatically with pip):

```bash
git clone git@github.com:cdiazbas/inspectorch.git
cd inspectorch
pip install .
```



## Usage

Inspectorch includes two flow-based methods:
1. **Normalizing Flows** (for accuracy): Recommended for high precision density estimation. See `inspectorch/example/example.py`.
2. **Flow Matching** (for scalability): Recommended for larger datasets or higher dimensions where scalability is a concern. See `inspectorch/example/example_mflow.py`.

Explore these scripts to see how to use Inspectorch with your own data. The scripts include comments to guide you through the process.

## Available Backends

Inspectorch provides multiple density estimation backends, each with different trade-offs:

| Backend | Name | Library | Type | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `normalizing_flow` | Masked Autoregressive Flow (MAF) | `nflows` | Normalizing Flow | **Fastest inference**, density estimation |
| `flow_matching_ffm` | Flow Matching (Facebook) | Facebook's `flow_matching` | SBI-style (torchdiffeq ODE) | General purpose, balanced approach |
| `flow_matching_sbi` | Flow Matching | Pure PyTorch | SBI-style (torchdiffeq ODE) | **Fastest training**, research |
| `flow_matching_cfm` | Conditional Flow Matching | `torchcfm` | TorchCFM API + SBI fallback | Advanced flow matching methods, fast training |


**Standardized Architectures:**
All Flow Matching backends (`flow_matching`, `flow_matching_sbi`, `flow_matching_cfm`) support the following velocity network architectures:
*   `AdaMLP` (Default): Adaptive MLP with time embedding injection (Standardized: `hidden_features=64`, `num_layers=2`, `time_embedding_dim=32`).
*   `ResNet`: Residual Neural Network.
*   `ResNetFlow`: ResNet adapted for flows (`nflows` style).
*   `FourierMLP`: MLP with Gaussian Fourier feature mappings.
*   `MLPLegacy`: Simple concatenated time embedding MLP.


**How to use:**
```python
from inspectorch import DensityEstimator

# Initialize model
model = DensityEstimator(type="flow_matching_ffm")

# Create flow with specific architecture
model.create_flow(input_size=2, architecture="AdaMLP")

# Train
model.train_flow(train_loader, num_epochs=100)

# Evaluate log likelihood
log_prob = model.log_prob(test_data)
```

## Citation

If you use Inspectorch in your research, please cite the corresponding paper (arXiv:X).

## For development purposes

```bash
git clone git@github.com:cdiazbas/inspectorch.git
cd inspectorch
pip install -e .
```

Run `pre-commit install` to install the pre-commit hooks. Run `pre-commit run --all-files` to check all files.
