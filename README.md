# Inspectorch: Efficient rare event exploration with normalizing flows

This repository contains the code for the paper "Inspectorch: Efficient rare event exploration with normalizing flows", available at http://arxiv.org/abs/2602.20316

## Abstract
With current and future solar telescopes, the Sun is observed in unprecedented detail, making it possible to study its activity on very small scales. As a result, large volumes of data are collected that cannot be reasonably analyzed with conventional methods. Automatic methods based on machine learning can be optimized to identify general trends from observations. However, these methods tend to overlook unique and rare events due to their low frequency of occurrence. We aim to develop a method to efficiently identify rare events in multidimensional solar observations where existing classic approaches fail. We use normalizing flows, a flexible density estimator, to model the multidimensional distribution of solar spectra. After training, the normalizing flow assigns a probability to each sample, allowing us to identify rare events. We illustrate the potential of this method by applying it to observations from the Hinode Spectro-Polarimeter (Hinode/SP), the CRisp Imaging SpectroPolarimeter (CRISP) at the Swedish 1-m Solar Telescope (SST), the Interface Region Imaging Spectrograph (IRIS) and the Microlensed Hyperspectral Imager (MiHI). On those datasets, we identify extreme spectra in the data that occupy a tiny percentage of the total number of samples and are overlooked by traditional clustering methods. The probabilistic nature of the method makes it ideal for implementing denoising algorithms while preserving the rare events and investigating the correlation between different spectral lines. Density estimators like normalizing flows can be a powerful tool to identify rare events in extensive datasets. The method requires minimal parameter tuning and can be applied to large data sets with minimal effort. With these methods we can optimize our resources to apply them to the most interesting events of our observations.



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

**Start with the Normalizing Flow example**: [`example/01_normalizing_flow.ipynb`](example/01_normalizing_flow.ipynb) (or the script version, [`example/01_normalizing_flow.py`](example/01_normalizing_flow.py)). It walks through the full workflow on a sample solar dataset and is the recommended entry point.

Inspectorch includes two flow-based methods:
1. **Normalizing Flows** (for accuracy): Recommended for high precision density estimation, and the best place to start. See [`example/01_normalizing_flow.py`](example/01_normalizing_flow.py).
2. **Flow Matching** (for scalability): Recommended for larger datasets or higher dimensions where scalability is a concern. Explore it after the normalizing flow example. See [`example/02_flow_matching.py`](example/02_flow_matching.py).

Explore these scripts to see how to use Inspectorch with your own data. The scripts include comments to guide you through the process. For multidimensional solar data (e.g. 3D cubes with dimensions like `y, wav, x`), we recommend using the `GeneralizedPatchedDataset` to handle patch extraction and normalization automatically.

## Available Backends

Inspectorch provides multiple density estimation backends, each with different trade-offs:

| Backend | Name | Library | Type | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `normalizing_flow` | Masked Autoregressive Flow (MAF) | `nflows` | Normalizing Flow | **Fastest inference**, density estimation |
| `flow_matching_ffm` | Flow Matching (Facebook) | Facebook's `flow_matching` | SBI-style (torchdiffeq ODE) | General purpose, balanced approach |
| `flow_matching_sbi` | Flow Matching | Pure PyTorch | SBI-style (torchdiffeq ODE) | **Fastest training**, research |
| `flow_matching_cfm` | Conditional Flow Matching | `torchcfm` | TorchCFM API + SBI fallback | Advanced flow matching methods, fast training |


**Standardized Architectures:**
All Flow Matching backends (`flow_matching_ffm`, `flow_matching_sbi`, `flow_matching_cfm`) support the following velocity network architectures:
*   `AdaMLP` (Default): Adaptive MLP with time embedding injection. Two block
    styles are available via `block_style=`: `adaln_zero` (default, AdaLN-Zero
    as in DiT) and `legacy` (the original block, kept for reproducing older
    checkpoints -- see "Choosing a block style" below). Defaults differ per backend (`flow_matching_ffm` and `flow_matching_sbi`: `hidden_features=128`, `num_layers=5`; `flow_matching_cfm`: `hidden_features=64`, `num_layers=2`; all: `time_embedding_dim=32`). Pass them explicitly to compare backends fairly.
*   `ResNet`: Residual Neural Network.
*   `ResNetFlow`: ResNet adapted for flows (`nflows` style).
*   `FourierMLP`: MLP with Gaussian Fourier feature mappings.
*   `MLPLegacy`: Simple concatenated time embedding MLP.


**How to use:**
```python
import torch
from inspectorch import DensityEstimator, GeneralizedPatchedDataset

# 1. Prepare your data (e.g., a 3D spectral cube [y, wav, x])
# dataset = GeneralizedPatchedDataset(data, dim_names="y wav x", feature_dims=["wav"])
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

# 2. Initialize model
model = DensityEstimator(type="normalizing_flow")

# 3. Create flow with specific architecture
model.create_flow(input_size=data_dim, num_layers=5, hidden_features=32, num_bins=8)

# 4. Train
model.train_flow(train_loader, num_epochs=100, learning_rate=1e-3, device="auto")
# `device="auto"`: Auto-detect best available (CUDA > MPS > CPU)

# 5. Evaluate log likelihood
log_prob = model.log_prob(dataset)
```

## Choosing a block style

`AdaMLP` blocks default to `block_style="adaln_zero"`: the features are layer
normalised before the time/context modulation, and each block is gated from
zero so it starts as an exact identity. The earlier block (`block_style="legacy"`)
skipped the normalisation and offset the gate by 1, so it did not.

Measured on Hinode spectra, held-out log-likelihood over 3 seeds (higher is
better):

| `num_layers` | `legacy` | `adaln_zero` |
| ---: | ---: | ---: |
| 2 | 91.9 ± 1.2 | 92.2 ± 0.4 |
| 5 | 93.1 ± 0.6 | 92.7 ± 0.7 |
| 20 | 83.2 ± 6.4 | 92.8 ± 0.2 |

At shallow depth the two are within seed-to-seed noise. At depth the legacy
block degrades sharply and becomes unstable across seeds, which is why
`adaln_zero` is the default.

**Loading an older checkpoint:** the two styles have identical parameter sets,
so a `state_dict` trained under one loads into the other *without any error*
and silently produces different results. Pass `block_style="legacy"` when
loading a model trained before this became the default.

## A note on ODE solver accuracy

The flow-matching backends obtain `log_prob` by integrating an ODE, so the
solver and its step count decide how accurate the log-densities are. Fixed-step
`euler` is the default and is usually the right trade-off, but the **number of
steps matters**: measured on a perturbed (untrained) model, where a correct
density must integrate to exactly 1, `euler` gives

| steps | integral |
| ---: | ---: |
| 10 | 1.06 |
| 50 | 1.01 |
| 100 | 1.005 |
| 400 | 1.001 |

so roughly 100 steps or more is sensible; 10 is too coarse. Pass `steps=` (or
`step_size=1/steps`) to set it.

Adaptive `dopri5` is more accurate still, but on the `flow_matching_cfm`
backend it can be a few hundred times slower for the same input, so it is not
the default. If absolute log-density values matter to you (rather than the
ranking, which is far less sensitive), check convergence by re-running with
more steps and confirming the values stop moving.

## Citation

If you use Inspectorch in your research, please cite the corresponding paper:

```bibtex
@article{diazbas2026inspectorch,
  title   = {Inspectorch: Efficient rare event exploration with normalizing flows},
  author  = {Diaz Baso, Carlos Jose},
  journal = {arXiv preprint arXiv:2602.20316},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.20316}
}
```

## For development purposes

```bash
git clone git@github.com:cdiazbas/inspectorch.git
cd inspectorch
pip install -e .
```

Run `pre-commit install` to install the pre-commit hooks. Run `pre-commit run --all-files` to check all files.
