# Inspectorch: Efficient rare event exploration with normalizing flows

This repository contains the code for the paper "Inspectorch: Efficient rare event exploration with normalizing flows" (arXiv:X).

## Abstract
With current and future solar telescopes, the Sun is observed in unprecedented detail, making it possible to study its activity on very small scales. As a result, large volumes of data are collected that cannot be reasonably analyzed with conventional methods. Automatic methods based on machine learning can be optimized to identify general trends from observations. However, these methods tend to overlook unique and rare events due to their low frequency of occurrence. We aim to develop a method to efficiently identify rare events in multidimensional solar observations where existing classic approaches fail. We use normalizing flows, a flexible density estimator, to model the multidimensional distribution of solar spectra. After training, the normalizing flow assigns a probability to each sample, allowing us to identify rare events. We illustrate the potential of this method by applying it to observations from the Hinode Spectro-Polarimeter (Hinode/SP), the CRisp Imaging SpectroPolarimeter (CRISP) at Swedish 1-m Solar Telescope (SST), the Interface Region Imaging Spectrograph (IRIS) and the Microlensed Hyperspectral Imager (MiHI). On those datasets, we identify extreme spectra in the data that occupy a tiny percentage of the total number of samples and are overlooked by traditional clustering methods. The probabilistic nature of the method makes it ideal for implementing denoising algorithms while preserving the rare events and investigating the correlation between different spectral lines.Density estimators like normalizing flows can be a powerful tool to identify rare events in extensive datasets. The method requires minimal parameter tuning and can be applied to large data sets with minimal effort. With these methods we can optimize our resources to apply them to the most interesting events of our observations.


## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/inspectorch.git
cd inspectorch
pip install .
```

You may also want to install in editable mode for development:

```bash
pip install -e .
```

## Usage

- Prepare your data in the expected format (see `example/example.py`).
- Run the example script to train and evaluate the normalizing flow:

```bash
python inspectorch/example/example.py
```

- Modify the script or use the provided API to analyze your own datasets.

## Citation

If you use Inspectorch in your research, please cite the corresponding paper (arXiv:X).


## Pre-commit hooks (only for development)
Run `pre-commit install` to install the pre-commit hooks. Run `pre-commit run --all-files` to check all files.
