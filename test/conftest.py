"""
Shared fixtures and helpers for the inspectorch test suite.
"""

import numpy as np
import pytest
import torch

ALL_BACKENDS = [
    "normalizing_flow",
    "flow_matching_ffm",
    "flow_matching_sbi",
    "flow_matching_cfm",
]

FLOW_MATCHING_BACKENDS = [b for b in ALL_BACKENDS if b != "normalizing_flow"]

ARCHITECTURES = ["AdaMLP", "ResNet", "ResNetFlow", "FourierMLP", "MLPLegacy"]

# Which third-party library each backend needs, so a missing one skips
# rather than fails.
_BACKEND_REQUIREMENTS = {
    "normalizing_flow": ["nflows"],
    "flow_matching_ffm": ["flow_matching", "torchdiffeq"],
    "flow_matching_sbi": ["torchdiffeq"],
    "flow_matching_cfm": ["torchcfm", "torchdiffeq"],
}


def require_backend(name):
    """Skip the calling test when the backend's dependencies are absent."""
    for module in _BACKEND_REQUIREMENTS[name]:
        pytest.importorskip(module)


class ToyDataset(torch.utils.data.Dataset):
    """
    Stands in for GeneralizedPatchedDataset with a deliberately
    non-unit, non-uniform scale so that a missing change-of-variables
    correction shows up.
    """

    def __init__(self, n_samples=256, scale=(5.0, 0.5, 2.0, 1.0), seed=0):
        generator = torch.Generator().manual_seed(seed)
        scale = torch.tensor(scale)
        self.patches = torch.randn(n_samples, len(scale), generator=generator) * scale
        self.y_mean = self.patches.mean(dim=0)
        self.y_std = self.patches.std(dim=0)

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, index):
        return self.patches[index]

    def normalized_patches(self):
        return (self.patches - self.y_mean) / self.y_std


class GridDataset(ToyDataset):
    """A dataset whose patches are a fixed grid, for quadrature."""

    def __init__(self, patches, y_mean, y_std):
        self.patches = patches
        self.y_mean = y_mean
        self.y_std = y_std


@pytest.fixture
def toy_dataset():
    return ToyDataset()


@pytest.fixture
def toy_loader(toy_dataset):
    return torch.utils.data.DataLoader(toy_dataset, batch_size=64)


def build_estimator(kind, input_size=4, seed=0, **kwargs):
    """
    Create a DensityEstimator with per-backend create_flow arguments.

    Seeds before construction: model initialisation consumes the global
    RNG, so without this the starting weights differ between runs and any
    numerical assertion on them is flaky.
    """
    require_backend(kind)
    torch.manual_seed(seed)
    np.random.seed(seed)
    from inspectorch import DensityEstimator

    estimator = DensityEstimator(type=kind)
    if kind == "normalizing_flow":
        estimator.create_flow(
            input_size=input_size, num_layers=2, hidden_features=16, num_bins=4
        )
    else:
        estimator.create_flow(
            input_size=input_size, num_layers=2, hidden_features=16, **kwargs
        )
    return estimator


def randomize(estimator, seed=3, magnitude=0.3):
    """
    Perturb the weights so the model is not a trivial identity.

    Flow-matching velocity nets zero-initialise their output layer, which
    makes the flow the identity; normalizing flows start near-Gaussian.
    A density is normalised regardless of training, so randomising lets
    the normalisation tests run without any training at all.

    `magnitude` is kept modest on purpose: a large perturbation produces
    a stiff velocity field that no practical fixed-step solver resolves,
    which would test the solver rather than the normalisation.
    """
    torch.manual_seed(seed)
    backend = estimator.backend
    with torch.no_grad():
        velocity = getattr(backend, "velocity_model", None)
        if velocity is not None and hasattr(velocity, "final_layer"):
            velocity.final_layer.weight.add_(
                torch.randn_like(velocity.final_layer.weight) * magnitude
            )
            velocity.final_layer.bias.add_(
                torch.randn_like(velocity.final_layer.bias) * magnitude * 0.55
            )
        elif getattr(backend, "flow_model", None) is not None:
            for param in backend.flow_model.parameters():
                if param.dim() > 1:
                    param.add_(torch.randn_like(param) * magnitude)
    return estimator


def integrate_density(log_prob, axis, jacobian=1.0):
    """Rectangle-rule integral of exp(log_prob) over a 1-D grid."""
    return float(np.exp(log_prob).sum() * (axis[1] - axis[0]) * jacobian)
