"""
Regression tests for the density-normalisation maths.

A normalizing flow (or CNF) defines a properly normalised density
regardless of how well it is trained. These tests exploit that: they
randomise the weights, integrate ``exp(log_prob)`` over a grid, and
assert the result is 1. No training required, so nothing here depends
on model quality.

They pin down two things that were previously wrong or unverified:

1. ``log_prob`` must be a density in the DATA coordinates. Fitting on
   standardised data ``y = (x - mu) / std`` and reporting ``log p(y)``
   makes the integral come out at ``prod(std)`` instead of 1; the
   change-of-variables term ``-sum(log std)`` is what fixes it. The
   assignment computing that term was once silently deleted by
   ``autoflake --remove-unused-variable``.

2. The CNF sign convention ``log p(x) = log p(z) - delta_log_p``.

The flow-matching tests are skipped when torchdiffeq is unavailable.
"""

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal, norm

from inspectorch.utils import normalization_log_det



class _Shim:
    """Minimal stand-in for GeneralizedPatchedDataset."""

    def __init__(self, patches, y_mean, y_std):
        self.patches = patches
        self.y_mean = y_mean
        self.y_std = y_std

    def normalized_patches(self):
        return (self.patches - self.y_mean) / self.y_std


# ---------------------------------------------------------------- analytic
def test_change_of_variables_matches_analytic_gaussian():
    """log p_x(x) = log p_y(y) - sum(log std), to machine precision."""
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -2.0, 0.5])
    sigma = np.array([5.0, 0.5, 2.0])

    x = rng.standard_normal((2000, 3)) * sigma + mu
    y = (x - mu) / sigma

    claimed = norm.logpdf(y).sum(axis=1) - np.sum(np.log(sigma))
    exact = multivariate_normal.logpdf(x, mu, np.diag(sigma**2))

    assert np.abs(claimed - exact).max() < 1e-9


def test_normalization_log_det_scalar_and_vector_agree():
    """A scalar std broadcast over D features equals a length-D vector."""
    scalar = normalization_log_det(torch.tensor([3.0]), 3)
    vector = normalization_log_det(torch.full((3,), 3.0), 3)

    assert scalar == pytest.approx(vector)
    assert scalar == pytest.approx(-3.0 * np.log(3.0))
    assert normalization_log_det(None, 5) == 0.0


# ------------------------------------------------------- normalizing flow
def test_normalizing_flow_log_prob_integrates_to_one_in_data_space():
    """
    exp(log_prob) must integrate to 1 over the DATA coordinates.

    Without the -sum(log std) correction this integrates to prod(std),
    which for these values is 2.5.
    """
    from inspectorch.normalizing_flow import NormalizingFlowBackend

    torch.manual_seed(0)
    backend = NormalizingFlowBackend()
    backend.create_flow(input_size=2, num_layers=3, hidden_features=16, num_bins=6)
    with torch.no_grad():  # randomise so the density is not trivially Gaussian
        for param in backend.flow_model.parameters():
            if param.dim() > 1:
                param.add_(torch.randn_like(param) * 0.25)

    mu = np.array([1.0, -2.0])
    sigma = np.array([5.0, 0.5])

    axis = np.linspace(-7, 7, 300)
    mesh = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    grid_x = mesh * sigma + mu

    shim = _Shim(
        torch.tensor(grid_x, dtype=torch.float32),
        torch.tensor(mu, dtype=torch.float32),
        torch.tensor(sigma, dtype=torch.float32),
    )
    log_prob = backend.log_prob(shim, dataset_normalization=True, device="cpu")

    cell = (axis[1] - axis[0]) ** 2 * np.prod(sigma)
    assert np.exp(log_prob).sum() * cell == pytest.approx(1.0, abs=5e-3)


# ----------------------------------------------------------- flow matching
def _randomised_sbi_backend(input_size=1):
    pytest.importorskip("torchdiffeq")
    from inspectorch.flow_matching_sbi import FlowMatchingSBIBackend

    torch.manual_seed(3)
    backend = FlowMatchingSBIBackend()
    backend.create_flow(
        input_size=input_size, num_layers=2, hidden_features=32, architecture="AdaMLP"
    )
    with torch.no_grad():  # final_layer is zero-init => identity flow
        final = backend.velocity_model.final_layer
        final.weight.add_(torch.randn_like(final.weight) * 0.9)
        final.bias.add_(torch.randn_like(final.bias) * 0.5)
    return backend


# The randomised velocity field has heavy tails: a narrower grid truncates
# real mass and the integral undershoots. +/-40 is converged to ~1e-5.
_HALF_WIDTH, _POINTS = 40, 6000


@pytest.mark.parametrize("solver", ["dopri5", "rk4"])
def test_cnf_log_prob_integrates_to_one(solver):
    """Pins the sign convention log p(x) = log p(z) - delta_log_p."""
    backend = _randomised_sbi_backend()
    backend.y_mean, backend.y_std = torch.zeros(1), torch.ones(1)

    axis = np.linspace(-_HALF_WIDTH, _HALF_WIDTH, _POINTS)
    grid = torch.tensor(axis.reshape(-1, 1), dtype=torch.float32)
    kwargs = {"steps": 200} if solver == "rk4" else {}

    log_prob = backend.log_prob(
        grid,
        dataset_normalization=False,
        batch_size=8000,
        device="cpu",
        solver_method=solver,
        exact_divergence=True,
        **kwargs,
    )

    integral = np.exp(log_prob).sum() * (axis[1] - axis[0])
    assert integral == pytest.approx(1.0, abs=0.01)


def test_cnf_log_prob_is_data_space_density_on_the_dataset_path():
    """
    The dataset path must apply the change-of-variables correction too.

    This is the regression that matters: the correction used to be
    computed only on the bare-tensor path, and even there the result was
    discarded, so both paths returned a density off by prod(std).
    """
    sigma = 3.0
    backend = _randomised_sbi_backend()
    backend.y_mean, backend.y_std = torch.zeros(1), torch.full((1,), sigma)

    axis = np.linspace(-_HALF_WIDTH, _HALF_WIDTH, _POINTS)
    data = torch.tensor((axis * sigma).reshape(-1, 1), dtype=torch.float32)
    shim = _Shim(data, torch.zeros(1), torch.full((1,), sigma))

    log_prob = backend.log_prob(
        shim,
        dataset_normalization=True,
        batch_size=8000,
        device="cpu",
        solver_method="dopri5",
        exact_divergence=True,
    )

    integral = np.exp(log_prob).sum() * (axis[1] - axis[0]) * sigma
    assert integral == pytest.approx(1.0, abs=0.01)
