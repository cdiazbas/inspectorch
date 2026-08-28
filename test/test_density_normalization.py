"""
Regression tests for the density-normalisation maths, across every backend.

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
   ``autoflake --remove-unused-variable``, and the correction was
   missing from the CFM backend altogether.

2. The CNF sign convention ``log p(x) = log p(z) - delta_log_p``.

Backends whose third-party library is absent are skipped, not failed.
"""

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal, norm

from conftest import (
    ALL_BACKENDS,
    FLOW_MATCHING_BACKENDS,
    GridDataset,
    build_estimator,
    randomize,
)
from inspectorch.utils import normalization_log_det

# Integration settings, chosen by measuring convergence rather than guessed:
# +/-14 covers the tails of the perturbed field, and euler at 200 steps with
# an exact divergence lands within ~1% while staying well under a second.
_HALF, _POINTS, _STEPS = 14, 1200, 200
_TOL = 0.02
_SCALE = 3.0


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


def test_wrong_sign_does_not_match():
    """Guards the direction of the correction, not just its magnitude."""
    rng = np.random.default_rng(0)
    sigma = np.array([5.0, 0.5, 2.0])
    x = rng.standard_normal((500, 3)) * sigma
    y = x / sigma

    flipped = norm.logpdf(y).sum(axis=1) + np.sum(np.log(sigma))
    exact = multivariate_normal.logpdf(x, np.zeros(3), np.diag(sigma**2))

    assert np.abs(flipped - exact).max() > 1.0


def test_normalization_log_det_scalar_and_vector_agree():
    """A scalar std broadcast over D features equals a length-D vector."""
    scalar = normalization_log_det(torch.tensor([3.0]), 3)
    vector = normalization_log_det(torch.full((3,), 3.0), 3)

    assert scalar == pytest.approx(vector)
    assert scalar == pytest.approx(-3.0 * np.log(3.0))
    assert normalization_log_det(None, 5) == 0.0


# ------------------------------------------------------- integration helper
def _integrate(kind, scale=_SCALE):
    """
    exp(log_prob) integrated over the data coordinates for one backend.

    A 1-D grid is used for the ODE-based backends, where every grid point
    costs an ODE solve; the normalizing flow uses a 2-D grid because its
    1-D path builds a much slower UMNN transform.
    """
    if kind == "normalizing_flow":
        axis = np.linspace(-8, 8, 320)
        mesh = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        sigma = np.array([scale, 1.0 / scale])
        estimator = randomize(build_estimator(kind, input_size=2))
        grid = GridDataset(
            torch.tensor(mesh * sigma, dtype=torch.float32),
            torch.zeros(2),
            torch.tensor(sigma, dtype=torch.float32),
        )
        log_prob = estimator.log_prob(grid, device="cpu")
        cell = (axis[1] - axis[0]) ** 2 * float(np.prod(sigma))
        return float(np.exp(log_prob).sum() * cell)

    axis = np.linspace(-_HALF, _HALF, _POINTS)
    estimator = randomize(build_estimator(kind, input_size=1))
    estimator.backend.y_mean = torch.zeros(1)
    estimator.backend.y_std = torch.full((1,), scale)
    grid = GridDataset(
        torch.tensor((axis * scale).reshape(-1, 1), dtype=torch.float32),
        torch.zeros(1),
        torch.full((1,), scale),
    )
    log_prob = estimator.log_prob(
        grid,
        device="cpu",
        batch_size=4000,
        solver_method="euler",
        step_size=1.0 / _STEPS,
        steps=_STEPS,
        exact_divergence=True,  # deterministic; Hutchinson would add noise
    )
    return float(np.exp(log_prob).sum() * (axis[1] - axis[0]) * scale)


# ------------------------------------------------ every backend, one rule
@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_every_backend_returns_a_data_space_density(kind):
    """
    Every backend must agree on what log_prob means.

    Whatever the backend, exp(log_prob) has to integrate to 1 over the
    data coordinates. A backend that omits the -sum(log std) correction
    integrates to prod(std) instead -- here 3.0, which is how the missing
    correction in the CFM backend was found.
    """
    integral = _integrate(kind)
    assert integral == pytest.approx(1.0, abs=_TOL), (
        f"{kind}: exp(log_prob) integrates to {integral:.4f}, not 1. "
        f"A value near {_SCALE} means the change-of-variables correction "
        f"for the standardisation is missing."
    )


@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_log_prob_is_deterministic_for_a_fixed_model(kind):
    """Two identical calls must agree, so the tests above are meaningful."""
    first, second = _integrate(kind), _integrate(kind)
    assert first == pytest.approx(second, rel=1e-6)


@pytest.mark.parametrize("kind", FLOW_MATCHING_BACKENDS)
def test_cnf_sign_convention(kind):
    """
    Pins log p(x) = log p(z) - delta_log_p.

    With the sign flipped the result is not a density at all, so the
    integral moves far away from 1 rather than slightly.
    """
    integral = _integrate(kind, scale=1.0)
    assert integral == pytest.approx(1.0, abs=_TOL)
