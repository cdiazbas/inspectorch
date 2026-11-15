"""
Few-step flow maps (inspectorch.fast_flows).

The core checks need no training: for Gaussian data N(m, s^2 I) and noise
N(0, I) with x_t = (1 - t) x0 + t x1, the probability flow is linear and
known in closed form, so the training targets, the divergence and the
log-densities can be compared with the truth.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from inspectorch import fast_flows as ff

D = 3
M = torch.tensor([1.5, -0.5, 0.25], dtype=torch.float64)
S = 0.4


class GaussianFlow(nn.Module):
    """Exact flow from N(0, I) at t=0 to N(M, S^2 I) at t=1."""

    @staticmethod
    def mu(t):
        return t * M

    @staticmethod
    def sig(t):
        return torch.sqrt((1 - t) ** 2 + (t * S) ** 2)

    @staticmethod
    def dsig(t):
        return (-(1 - t) + t * S**2) / GaussianFlow.sig(t)

    def velocity(self, x, t):
        return M + self.dsig(t) / self.sig(t) * (x - self.mu(t))

    def X(self, x, a, b):
        return self.mu(b) + self.sig(b) / self.sig(a) * (x - self.mu(a))

    def forward(self, x, a, b):
        d = b - a
        safe = torch.where(d.abs() > 1e-9, d, torch.ones_like(d))
        return torch.where(
            d.abs() > 1e-9, (self.X(x, a, b) - x) / safe, self.velocity(x, a)
        )

    def G(self, x, a, b):
        """Average of div v over [a, b]; div v = D sig'/sig, independent of x."""
        d = b - a
        safe = torch.where(d.abs() > 1e-9, d, torch.ones_like(d))
        g = torch.where(
            d.abs() > 1e-9,
            D * (torch.log(self.sig(b)) - torch.log(self.sig(a))) / safe,
            D * self.dsig(a) / self.sig(a),
        )
        return g[..., 0] + 0 * x[..., 0]

    def both(self, x, a, b):
        return self(x, a, b), self.G(x, a, b)


def _marginal(t, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, D, generator=g, dtype=torch.float64)
    return GaussianFlow.mu(t) + GaussianFlow.sig(t) * z


def _exact_log_p_data(x):
    return (
        -0.5 * (((x - M) / S) ** 2).sum(-1)
        - D * math.log(S)
        - 0.5 * D * math.log(2 * math.pi)
    )


@pytest.mark.parametrize("a0,b0", [(0.2, 0.9), (0.9, 0.1), (1.0, 0.0), (0.0, 1.0)])
def test_average_velocity_identity_gives_the_true_u(a0, b0):
    """u = v + (b - a) du/da, with du/da along the trajectory, holds for the true flow."""
    flow = GaussianFlow()
    n = 64
    a = torch.full((n, 1), a0, dtype=torch.float64)
    b = torch.full((n, 1), b0, dtype=torch.float64)
    x = _marginal(a)
    v = flow.velocity(x, a)
    target = ff.average_velocity_target(flow, x, a, b, v)
    torch.testing.assert_close(target, flow(x, a, b), rtol=0, atol=1e-10)


@pytest.mark.parametrize("a0,b0", [(0.3, 0.8), (1.0, 0.0)])
def test_average_divergence_identity_gives_the_true_g(a0, b0):
    """G = div v + (b - a) dG/da holds for the true average divergence."""
    flow = GaussianFlow()
    n = 16
    a = torch.full((n, 1), a0, dtype=torch.float64)
    b = torch.full((n, 1), b0, dtype=torch.float64)
    x = _marginal(a, n)
    v = flow.velocity(x, a)
    div_v = ff.divergence(flow, x, a, exact=True)
    target = ff.average_divergence_target(flow.G, x, a, b, v, div_v)
    torch.testing.assert_close(target, flow.G(x, a, b), rtol=0, atol=1e-10)


def test_exact_divergence_matches_closed_form():
    flow = GaussianFlow()
    t = torch.full((8, 1), 0.37, dtype=torch.float64)
    x = _marginal(t, 8)
    expected = (D * flow.dsig(t) / flow.sig(t))[:, 0]
    torch.testing.assert_close(
        ff.divergence(flow, x, t, exact=True), expected, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("steps", [1, 3])
def test_map_log_prob_is_the_true_density(steps):
    """Change of variables through the exact map gives the data density."""
    x = _marginal(torch.ones(1, 1, dtype=torch.float64), 32, seed=1)
    lp = ff.map_log_prob(GaussianFlow(), x, steps=steps)
    torch.testing.assert_close(lp, _exact_log_p_data(x), rtol=0, atol=1e-9)


@pytest.mark.parametrize("steps", [1, 4])
def test_divergence_sum_log_prob_is_the_true_density(steps):
    """The f2d2 route (sum of average divergences) gives the same density."""
    flow = GaussianFlow()
    x = _marginal(torch.ones(1, 1, dtype=torch.float64), 32, seed=2)
    lp = ff.map_log_prob(flow, x, steps=steps, g_fn=flow.G)
    torch.testing.assert_close(lp, _exact_log_p_data(x), rtol=0, atol=1e-9)


def test_flow_step_there_and_back_is_the_identity():
    flow = GaussianFlow()
    x = _marginal(torch.full((1, 1), 0.2, dtype=torch.float64), 16)
    a = torch.full((16, 1), 0.2, dtype=torch.float64)
    b = torch.full((16, 1), 0.85, dtype=torch.float64)
    back = ff.flow_step(flow, ff.flow_step(flow, x, a, b), b, a)
    torch.testing.assert_close(back, x, rtol=0, atol=1e-12)


# --- every kind runs end to end on a toy problem (a few optimiser steps) ---
class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n=256, d=D):
        g = torch.Generator().manual_seed(0)
        self.patches = (
            torch.randn(n, d, generator=g) * torch.tensor([3.0, 0.5, 1.0]) + 2.0
        )
        self.y_mean, self.y_std = self.patches.mean(0), self.patches.std(0)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        return self.patches[i]

    def normalized_patches(self):
        return (self.patches - self.y_mean) / self.y_std


@pytest.mark.parametrize("kind", ff.KINDS)
def test_each_kind_trains_scores_and_samples(kind):
    if kind == "regflow":
        pytest.importorskip("ot")
    torch.manual_seed(0)
    ds = _ToyDataset()
    est = ff.FastFlowEstimator(kind).create_flow(D, hidden_features=32, num_layers=1)
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    est.train_flow(loader, num_epochs=1, device="cpu")
    assert len(est.training_loss) == 1 and np.isfinite(est.training_loss[0])

    x = ds.patches.clone()
    x[3, 1] = float("nan")
    lp = est.log_prob(x, steps=2)
    assert lp.shape == (len(x),)
    assert np.isnan(lp[3]) and np.isfinite(np.delete(lp, 3)).all()

    samples = est.sample(10, steps=2)
    assert samples.shape == (10, D) and torch.isfinite(samples).all()


def test_regflow_trains_on_teacher_pairs():
    torch.manual_seed(0)
    ds = _ToyDataset()
    teacher = ff.FastFlowEstimator("falcon").create_flow(
        D, hidden_features=32, num_layers=1
    )
    teacher.train_flow(
        torch.utils.data.DataLoader(ds, batch_size=64), num_epochs=1, device="cpu"
    )
    pairs = ff.teacher_pairs(teacher, 128, D, steps=2)
    student = ff.FastFlowEstimator("regflow").create_flow(D)
    student.y_mean, student.y_std = teacher.y_mean, teacher.y_std
    student.train_flow(pairs=pairs, num_epochs=1, batch_size=64, device="cpu")
    assert np.isfinite(student.log_prob(ds.patches)).all()
