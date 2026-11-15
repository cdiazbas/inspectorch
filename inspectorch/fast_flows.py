"""
Few-step density estimators, for comparison with the existing backends:

    meanflow  average-velocity flow map (Geng et al., NeurIPS 2025);
              log p from the exact Jacobian of each step
    falcon    meanflow + cycle-consistency, so the map stays invertible
              (Rehman et al., ICLR 2026)
    f2d2      meanflow + a head that predicts the average divergence, so
              log p needs no Jacobian (Ai et al., arXiv 2512.02636)
    regflow   the normalizing flow of this package trained by regression
              onto noise/data pairs instead of maximum likelihood
              (Rehman et al., arXiv 2506.01158). A sampler, not a density
              estimator here: the regression fits noise -> data on the pairs
              and leaves the data -> noise map, which the likelihood uses,
              unconstrained. On the Hinode example its held-out NLL/dim is
              -1.75 (-0.45 with teacher pairs) against -3.29 for the same
              flow trained by maximum likelihood; do not rank with it.

Convention throughout, as in the SBI backend: t = 0 is noise N(0, I),
t = 1 is (standardised) data, x_t = (1 - t) x0 + t x1, conditional
velocity v = x1 - x0.

The flow map is X(x, a, b) = x + (b - a) u(x, a, b), with u the average
velocity over [a, b]. Holding b fixed and moving a along a trajectory,
dX/da = 0 gives

    u = v + (b - a) du/da,      du/da = d_a u + (grad_x u) v,

the MeanFlow identity in this convention. The same argument applied to
the average divergence G(x, a, b) = (1/(b - a)) int_a^b div v dt gives
G = div v + (b - a) dG/da. Both identities, and the log-densities built
on them, are checked against a Gaussian whose flow is known in closed
form (test/test_fast_flows.py).

Density of a K-step map from data (t=1) to noise (t=0):
    log p(x) = log N(T(x)) + sum_k log|det dX_k/dx|      (meanflow, falcon)
    log p(x) = log N(T(x)) + sum_k (b_k - a_k) G_k        (f2d2)
The first is the exact density of the learned map when the map is
invertible; the second is exact only as far as G is.
"""

import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, jvp, vmap

from . import utils

KINDS = ("meanflow", "falcon", "f2d2", "regflow")


# =============================================================================
# Network u(x, a, b) (and G(x, a, b) for f2d2)
# =============================================================================
class _TimeEmbedding(nn.Module):
    """Sinusoidal embedding, written without in-place ops so torch.func works."""

    def __init__(self, dim=32, max_period=1000.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        arg = t * self.freqs
        return torch.cat((torch.sin(arg), torch.cos(arg)), dim=-1)


class TwoTimeNet(nn.Module):
    """
    MLP on [x, emb(a), emb(b - a)] with residual blocks. Returns u(x, a, b);
    with div_head=True, `both` also returns the average divergence G.
    """

    def __init__(
        self, input_size, hidden_features=256, num_layers=4, time_dim=32, div_head=False
    ):
        super().__init__()
        self.emb_a = _TimeEmbedding(time_dim)
        self.emb_d = _TimeEmbedding(time_dim)
        self.inp = nn.Linear(input_size + 2 * time_dim, hidden_features)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_features),
                nn.Linear(hidden_features, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, hidden_features),
            )
            for _ in range(num_layers)
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_features), nn.Linear(hidden_features, input_size)
        )
        self.div = (
            nn.Sequential(
                nn.LayerNorm(hidden_features),
                nn.Linear(hidden_features, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
            if div_head
            else None
        )

    def _features(self, x, a, b):
        h = self.inp(torch.cat((x, self.emb_a(a), self.emb_d(b - a)), dim=-1))
        for block in self.blocks:
            h = h + block(h)
        return h

    def forward(self, x, a, b):
        return self.out(self._features(x, a, b))

    def both(self, x, a, b):
        h = self._features(x, a, b)
        return self.out(h), self.div(h)[..., 0]


# =============================================================================
# Map, targets and divergences (shared by training, likelihood and tests)
# =============================================================================
def flow_step(u_fn, x, a, b):
    """X(x, a, b) = x + (b - a) u(x, a, b); a, b broadcast against x[..., :1]."""
    return x + (b - a) * u_fn(x, a, b)


def total_derivative(fn, x, a, b, v):
    """
    fn(x, a, b) and its derivative along the trajectory, d/da with x moving
    at velocity v and b fixed: a forward-mode JVP with tangents (v, 1).
    """
    return jvp(lambda x_, a_: fn(x_, a_, b), (x, a), (v, torch.ones_like(a)))


def average_velocity_target(u_fn, x, a, b, v):
    """sg(v + (b - a) du/da): the regression target of u(x, a, b)."""
    with torch.no_grad():
        _, dudt = total_derivative(u_fn, x, a, b, v)
        return v + (b - a) * dudt


def average_divergence_target(g_fn, x, a, b, v, div_v):
    """sg(div v + (b - a) dG/da): the regression target of G(x, a, b)."""
    with torch.no_grad():
        _, dgdt = total_derivative(g_fn, x, a, b, v)
        return div_v + (b - a)[..., 0] * dgdt


def divergence(u_fn, x, t, exact=False, generator=None):
    """div_x u(x, t, t): exact (D forward-mode passes) or one Hutchinson probe."""
    f = lambda x_: u_fn(x_, t, t)  # noqa: E731
    if exact:
        eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        jv = vmap(lambda e: jvp(f, (x,), (e.expand_as(x),))[1], out_dims=1)(
            eye
        )  # (B, D, D): column j = J e_j
        return torch.diagonal(jv, dim1=-2, dim2=-1).sum(-1)
    e = torch.randn(x.shape, generator=generator, device="cpu", dtype=x.dtype).to(
        x.device
    )
    return (e * jvp(f, (x,), (e,))[1]).sum(-1)


def step_logdet(u_fn, x, a, b, chunk=2048, return_sign=False):
    """
    log|det dX/dx| of one map step, exact (reverse-mode Jacobian per row).

    With return_sign, also the sign of det. A step that is close to the
    identity has det > 0; a row with det <= 0 sits where the learned map
    folds, so the change of variables does not hold there.
    """
    out, signs = [], []
    for s in range(0, x.shape[0], chunk):
        xs = x[s : s + chunk]
        a1 = a.reshape(1, 1).to(xs)
        b1 = b.reshape(1, 1).to(xs)

        def single(xi):
            return flow_step(u_fn, xi[None], a1, b1)[0]

        jac = vmap(jacrev(single))(xs)
        sign, logabs = torch.linalg.slogdet(jac)
        out.append(logabs)
        signs.append(sign)
    if return_sign:
        return torch.cat(out), torch.cat(signs)
    return torch.cat(out)


def log_normal(z):
    return -0.5 * (z.shape[-1] * math.log(2 * math.pi) + (z**2).sum(-1))


def map_log_prob(u_fn, x, steps=1, g_fn=None, return_folded=False):
    """
    log p(x) for data-side x (t = 1) through `steps` equal map steps to t = 0.
    With g_fn (average divergence), the f2d2 sum replaces the Jacobian.
    With return_folded (Jacobian route only), also a mask of the rows for
    which some step has det <= 0: there the map is not invertible and the
    value is not a density.
    """
    grid = torch.linspace(1.0, 0.0, steps + 1, device=x.device, dtype=x.dtype)
    total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    folded = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for a, b in zip(grid[:-1], grid[1:]):
        a_ = a.expand(x.shape[0], 1)
        b_ = b.expand(x.shape[0], 1)
        if g_fn is None:
            logabs, sign = step_logdet(u_fn, x, a, b, return_sign=True)
            total = total + logabs
            folded |= sign <= 0
        else:
            total = total + (b - a) * g_fn(x, a_, b_)
        x = flow_step(u_fn, x, a_, b_)
    if return_folded:
        return log_normal(x) + total, folded
    return log_normal(x) + total


def sample_times(
    n, device, generator=None, mu=-0.4, sigma=1.0, ratio=0.25, both_directions=True
):
    """
    (a, b) pairs: logit-normal times; a fraction `ratio` with a != b (the rest
    a = b, plain flow matching); with both_directions, half the pairs run
    data -> noise (b < a), which is the direction the likelihood needs.
    """

    def lognorm():
        return torch.sigmoid(mu + sigma * torch.randn(n, 1, generator=generator)).to(
            device
        )

    t1, t2 = lognorm(), lognorm()
    a, b = torch.minimum(t1, t2), torch.maximum(t1, t2)
    same = (torch.rand(n, 1, generator=generator) >= ratio).to(device)
    b = torch.where(same, a, b)
    if both_directions:
        flip = (torch.rand(n, 1, generator=generator) < 0.5).to(device)
        a, b = torch.where(flip, b, a), torch.where(flip, a, b)
    return a, b


# =============================================================================
# Losses
# =============================================================================
def meanflow_loss(net, x1, p=1.0, c=1e-3, time_kw=None):
    """MeanFlow regression with its adaptive weight sg(1 / (|delta|^2 + c)^p)."""
    x0 = torch.randn_like(x1)
    a, b = sample_times(x1.shape[0], x1.device, **(time_kw or {}))
    xa = (1 - a) * x0 + a * x1
    v = x1 - x0
    target = average_velocity_target(net, xa, a, b, v)
    delta = net(xa, a, b) - target
    sq = (delta**2).sum(-1)
    w = 1.0 / (sq.detach() + c) ** p
    return (w * sq).mean(), (xa, a, b, x0, v)


def falcon_loss(net, x1, lambda_cycle=10.0, **kw):
    """meanflow + lambda * |x_a - X(X(x_a, a, b), b, a)|^2 (Rehman et al. Eq. 8-9)."""
    loss, (xa, a, b, _, _) = meanflow_loss(net, x1, **kw)
    back = flow_step(net, flow_step(net, xa, a, b), b, a)
    return loss + lambda_cycle * ((back - xa) ** 2).sum(-1).mean()


def f2d2_loss(
    net, x1, div_weight=1.0, exact_div=False, teacher=None, generator=None, **kw
):
    """
    meanflow for u, plus the average divergence head G regressed onto
    sg(div v(x_a, a) + (b - a) dG/da). div v comes from the model's own
    instantaneous velocity u(x, a, a), or from `teacher(x, t)` if given.
    """
    x0 = torch.randn_like(x1)
    a, b = sample_times(x1.shape[0], x1.device, **(kw.pop("time_kw", None) or {}))
    xa = (1 - a) * x0 + a * x1
    v = x1 - x0
    u_target = average_velocity_target(net, xa, a, b, v)
    if teacher is None:
        velocity = net  # the model's own instantaneous velocity u(x, t, t)
    else:

        def velocity(x_, s_, t_):
            return teacher(x_, s_)

    with torch.no_grad():
        div_v = divergence(velocity, xa, a, exact=exact_div, generator=generator)
    g_target = average_divergence_target(
        lambda x_, a_, b_: net.both(x_, a_, b_)[1], xa, a, b, v, div_v
    )
    u, g = net.both(xa, a, b)
    sq = ((u - u_target) ** 2).sum(-1)
    w = 1.0 / (sq.detach() + kw.get("c", 1e-3)) ** kw.get("p", 1.0)
    return (w * sq).mean() + div_weight * ((g - g_target) ** 2).mean()


def ot_pairs(x0, x1, batch=1024):
    """Re-pair x0 with x1 by exact OT in blocks of `batch` (squared distance)."""
    import ot

    out = torch.empty_like(x0)
    for s in range(0, x0.shape[0], batch):
        z, x = x0[s : s + batch], x1[s : s + batch]
        cost = torch.cdist(z, x).pow(2).detach().cpu().double().numpy()
        plan = ot.emd(np.full(len(z), 1 / len(z)), np.full(len(x), 1 / len(x)), cost)
        out[s : s + batch] = z[torch.as_tensor(plan.argmax(axis=0), device=z.device)]
    return out


def regflow_loss(
    flow, x1, x0=None, lambda_logdet=1e-5, target_noise=0.05, ot_batch=1024
):
    """
    |f(x0) - x1|^2 + lambda (log|det df/dx0|)^2 with f: noise -> data the
    inverse of the nflows transform. Without given x0, pairs are built by
    minibatch OT (the paper uses full-batch OT or a pretrained CNF; pass
    x0 from `teacher_pairs` for the latter).
    """
    if x0 is None:
        x0 = ot_pairs(torch.randn_like(x1), x1, batch=ot_batch)
    x1 = x1 + target_noise * torch.randn_like(x1)
    xhat, logdet = flow._transform.inverse(x0)
    return ((xhat - x1) ** 2).sum(-1).mean() + lambda_logdet * (logdet**2).mean()


# =============================================================================
# Estimator with the package's API
# =============================================================================
class FastFlowEstimator(nn.Module):
    """
    create_flow / train_flow / log_prob / sample, as DensityEstimator, for
    kind in KINDS. Works on standardised data; log_prob returns a density in
    the data coordinates, like the other backends.
    """

    def __init__(self, kind="falcon"):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind = kind
        self.model: Optional[nn.Module] = None
        self.y_mean = self.y_std = None
        self.training_loss = []
        self.train_seconds = 0.0

    # ------------------------------------------------------------------
    def create_flow(
        self, input_size, hidden_features=256, num_layers=4, time_dim=32, **nf_kwargs
    ):
        if self.kind == "regflow":
            from .normalizing_flow import create_flow

            self.model = create_flow(
                input_size,
                num_layers=nf_kwargs.get("nf_layers", 5),
                hidden_features=nf_kwargs.get("nf_hidden", 64),
                num_bins=nf_kwargs.get("num_bins", 8),
            )
        else:
            self.model = TwoTimeNet(
                input_size,
                hidden_features,
                num_layers,
                time_dim,
                div_head=self.kind == "f2d2",
            )
        return self

    def _loss(self, x1, x0=None, **kw):
        if self.kind == "meanflow":
            return meanflow_loss(self.model, x1, **kw)[0]
        if self.kind == "falcon":
            return falcon_loss(self.model, x1, **kw)
        if self.kind == "f2d2":
            return f2d2_loss(self.model, x1, **kw)
        return regflow_loss(self.model, x1, x0=x0, **kw)

    # ------------------------------------------------------------------
    def train_flow(
        self,
        train_loader=None,
        learning_rate=1e-3,
        num_epochs=10,
        device="auto",
        max_seconds=None,
        pairs=None,
        batch_size=4096,
        **loss_kwargs,
    ):
        """
        Adam over train_loader (batches of raw rows; NaN rows dropped,
        standardised with the dataset's y_mean / y_std).

        regflow can instead take `pairs`: (x0, x1) tensors in standardised
        units, e.g. from `teacher_pairs`, cut into batches of `batch_size`.
        Set y_mean / y_std (the teacher's) before training on pairs.
        """
        if (train_loader is None) == (pairs is None):
            raise ValueError("Give either train_loader or pairs")
        if pairs is not None and self.kind != "regflow":
            raise ValueError("pairs are only used by regflow")
        device = torch.device(utils.resolve_device(device, verbose=False).split(",")[0])
        ds = getattr(train_loader, "dataset", None)
        if self.y_mean is None and ds is not None and hasattr(ds, "y_mean"):
            self.y_mean, self.y_std = ds.y_mean, ds.y_std
        mean = self.y_mean.to(device) if self.y_mean is not None else None
        std = self.y_std.to(device) if self.y_std is not None else None

        def batches():
            if pairs is not None:
                for idx in torch.randperm(len(pairs[0])).split(batch_size):
                    yield pairs[0][idx].to(device), pairs[1][idx].to(device)
                return
            for batch in train_loader:
                batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch = torch.as_tensor(batch, dtype=torch.float32)
                batch = batch[torch.isfinite(batch).all(1)].to(device)
                yield None, ((batch - mean) / std if mean is not None else batch)

        self.model.to(device).train()
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        t0 = time.perf_counter()
        out_of_time = False
        for _ in range(num_epochs):
            losses = []
            for x0, x1 in batches():
                if len(x1) == 0:
                    continue
                opt.zero_grad(set_to_none=True)
                if self.kind == "regflow":
                    loss = self._loss(x1, x0=x0, **loss_kwargs)
                else:
                    loss = self._loss(x1, **loss_kwargs)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                out_of_time = (
                    max_seconds is not None and time.perf_counter() - t0 > max_seconds
                )
                if out_of_time:
                    break
            self.training_loss.append(
                float(np.mean(losses)) if losses else float("nan")
            )
            if out_of_time:
                break
        self.train_seconds += time.perf_counter() - t0
        self.model.to("cpu").eval()

    # ------------------------------------------------------------------
    def log_prob(
        self, inputs, steps=4, batch_size=4096, device="cpu", dataset_normalization=True
    ):
        """
        log p for every row. NaN rows give NaN. `steps` is the number of map
        steps (ignored by regflow, whose density is exact in one pass).
        """
        data, mean, std, norm_std = utils.log_prob_inputs(
            inputs, dataset_normalization, self.y_mean, self.y_std
        )
        log_det_norm = utils.normalization_log_det(norm_std, data.shape[1])
        device = torch.device(utils.resolve_device(device, verbose=False).split(",")[0])
        self.model.to(device).eval()
        if mean is not None:
            mean, std = mean.to(device), std.to(device)
        out, n_folded, n_rows = [], 0, 0
        for s in range(0, data.shape[0], batch_size):
            x = torch.as_tensor(data[s : s + batch_size], dtype=torch.float32).to(
                device
            )
            if mean is not None:
                x = (x - mean) / std
            ok = torch.isfinite(x).all(1)
            lp = torch.full((x.shape[0],), float("nan"), device=device)
            if ok.any():
                with torch.no_grad():
                    if self.kind == "regflow":
                        lp[ok] = self.model.log_prob(x[ok])
                    elif self.kind == "f2d2":
                        lp[ok] = map_log_prob(
                            self.model,
                            x[ok],
                            steps,
                            g_fn=lambda x_, a_, b_: self.model.both(x_, a_, b_)[1],
                        )
                    else:
                        lp[ok], folded = map_log_prob(
                            self.model, x[ok], steps, return_folded=True
                        )
                        n_folded += int(folded.sum())
                        n_rows += int(ok.sum())
            out.append(lp.cpu().numpy() + log_det_norm)
        self.model.to("cpu")
        if n_folded:
            import warnings

            warnings.warn(
                f"{n_folded} of {n_rows} rows ({100 * n_folded / n_rows:.2f} %) pass "
                f"through a fold of the learned map (det <= 0 in some step): their "
                f"log p is not a density. More steps or the falcon cycle term reduce it.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.concatenate(out)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, num_samples, steps=4, device="cpu"):
        """Samples in the data coordinates."""
        device = torch.device(utils.resolve_device(device, verbose=False).split(",")[0])
        self.model.to(device).eval()
        if self.kind == "regflow":
            x, _ = self.model._transform.inverse(
                torch.randn(num_samples, self._dim(), device=device)
            )
        else:
            x = torch.randn(num_samples, self._dim(), device=device)
            grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
            for a, b in zip(grid[:-1], grid[1:]):
                x = flow_step(
                    self.model, x, a.expand(num_samples, 1), b.expand(num_samples, 1)
                )
        if self.y_mean is not None:
            x = x * self.y_std.to(device) + self.y_mean.to(device)
        self.model.to("cpu")
        return x.cpu()

    def _dim(self):
        if self.kind == "regflow":
            return self.model._distribution._shape[0]
        return self.model.out[-1].out_features


@torch.no_grad()
def teacher_pairs(teacher, n, input_size, steps=4, device="cpu", generator=None):
    """
    Noise/data pairs for regflow from a trained few-step estimator (the
    "reflow" targets of RegFlow, with a few-step teacher instead of a CNF).
    Returns (x0, x1) in standardised units.
    """
    dev = torch.device(utils.resolve_device(device, verbose=False).split(",")[0])
    teacher.model.to(dev).eval()
    x0 = torch.randn(n, input_size, generator=generator).to(dev)
    x = x0
    grid = torch.linspace(0.0, 1.0, steps + 1, device=dev)
    for a, b in zip(grid[:-1], grid[1:]):
        x = flow_step(teacher.model, x, a.expand(n, 1), b.expand(n, 1))
    teacher.model.to("cpu")
    return x0.cpu(), x.cpu()
