"""
API-surface tests across every backend.

The point is coverage of the whole matrix rather than depth: each public
entry point is exercised on each backend so that a change to one backend
cannot quietly break another. Backends whose third-party library is
absent are skipped, not failed.
"""

import numpy as np
import pytest
import torch

from conftest import (
    ALL_BACKENDS,
    ARCHITECTURES,
    FLOW_MATCHING_BACKENDS,
    build_estimator,
    require_backend,
)


# ------------------------------------------------------------ construction
@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_create_flow(kind):
    estimator = build_estimator(kind)
    assert estimator.backend is not None


@pytest.mark.parametrize("kind", FLOW_MATCHING_BACKENDS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_builds_on_every_flow_matching_backend(kind, architecture):
    """
    Guards the FourierMLP duplicate-keyword TypeError, among others.
    """
    estimator = build_estimator(kind, architecture=architecture)
    assert estimator.velocity_model is not None


@pytest.mark.parametrize("kind", FLOW_MATCHING_BACKENDS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_trains_scores_and_samples(
    kind, architecture, toy_dataset, toy_loader
):
    """Building is not running: train, log_prob and sample each cell once."""
    estimator = build_estimator(kind, architecture=architecture)
    estimator.train_flow(toy_loader, learning_rate=1e-3, num_epochs=1, device="cpu")

    log_prob = estimator.log_prob(
        toy_dataset, device="cpu", solver_method="euler", steps=10
    )
    samples = estimator.sample(4, device="cpu")
    samples = samples.detach().cpu().numpy() if torch.is_tensor(samples) else samples

    assert log_prob.shape == (len(toy_dataset),) and np.isfinite(log_prob).all()
    assert samples.shape == (4, 4) and np.isfinite(samples).all()


def test_unknown_backend_raises():
    from inspectorch import DensityEstimator

    with pytest.raises(ValueError, match="Unknown type"):
        DensityEstimator(type="not_a_backend")


@pytest.mark.parametrize("kind", FLOW_MATCHING_BACKENDS)
def test_unknown_architecture_raises(kind):
    require_backend(kind)
    from inspectorch import DensityEstimator

    estimator = DensityEstimator(type=kind)
    with pytest.raises(ValueError, match="[Uu]nknown architecture"):
        estimator.create_flow(input_size=4, architecture="NoSuchNet")


# ------------------------------------------------------------------ device
@pytest.mark.parametrize("kind", ALL_BACKENDS)
@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "mps"])
def test_train_accepts_any_device_string(kind, device, toy_loader):
    """
    Regression: configure_device's resolved value used to be discarded,
    so a string like 'auto' reached .to() and raised RuntimeError. Every
    one of these must resolve to a real device on any machine.
    """
    estimator = build_estimator(kind)
    estimator.train_flow(toy_loader, learning_rate=1e-3, num_epochs=1, device=device)


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "mps", "cuda:0", "nonsense"])
def test_resolve_device_always_returns_a_usable_device(device):
    from inspectorch.utils import resolve_device

    resolved = resolve_device(device, verbose=False)
    torch.empty(1, device=torch.device(resolved))  # must not raise


# ---------------------------------------------------------------- training
@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_train_flow_records_one_loss_per_epoch(kind, toy_loader):
    estimator = build_estimator(kind)
    estimator.train_flow(toy_loader, learning_rate=1e-3, num_epochs=2, device="cpu")

    losses = estimator.backend.training_loss
    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)


@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_train_flow_argument_order_is_identical_across_backends(kind):
    """
    Regression: CFM took (train_loader, num_epochs, learning_rate) where
    the others take (train_loader, learning_rate, num_epochs), so calling
    positionally trained for a fraction of an epoch.
    """
    import inspect

    estimator = build_estimator(kind)
    names = list(inspect.signature(estimator.backend.train_flow).parameters)
    assert names[:3] == ["train_loader", "learning_rate", "num_epochs"]
    assert "device" in names


@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_save_model_writes_a_loadable_checkpoint(kind, toy_loader, tmp_path):
    """
    Regression: CFM accepted save_model=True and never wrote a file.
    """
    target = tmp_path / "model.pth"
    estimator = build_estimator(kind)
    estimator.train_flow(
        toy_loader,
        learning_rate=1e-3,
        num_epochs=1,
        device="cpu",
        output_model=str(target),
        save_model=True,
    )

    assert target.exists(), f"{kind} did not write a checkpoint"
    assert torch.load(target, map_location="cpu")


@pytest.mark.parametrize(
    "kind", ["normalizing_flow", "flow_matching_ffm", "flow_matching_sbi"]
)
def test_reloaded_model_scores_like_the_trained_one(
    kind, toy_dataset, toy_loader, tmp_path
):
    """
    A checkpoint carries the training statistics, so a model reloaded without
    the training loader scores raw rows exactly as the trained one did.
    """
    target = tmp_path / "model.pth"
    kwargs = {} if kind == "normalizing_flow" else dict(solver_method="euler", steps=10)
    trained = build_estimator(kind)
    trained.train_flow(
        toy_loader,
        learning_rate=1e-3,
        num_epochs=1,
        device="cpu",
        output_model=str(target),
        save_model=True,
    )
    reloaded = build_estimator(kind, seed=1)
    reloaded.train_flow(
        None, num_epochs=0, load_existing=True, output_model=str(target), device="cpu"
    )
    rows = toy_dataset.patches.clone()
    np.testing.assert_array_equal(
        reloaded.log_prob(rows, device="cpu", **kwargs),
        trained.log_prob(rows, device="cpu", **kwargs),
    )


def test_a_bare_state_dict_checkpoint_still_loads(tmp_path):
    """Checkpoints written before the statistics were stored keep loading."""
    from inspectorch.utils import load_checkpoint

    net = torch.nn.Linear(3, 2)
    torch.save(net.state_dict(), tmp_path / "old.pth")
    state, y_mean, y_std = load_checkpoint(tmp_path / "old.pth")
    assert y_mean is None and y_std is None
    torch.nn.Linear(3, 2).load_state_dict(state)


# ----------------------------------------------------------------- outputs
@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_log_prob_returns_one_finite_value_per_sample(kind, toy_dataset, toy_loader):
    estimator = build_estimator(kind)
    estimator.train_flow(toy_loader, learning_rate=1e-3, num_epochs=1, device="cpu")

    log_prob = estimator.log_prob(toy_dataset, device="cpu")

    assert log_prob.shape == (len(toy_dataset),)
    assert np.isfinite(log_prob).all()


@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_sample_returns_the_requested_shape(kind, toy_loader):
    estimator = build_estimator(kind)
    estimator.train_flow(toy_loader, learning_rate=1e-3, num_epochs=1, device="cpu")

    samples = estimator.sample(8, device="cpu")
    samples = samples.detach().cpu().numpy() if torch.is_tensor(samples) else samples

    assert samples.shape == (8, 4)
    assert np.isfinite(samples).all()


@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_sample_is_in_the_data_coordinates(kind):
    """sample() undoes the training standardisation in every backend."""
    estimator = build_estimator(kind)
    estimator.backend.y_mean = torch.full((4,), 100.0)
    estimator.backend.y_std = torch.full((4,), 0.01)

    samples = estimator.sample(500, device="cpu")
    samples = samples.detach().cpu().numpy() if torch.is_tensor(samples) else samples

    assert abs(float(np.mean(samples)) - 100.0) < 1.0


class _PlainDataset(torch.utils.data.Dataset):
    """A map-style dataset with stats but no .patches."""

    def __init__(self, n=200, d=4):
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, d, generator=generator)
        self.y_mean, self.y_std = self.x.mean(0), self.x.std(0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_check_variables_accepts_a_dataset_without_patches(kind):
    import matplotlib.pyplot as plt

    estimator = build_estimator(kind)
    loader = torch.utils.data.DataLoader(_PlainDataset(), batch_size=50)
    try:
        estimator.check_variables(loader, plot_variables=[0, 1], rel_size=0.5)
    finally:
        plt.close("all")


@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_print_summary_before_and_after_create(kind, capsys):
    require_backend(kind)
    from inspectorch import DensityEstimator

    DensityEstimator(type=kind)  # summary on an uncreated model must not raise
    estimator = build_estimator(kind)
    estimator.print_summary()

    assert "params" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("kind", ALL_BACKENDS)
def test_plot_train_loss_works_before_training(kind):
    """
    Regression: CFM only created training_loss inside train_flow, so
    plotting before training raised AttributeError.
    """
    import matplotlib

    matplotlib.use("Agg")

    estimator = build_estimator(kind)
    estimator.plot_train_loss(show_plot=False)


# ------------------------------------------------------------ lazy imports
def test_package_imports_without_optional_backends(monkeypatch):
    """
    Regression: importing the package used to require flow_matching,
    torchcfm, torchdiffeq and pot all to be installed, so the
    normalizing-flow path could not be used on its own.
    """
    import builtins
    import importlib

    blocked = {"flow_matching", "torchcfm", "torchdiffeq", "ot"}
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"{name} blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for module in list(__import__("sys").modules):
        if module.split(".")[0] in blocked or module.startswith("inspectorch"):
            monkeypatch.delitem(__import__("sys").modules, module, raising=False)

    package = importlib.import_module("inspectorch")
    estimator = package.DensityEstimator(type="normalizing_flow")
    estimator.create_flow(input_size=4, num_layers=2, hidden_features=16, num_bins=4)

    with pytest.raises(ImportError):
        package.DensityEstimator(type="flow_matching_ffm")


# ------------------------------------------------------------ block style
@pytest.mark.parametrize("kind", FLOW_MATCHING_BACKENDS)
@pytest.mark.parametrize("style", ["adaln_zero", "legacy"])
def test_adamlp_block_style_is_selectable(kind, style):
    estimator = build_estimator(kind, architecture="AdaMLP", block_style=style)
    blocks = [
        layer for layer in estimator.velocity_model.layers if hasattr(layer, "style")
    ]
    assert blocks and all(block.style == style for block in blocks)


def test_block_styles_share_a_parameter_set():
    """
    The two styles differ only in the forward pass.

    That is what makes a checkpoint load silently into the wrong style,
    so it is asserted rather than left implicit: anyone adding a
    parameter to one style has to confront this test.
    """
    from inspectorch.velocity_models import AdaMLPBlock

    zero = AdaMLPBlock(16, 16, style="adaln_zero")
    legacy = AdaMLPBlock(16, 16, style="legacy")

    assert set(zero.state_dict()) == set(legacy.state_dict())


def test_adaln_zero_starts_as_identity_and_legacy_does_not():
    """
    The behavioural difference the styles exist to express.
    """
    from inspectorch.velocity_models import AdaMLPBlock

    x, cond = torch.randn(8, 16), torch.zeros(8, 16)

    zero = AdaMLPBlock(16, 16, style="adaln_zero")
    legacy = AdaMLPBlock(16, 16, style="legacy")

    with torch.no_grad():
        assert torch.allclose(zero(x, cond), x, atol=1e-6)
        assert not torch.allclose(legacy(x, cond), x, atol=1e-6)


def test_unknown_block_style_raises():
    from inspectorch.velocity_models import AdaMLPBlock

    with pytest.raises(ValueError, match="Unknown block style"):
        AdaMLPBlock(16, 16, style="not_a_style")


def test_time_scale_scales_the_time_input_and_defaults_to_one():
    """time_scale=k is the same network fed k * t; the default changes nothing."""
    from inspectorch.velocity_models import VectorFieldAdaMLP

    torch.manual_seed(0)
    scaled = VectorFieldAdaMLP(
        4, 0, hidden_features=16, num_layers=2, time_scale=1000.0
    )
    plain = VectorFieldAdaMLP(4, 0, hidden_features=16, num_layers=2)
    # Break the zero initialisation, or the output does not depend on t at all.
    with torch.no_grad():
        for layer in scaled.layers[1:]:
            layer.ada_ln[-1].weight.normal_(0, 0.1)
        scaled.final_layer.weight.normal_(0, 0.1)
    plain.load_state_dict(scaled.state_dict())  # same parameter set

    t, x = torch.rand(8), torch.randn(8, 4)
    with torch.no_grad():
        torch.testing.assert_close(scaled(t, x), plain(1000.0 * t, x), rtol=0, atol=0)
    assert plain.time_scale == 1.0


@pytest.mark.parametrize("kind", FLOW_MATCHING_BACKENDS)
def test_time_scale_reaches_the_adamlp_network(kind):
    estimator = build_estimator(kind, architecture="AdaMLP", time_scale=1000.0)
    assert estimator.velocity_model.time_scale == 1000.0
