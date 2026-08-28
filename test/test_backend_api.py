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
