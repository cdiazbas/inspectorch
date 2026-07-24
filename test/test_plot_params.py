"""
Tests for the LaTeX availability fallback in plot_params.set_params().

The "no latex installed" machine is faked by mocking shutil.which, so
these tests pass regardless of whether the host has a TeX distribution.
"""

import importlib.util
import os
import shutil
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as pylab
import pytest

# Load plot_params directly from its file so the test doesn't pull in the
# heavy optional dependencies imported by the package __init__.
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "inspectorch", "plot_params.py"
)
_spec = importlib.util.spec_from_file_location("plot_params", _MODULE_PATH)
plot_params = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_params)


@pytest.fixture(autouse=True)
def restore_rcparams():
    """Snapshot rcParams so set_params() doesn't leak into other tests."""
    saved = pylab.rcParams.copy()
    yield
    dict.update(pylab.rcParams, saved)


def test_fallback_when_latex_missing(monkeypatch):
    """Without a latex executable, usetex is disabled with a warning."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    with pytest.warns(UserWarning, match="No 'latex' executable found"):
        plot_params.set_params()

    assert pylab.rcParams["text.usetex"] is False
    assert pylab.rcParams["text.latex.preamble"] == ""


def test_usetex_kept_when_latex_present(monkeypatch):
    """With a latex executable available, usetex stays enabled."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/bin/latex")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_params.set_params()

    assert pylab.rcParams["text.usetex"] is True
    assert "newcent" in pylab.rcParams["text.latex.preamble"]


def test_params_dict_not_mutated(monkeypatch):
    """The fallback works on a copy; the module-level dict keeps usetex True."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    with pytest.warns(UserWarning):
        plot_params.set_params()

    assert plot_params.params["text.usetex"] is True
    assert "text.latex.preamble" in plot_params.params
