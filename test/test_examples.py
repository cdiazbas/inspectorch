"""
Runs the shipped example scripts end to end and checks their outcome.

The examples are the documented entry point, so a change that breaks
them breaks the first thing a new user runs. Each script is executed
once, in a temporary directory, against a subsampled cube and with the
epoch count capped -- the test checks that the whole pipeline holds
together (data loading, patching, training, log_prob, plotting), not
that the model converges.

They are marked `slow`. Run everything else with:

    pytest -m "not slow"
"""

import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import require_backend

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "example"

EXAMPLES = {
    "01_normalizing_flow.py": "normalizing_flow",
    "02_flow_matching.py": "flow_matching_ffm",
}

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run_example(tmp_path_factory):
    """
    Execute an example once and cache its namespace.

    Module-scoped so each script runs a single time no matter how many
    assertions read from it; the scripts are the slowest thing here.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from inspectorch.density_estimator import DensityEstimator

    cache = {}

    def _run(script):
        if script in cache:
            return cache[script]

        require_backend(EXAMPLES[script])
        source = EXAMPLE_DIR / script
        if not source.exists():
            pytest.skip(f"{script} not present")

        sandbox = tmp_path_factory.mktemp(script.replace(".py", ""))
        # Subsample the bundled cube rather than editing the examples:
        # they load it by name from the working directory. The full cube
        # is ~87k spectra and example 02 runs an ODE solve per batch over
        # all of them, which takes minutes.
        for asset in EXAMPLE_DIR.glob("*.npz"):
            with np.load(asset) as bundle:
                contents = dict(bundle)
            if "data" in contents and contents["data"].ndim == 3:
                contents["data"] = contents["data"][:48, :, :48]
            np.savez(sandbox / asset.name, **contents)
        (sandbox / "models").mkdir(exist_ok=True)
        shutil.copy(source, sandbox / script)

        real_show, real_train = plt.show, DensityEstimator.train_flow
        original_cwd = Path.cwd()

        def brief_train(self, *args, **kwargs):
            kwargs["num_epochs"] = 1
            kwargs.pop("load_existing", None)
            return real_train(self, *args, **kwargs)

        plt.show = lambda *a, **k: None
        DensityEstimator.train_flow = brief_train
        sys.path.insert(0, str(sandbox))
        import os

        os.chdir(sandbox)
        try:
            namespace = runpy.run_path(str(sandbox / script), run_name="__main__")
        finally:
            os.chdir(original_cwd)
            sys.path.remove(str(sandbox))
            plt.show = real_show
            DensityEstimator.train_flow = real_train

        cache[script] = (namespace, sandbox)
        return cache[script]

    return _run


@pytest.mark.parametrize("script", list(EXAMPLES))
def test_example_writes_the_output_it_promises(script, run_example):
    """The script completes and produces the figures and model it saves."""
    _, sandbox = run_example(script)

    written = list((sandbox / "models").glob("*"))
    assert written, f"{script} produced no output in models/"
    assert [f for f in written if f.suffix == ".png"], f"{script} wrote no figures"
    assert all(f.stat().st_size > 0 for f in written), "an output file is empty"


@pytest.mark.parametrize("script", list(EXAMPLES))
def test_example_log_prob_is_finite(script, run_example):
    """A NaN log-prob would still plot, just as a blank map."""
    namespace, _ = run_example(script)

    log_prob = namespace.get("log_prob")
    assert log_prob is not None, "example did not expose log_prob"
    log_prob = np.asarray(log_prob)

    assert log_prob.ndim == 1
    assert np.isfinite(log_prob).all(), "example produced non-finite log-probs"


@pytest.mark.parametrize("script", list(EXAMPLES))
def test_example_log_prob_map_matches_the_image(script, run_example):
    """
    The examples reshape log_prob back onto the spatial grid. A length
    mismatch silently corrupts the map rather than raising, so the
    rare-event positions would be wrong without any error.
    """
    namespace, _ = run_example(script)

    log_prob = np.asarray(namespace["log_prob"])
    reshaped = np.asarray(namespace["log_prob_reshaped"])

    assert reshaped.ndim == 2
    assert reshaped.size == log_prob.size
    assert reshaped.shape == (namespace["ny"], namespace["nx"])


@pytest.mark.parametrize("script", list(EXAMPLES))
def test_example_identifies_rare_events(script, run_example):
    """
    The point of the library: the lowest-log-prob spectra are selected.
    Check the example actually picked some, and that they really are the
    least likely ones.
    """
    namespace, _ = run_example(script)

    log_prob = np.asarray(namespace["log_prob"])
    indices = np.asarray(namespace["idxs"])

    assert indices.size > 0
    assert np.isin(indices, np.argsort(log_prob)[: indices.size]).all(), (
        "the selected 'least common' spectra are not the lowest-log-prob ones"
    )
