import torch

# MPS (Apple Silicon GPU) Acceleration Support
# ============================================
# Apple's Metal Performance Shaders (MPS) backend doesn't support float64,
# This monkey patch intercepts odeint calls and forces float32 when MPS is detected.
# See: https://github.com/pytorch/pytorch/issues/issues_about_mps_float64

try:
    import torchdiffeq
    _original_odeint = torchdiffeq.odeint

    def _mps_odeint(*args, **kwargs):
        """
        Wrapper for torchdiffeq.odeint that forces float32 dtype on MPS devices.
        
        MPS doesn't support float64, so we intercept the call and inject dtype=torch.float32
        in the options to work around this upstream limitation in torchdiffeq.
        """
        y0 = kwargs.get('y0')
        if y0 is None and len(args) > 1:
            y0 = args[1]
        t = kwargs.get('t')
        if t is None and len(args) > 2:
            t = args[2]

        def _check_mps(x):
            """Recursively check if any tensor in x is on MPS device."""
            if isinstance(x, torch.Tensor) and x.device.type == 'mps':
                return True
            if isinstance(x, (tuple, list)):
                return any(_check_mps(item) for item in x)
            return False

        # If input tensors are on MPS, force float32 for odeint
        if _check_mps(y0) or _check_mps(t):
            options = kwargs.get('options', {})
            if options is None:
                options = {}
            if 'dtype' not in options:
                options['dtype'] = torch.float32
            kwargs['options'] = options
            
        return _original_odeint(*args, **kwargs)

    # Apply patch to torchdiffeq
    torchdiffeq.odeint = _mps_odeint

    # Also patch flow_matching.solver.ode_solver if available
    try:
        import flow_matching.solver.ode_solver
        flow_matching.solver.ode_solver.odeint = _mps_odeint
    except ImportError:
        pass

except ImportError:
    # If torchdiffeq is not available, continue without MPS patch
    pass

from .density_estimator import DensityEstimator
from .normalizing_flow import NormalizingFlowBackend
from .flow_matching_ffm import FlowMatchingBackend as FlowMatchingFFMBackend
from .flow_matching_sbi import FlowMatchingSBIBackend
from .flow_matching_cfm import FlowMatchingCFMBackend

from .datasets import GeneralizedPatchedDataset

__all__ = [
    "DensityEstimator",
    "NormalizingFlowBackend",
    "FlowMatchingFFMBackend",
    "FlowMatchingSBIBackend",
    "FlowMatchingCFMBackend",
    "GeneralizedPatchedDataset",
]

__version__ = "0.1.0"
__author__ = "Carlos Jose Diaz Baso"
