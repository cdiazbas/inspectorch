import torch.nn as nn
import numpy as np

from .normalizing_flow import NormalizingFlowBackend
from .flow_matching import FlowMatchingBackend
from .flow_matching_sbi import FlowMatchingSBIBackend



class DensityEstimator(nn.Module):
    """
    Unified interface for Density Estimation using either Normalizing Flows or
    Flow Matching.

    Args:
        type (str): The type of density estimator to use.
                    Options: "normalizing_flow" (default), "flow_matching", or "flow_matching_sbi".
        *args: Additional positional arguments passed to the backend constructor.
        **kwargs: Additional keyword arguments passed to the backend constructor.
    """

    def __init__(self, type: str = "normalizing_flow", *args, **kwargs):
        super().__init__()
        self.type = type.lower()

        if self.type == "normalizing_flow":
            self.backend = NormalizingFlowBackend(*args, **kwargs)
        elif self.type == "flow_matching":
            self.backend = FlowMatchingBackend(*args, **kwargs)
        elif self.type == "flow_matching_sbi":
            self.backend = FlowMatchingSBIBackend(*args, **kwargs)
        elif self.type == "flow_matching_cfm":
            from .flow_matching_cfm import FlowMatchingCFMBackend
            self.backend = FlowMatchingCFMBackend(*args, **kwargs)
        else:
            raise ValueError(
                f"Unknown type: {type}. Choose 'normalizing_flow', 'flow_matching', 'flow_matching_sbi', or 'flow_matching_cfm'."
            )

    @property
    def velocity_model(self):
        """
        Expose the backend's velocity model directly.
        """
        if hasattr(self.backend, "velocity_model"):
            return self.backend.velocity_model
        return None

    def create_flow(self, *args, **kwargs) -> None:
        """
        Forward creation call to backend.
        """
        self.backend.create_flow(*args, **kwargs)

    def train_flow(self, *args, **kwargs) -> None:
        """
        Forward training call to backend.
        """
        self.backend.train_flow(*args, **kwargs)

    def log_prob(self, *args, **kwargs) -> np.ndarray:
        """
        Forward log_prob call to backend.
        """
        return self.backend.log_prob(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Forward sample call to backend (if supported).
        """
        if hasattr(self.backend, "sample"):
            return self.backend.sample(*args, **kwargs)
        else:
            raise NotImplementedError(f"Backend {self.type} does not support sampling.")

    def print_summary(self):
        """
        Forward summary call to backend.
        """
        self.backend.print_summary()

    def check_variables(self, *args, **kwargs):
        """
        Forward check_variables call to backend.
        """
        self.backend.check_variables(*args, **kwargs)

    def plot_train_loss(self, *args, **kwargs):
        """
        Forward plot_train_loss call to backend.
        """
        self.backend.plot_train_loss(*args, **kwargs)
