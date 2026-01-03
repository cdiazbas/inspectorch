from .density_estimator import DensityEstimator
from .normalizing_flow import NormalizingFlowBackend
from .flow_matching_ffm import FlowMatchingBackend as FlowMatchingFFMBackend
from .flow_matching_sbi import FlowMatchingSBIBackend

from .datasets import GeneralizedPatchedDataset

__all__ = [
    "DensityEstimator",
    "NormalizingFlowBackend",
    "FlowMatchingFFMBackend",
    "FlowMatchingSBIBackend",
    "GeneralizedPatchedDataset",
]

__version__ = "0.1.0"
__author__ = "Carlos Jose Diaz Baso"
