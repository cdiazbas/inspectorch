from .density_estimator import DensityEstimator
from .normalizing_flow import NormalizingFlowBackend
from .flow_matching import FlowMatchingBackend
from .flow_matching_sbi import FlowMatchingSBIBackend

from .datasets import GeneralizedPatchedDataset

__all__ = [
    "DensityEstimator",
    "NormalizingFlowBackend",
    "FlowMatchingBackend",
    "FlowMatchingSBIBackend",
    "GeneralizedPatchedDataset",
]

__version__ = "0.1.0"
__author__ = "Carlos Jose Diaz Baso"
