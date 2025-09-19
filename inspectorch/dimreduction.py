import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

DEFAULT_N_COMPONENTS = 10


def apply_dim_reduction(data, dim_reduction):
    """
    Applies dimensionality reduction to the input data using the specified method.

    Parameters:
        data (array-like): The input data to be reduced.
        dim_reduction (dict): Dictionary specifying the dimensionality reduction method and its parameters.
            Supported keys:
                - 'method' (str): The reduction method to use ('pca' supported).
                - 'n_components' (int, optional): Number of components for PCA (default: 10).

    Returns:
        array-like: The data transformed to reduced dimensions.

    Raises:
        ValueError: If an unknown dimensionality reduction method is specified.
    """
    method = dim_reduction.get('method', 'pca')
    if method == 'pca':
        n_components = dim_reduction.get('n_components', DEFAULT_N_COMPONENTS)
        reducer = PCAReducer(n_components)
        reduced_data = reducer.fit_transform(data)
        return reduced_data
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


class PCAReducer:
    """
    A wrapper class for performing Principal Component Analysis (PCA) for dimensionality reduction.

    Args:
        n_components (int): Number of principal components to retain.

    Attributes:
        n_components (int): Number of principal components.
        model (sklearn.decomposition.PCA): The underlying PCA model.
        fitted (bool): Indicates whether the PCA model has been fitted.

    Methods:
        fit(X):
            Fits the PCA model to the input data X.
            Args:
                X (numpy.ndarray or torch.Tensor): Input data to fit the PCA model.
            Returns:
                None

        transform(X):
            Applies the fitted PCA model to transform the input data X.
            Args:
                X (numpy.ndarray or torch.Tensor): Input data to transform.
            Returns:
                torch.Tensor: Transformed data with reduced dimensions (output is always a torch.Tensor, regardless of input type).

        fit_transform(X):
            Fits the PCA model to X and then transforms X.
            Args:
                X (numpy.ndarray or torch.Tensor): Input data to fit and transform.
            Returns:
                torch.Tensor: Transformed data with reduced dimensions (output is always a torch.Tensor, regardless of input type).
    """
    def __init__(self, n_components):
        self.n_components = n_components
        # Initialize PCA model
        self.model = PCA(n_components=n_components)
        self.fitted = False

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.model.fit(X)
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("PCA model is not fitted yet. Call 'fit' before 'transform'.")
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return torch.tensor(self.model.transform(X), dtype=torch.float32)

    def fit_transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.fit(X)
        return self.transform(X)



