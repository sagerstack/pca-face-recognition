"""
PCA (Principal Component Analysis) Implementation from First Principles

This module implements PCA from mathematical fundamentals without relying on
scikit-learn's PCA functionality. It follows the mathematical approach from the
existing notebook and is specifically designed for face recognition applications.

Mathematical Foundation:
- PCA Optimization Problem: maximize w^T C w subject to ||w||² = 1
- Eigenvalue Equation: Cw = λw
- Covariance Matrix: C = (1/(n-1)) Σ(x_i - μ)(x_i - μ)^T

Author: PCA Face Recognition Team
"""

import numpy as np
from scipy.linalg import eigh
from typing import Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


class PCA:
    """
    Principal Component Analysis implementation from first principles.

    This class implements PCA using the covariance matrix approach and eigenvalue
    decomposition, following the mathematical foundations without relying on
    scikit-learn's PCA implementation.

    Mathematical Approach:
    1. Compute the mean of the data
    2. Center the data by subtracting the mean
    3. Compute the covariance matrix: C = (1/(n-1)) * X_centered.T @ X_centered
    4. Perform eigenvalue decomposition: C @ w = λ @ w
    5. Sort eigenvalues and eigenvectors in descending order
    6. Select top k eigenvectors as principal components
    """

    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize PCA from first principles.

        Args:
            n_components: Number of principal components to keep.
                         If None, keeps all components.
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.n_samples_seen_ = 0
        self.n_features_ = 0

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and convert input data.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Validated numpy array

        Raises:
            ValueError: If input is invalid
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {X.ndim}D array")

        if X.shape[0] < 1:
            raise ValueError("Input must have at least 1 sample")

        if X.shape[1] < 2:
            raise ValueError("PCA requires at least 2 features")

        # Check for NaN or infinite values
        if not np.all(np.isfinite(X)):
            raise ValueError("Input contains NaN or infinite values")

        return X.astype(np.float64)

    def _validate_fitting_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate input data for fitting (requires at least 2 samples).

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Validated numpy array

        Raises:
            ValueError: If input is invalid
        """
        X = self._validate_input(X)

        if X.shape[0] < 2:
            raise ValueError("PCA fitting requires at least 2 samples")

        return X

    def _compute_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the mean of the data.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Mean vector of shape (n_features,)
        """
        return np.mean(X, axis=0)

    def _center_data(self, X: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """
        Center the data by subtracting the mean.

        Args:
            X: Input data of shape (n_samples, n_features)
            mean: Mean vector of shape (n_features,)

        Returns:
            Centered data of shape (n_samples, n_features)
        """
        return X - mean

    def _compute_covariance_matrix(self, X_centered: np.ndarray) -> np.ndarray:
        """
        Compute the covariance matrix.

        Mathematical formula:
        C = (1/(n-1)) * X_centered.T @ X_centered

        Args:
            X_centered: Centered data of shape (n_samples, n_features)

        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        n_samples = X_centered.shape[0]
        cov_matrix = (1.0 / (n_samples - 1)) * X_centered.T @ X_centered

        # Ensure symmetry for numerical stability
        cov_matrix = (cov_matrix + cov_matrix.T) / 2

        return cov_matrix

    def _eigen_decomposition(self, cov_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition of the covariance matrix.

        Uses scipy.linalg.eigh for symmetric matrices, which is more stable
        than general eigenvalue decomposition.

        Args:
            cov_matrix: Covariance matrix of shape (n_features, n_features)

        Returns:
            Tuple of (eigenvalues, eigenvectors)
            - eigenvalues: Array of shape (n_features,) sorted in descending order
            - eigenvectors: Array of shape (n_features, n_features) columns are eigenvectors
        """
        eigenvalues, eigenvectors = eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors

    def _select_components(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the top k principal components.

        Args:
            eigenvalues: All eigenvalues sorted in descending order
            eigenvectors: All eigenvectors columns sorted by eigenvalues

        Returns:
            Tuple of (selected_eigenvalues, selected_eigenvectors)
        """
        n_components = self.n_components if self.n_components is not None else len(eigenvalues)
        n_components = min(n_components, len(eigenvalues))

        selected_eigenvalues = eigenvalues[:n_components]
        selected_eigenvectors = eigenvectors[:, :n_components]

        return selected_eigenvalues, selected_eigenvectors

    def _compute_explained_variance_ratio(self, explained_variance: np.ndarray, total_variance: float) -> np.ndarray:
        """
        Compute the proportion of variance explained by each component.

        Args:
            explained_variance: Variance explained by each component
            total_variance: Total variance in the data

        Returns:
            Array of explained variance ratios
        """
        return explained_variance / total_variance

    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit the PCA model to the data.

        This method computes the principal components from the input data using
        the mathematical approach based on eigenvalue decomposition of the
        covariance matrix.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Self (the fitted PCA instance)

        Example:
            >>> pca = PCA(n_components=50)
            >>> pca.fit(X_train)
            >>> print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        """
        # Validate input
        X = self._validate_fitting_input(X)

        # Store data information
        self.n_samples_, self.n_features_ = X.shape

        # Step 1: Compute mean
        print("Computing data mean...")
        self.mean_ = self._compute_mean(X)

        # Step 2: Center data
        print("Centering data...")
        X_centered = self._center_data(X, self.mean_)

        # Step 3: Compute covariance matrix
        print("Computing covariance matrix...")
        cov_matrix = self._compute_covariance_matrix(X_centered)

        # Step 4: Eigenvalue decomposition
        print("Performing eigenvalue decomposition...")
        self.eigenvalues_, self.eigenvectors_ = self._eigen_decomposition(cov_matrix)

        # Step 5: Select top k components
        n_components = self.n_components if self.n_components is not None else len(self.eigenvalues_)
        n_components = min(n_components, len(self.eigenvalues_))
        print(f"Selecting top {n_components} components...")
        self.explained_variance_, selected_eigenvectors = self._select_components(
            self.eigenvalues_, self.eigenvectors_
        )

        # Store components in the correct shape (n_components, n_features)
        # The eigenvectors are stored as columns, so we transpose
        self.components_ = selected_eigenvectors.T

        # Store the actual number of components used
        self.n_components = n_components

        # Step 6: Compute explained variance ratio
        total_variance = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self._compute_explained_variance_ratio(
            self.explained_variance_, total_variance
        )

        print(f"PCA fitting complete! Selected {len(self.components_)} components")
        print(f"Total explained variance: {np.sum(self.explained_variance_ratio_):.4f}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected onto the principal component axes found during fitting.

        Mathematical formula: Y = (X - mean) @ components.T

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)

        Raises:
            ValueError: If the model has not been fitted yet

        Example:
            >>> X_transformed = pca.transform(X_test)
            >>> print(f"Original shape: {X_test.shape}, Transformed shape: {X_transformed.shape}")
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("This PCA instance is not fitted yet. Call 'fit' first.")

        X = self._validate_input(X)

        # Center the data using the mean from fitting
        X_centered = self._center_data(X, self.mean_)

        # Project onto principal components (components are already in correct shape)
        X_transformed = X_centered @ self.components_.T

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        This is equivalent to calling fit(X).transform(X), but more computationally
        efficient.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)

        Example:
            >>> X_transformed = pca.fit_transform(X_train)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to its original space.

        Reconstructs the original data from the reduced dimensionality representation.

        Mathematical formula: X_original_approx = X_transformed @ components + mean

        Args:
            X_transformed: Transformed data of shape (n_samples, n_components)

        Returns:
            Reconstructed data of shape (n_samples, n_features)

        Raises:
            ValueError: If the model has not been fitted yet

        Example:
            >>> X_reconstructed = pca.inverse_transform(X_transformed)
            >>> reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("This PCA instance is not fitted yet. Call 'fit' first.")

        X_transformed = np.asarray(X_transformed)

        if X_transformed.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {X_transformed.ndim}D array")

        if X_transformed.shape[1] != self.n_components:
            raise ValueError(f"Input has {X_transformed.shape[1]} components, but PCA expects {self.n_components}")

        # Reconstruct the data (components are stored as eigenvectors)
        X_reconstructed = X_transformed @ self.components_ + self.mean_

        return X_reconstructed

    def get_cumulative_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the cumulative explained variance ratio.

        Returns:
            Array of cumulative explained variance ratio for each component
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("This PCA instance is not fitted yet. Call 'fit' first.")

        return np.cumsum(self.explained_variance_ratio_)

    def get_optimal_components(self, variance_threshold: float = 0.95) -> int:
        """
        Get the optimal number of components to explain a given variance threshold.

        Args:
            variance_threshold: Desired proportion of variance to explain (0-1)

        Returns:
            Number of components needed to explain the specified variance

        Example:
            >>> n_components = pca.get_optimal_components(variance_threshold=0.95)
            >>> print(f"Need {n_components} components to explain 95% of variance")
        """
        if not 0 < variance_threshold <= 1:
            raise ValueError("variance_threshold must be between 0 and 1")

        cumulative_variance = self.get_cumulative_explained_variance_ratio()
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        return n_components

    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calculate the mean squared reconstruction error for given data.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Mean squared reconstruction error

        Example:
            >>> mse = pca.get_reconstruction_error(X_test)
            >>> print(f"Reconstruction MSE: {mse:.6f}")
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)

        mse = np.mean((X - X_reconstructed) ** 2)

        return mse


def main():
    """
    Example usage of PCA from first principles.
    """
    # Generate sample data
    print("Creating sample data...")
    np.random.seed(42)

    # Create data with intrinsic low dimensionality
    n_samples = 1000
    n_features = 50
    intrinsic_dim = 10

    # Generate low-dimensional data
    U = np.random.randn(n_samples, intrinsic_dim)
    V = np.random.randn(intrinsic_dim, n_features)
    X = U @ V + np.random.randn(n_samples, n_features) * 0.1

    print(f"Data shape: {X.shape}")
    print(f"Intrinsic dimensionality: {intrinsic_dim}")

    # Apply PCA
    print("\n" + "="*60)
    print("PCA FROM FIRST PRINCIPLES DEMONSTRATION")
    print("="*60)

    n_components = 20
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)

    print(f"\nResults:")
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Components shape: {pca.components_.shape}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Find optimal components for 95% variance
    optimal_components = pca.get_optimal_components(variance_threshold=0.95)
    print(f"Components for 95% variance: {optimal_components}")

    # Calculate reconstruction error
    reconstruction_error = pca.get_reconstruction_error(X)
    print(f"Reconstruction MSE: {reconstruction_error:.6f}")


if __name__ == "__main__":
    main()