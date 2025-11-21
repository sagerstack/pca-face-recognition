"""
Mathematical Utilities for PCA Face Recognition

This module provides mathematical utility functions and helper classes to support
the PCA implementation and face recognition system. It includes distance metrics,
normalization functions, matrix operations, and other mathematical tools
needed for eigenfaces-based face recognition.

Mathematical Foundations:
- Euclidean Distance: d(p, q) = √(Σ(p_i - q_i)²)
- Cosine Similarity: sim(p, q) = (p · q) / (||p|| ||q||)
- Manhattan Distance: d(p, q) = Σ|p_i - q_i|
- L2 Normalization: x_normalized = x / ||x||₂

Author: PCA Face Recognition Team
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DistanceMetrics:
    """
    Collection of distance metrics and similarity measures for face recognition.

    This class provides various mathematical distance functions used for comparing
    face feature vectors in the PCA-reduced space.
    """

    @staticmethod
    def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors.

        Mathematical formula:
        d(x1, x2) = √(Σ(x1_i - x2_i)²)

        Args:
            x1: First vector
            x2: Second vector

        Returns:
            Euclidean distance between the vectors

        Raises:
            ValueError: If vectors have different shapes
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)

        if x1.shape != x2.shape:
            raise ValueError(f"Vectors must have same shape: {x1.shape} vs {x2.shape}")

        return np.sqrt(np.sum((x1 - x2) ** 2))

    @staticmethod
    def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Manhattan (L1) distance between two vectors.

        Mathematical formula:
        d(x1, x2) = Σ|x1_i - x2_i|

        Args:
            x1: First vector
            x2: Second vector

        Returns:
            Manhattan distance between the vectors
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)

        if x1.shape != x2.shape:
            raise ValueError(f"Vectors must have same shape: {x1.shape} vs {x2.shape}")

        return np.sum(np.abs(x1 - x2))

    @staticmethod
    def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Mathematical formula:
        sim(x1, x2) = (x1 · x2) / (||x1|| ||x2||)

        Args:
            x1: First vector
            x2: Second vector

        Returns:
            Cosine similarity (-1 to 1, where 1 is identical)

        Raises:
            ValueError: If vectors have different shapes or are zero vectors
        """
        x1 = np.asarray(x1).flatten()
        x2 = np.asarray(x2).flatten()

        if x1.shape != x2.shape:
            raise ValueError(f"Vectors must have same shape: {x1.shape} vs {x2.shape}")

        # Calculate norms
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)

        if norm_x1 == 0 or norm_x2 == 0:
            raise ValueError("Cannot compute cosine similarity with zero vector")

        # Calculate cosine similarity
        dot_product = np.dot(x1, x2)
        cosine_sim = dot_product / (norm_x1 * norm_x2)

        # Clip to handle numerical precision issues
        return np.clip(cosine_sim, -1.0, 1.0)

    @staticmethod
    def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate cosine distance between two vectors.

        Mathematical formula:
        d_cos(x1, x2) = 1 - cos_sim(x1, x2)

        Args:
            x1: First vector
            x2: Second vector

        Returns:
            Cosine distance (0 to 2, where 0 is identical)
        """
        cosine_sim = DistanceMetrics.cosine_similarity(x1, x2)
        return 1 - cosine_sim


class NormalizationUtils:
    """
    Utility functions for data normalization and preprocessing.

    This class provides various normalization methods used in face recognition
    to ensure consistent scaling and preprocessing of face images and feature vectors.
    """

    @staticmethod
    def l2_normalize(data: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to the input data.

        Mathematical formula:
        x_normalized = x / ||x||₂

        Args:
            data: Input data of shape (n_samples, n_features) or single vector

        Returns:
            L2-normalized data
        """
        data = np.asarray(data)

        if data.ndim == 1:
            # Single vector
            norm = np.linalg.norm(data)
            if norm == 0:
                return data
            return data / norm
        else:
            # Multiple samples
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return data / norms

    @staticmethod
    def min_max_normalize(data: np.ndarray,
                         feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Apply min-max normalization to scale data to a specific range.

        Mathematical formula:
        x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min

        Args:
            data: Input data
            feature_range: Desired range (min, max)

        Returns:
            Min-max normalized data
        """
        data = np.asarray(data)
        min_val, max_val = feature_range

        if data.size == 0:
            return data

        data_min = np.min(data)
        data_max = np.max(data)

        if data_max == data_min:
            # All values are the same
            return np.full_like(data, min_val)

        # Scale to [0, 1] first
        data_scaled = (data - data_min) / (data_max - data_min)

        # Scale to desired range
        return data_scaled * (max_val - min_val) + min_val

    @staticmethod
    def z_score_normalize(data: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization (standardization).

        Mathematical formula:
        z = (x - μ) / σ

        Args:
            data: Input data

        Returns:
            Z-score normalized data
        """
        data = np.asarray(data)

        if data.size == 0:
            return data

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            # All values are the same
            return np.zeros_like(data)

        return (data - mean) / std


class MatrixUtils:
    """
    Utility functions for matrix operations and linear algebra.

    This class provides additional matrix operations useful for PCA and
    face recognition beyond what's available in NumPy and SciPy.
    """

    @staticmethod
    def is_symmetric(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
        """
        Check if a matrix is symmetric within a given tolerance.

        Args:
            matrix: Input matrix
            tolerance: Numerical tolerance for symmetry check

        Returns:
            True if matrix is symmetric within tolerance
        """
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return False
        return np.allclose(matrix, matrix.T, atol=tolerance)

    @staticmethod
    def is_positive_semi_definite(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
        """
        Check if a matrix is positive semi-definite.

        A matrix is positive semi-definite if all its eigenvalues are non-negative.

        Args:
            matrix: Input matrix (should be symmetric)
            tolerance: Tolerance for eigenvalue positivity

        Returns:
            True if matrix is positive semi-definite
        """
        matrix = np.asarray(matrix)

        if not MatrixUtils.is_symmetric(matrix):
            return False

        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.all(eigenvalues >= -tolerance)

    @staticmethod
    def matrix_condition_number(matrix: np.ndarray) -> float:
        """
        Calculate the condition number of a matrix.

        The condition number measures the sensitivity of the linear system
        to numerical errors. Higher values indicate ill-conditioning.

        Args:
            matrix: Input matrix

        Returns:
            Condition number of the matrix
        """
        matrix = np.asarray(matrix)
        return np.linalg.cond(matrix)

    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        """
        Calculate the Frobenius norm of a matrix.

        Mathematical formula:
        ||A||_F = √(Σ|a_ij|²)

        Args:
            matrix: Input matrix

        Returns:
            Frobenius norm of the matrix
        """
        matrix = np.asarray(matrix)
        return np.linalg.norm(matrix, 'fro')


class FaceRecognitionUtils:
    """
    Specialized utility functions for face recognition applications.

    This class provides specific mathematical operations used in eigenfaces
    face recognition, including confidence scoring and match analysis.
    """

    @staticmethod
    def calculate_confidence_score(distance: float,
                                  threshold: float,
                                  metric: str = 'euclidean') -> float:
        """
        Calculate confidence score for face recognition based on distance.

        Args:
            distance: Measured distance between face features
            threshold: Recognition threshold
            metric: Distance metric used ('euclidean', 'cosine', 'manhattan')

        Returns:
            Confidence score (0 to 1, where 1 is highest confidence)
        """
        if metric == 'cosine':
            # For cosine distance, lower distance = higher similarity
            confidence = 1.0 - (distance / threshold)
        else:
            # For euclidean/manhattan distance
            confidence = 1.0 - (distance / threshold)

        # Clip to [0, 1] range
        return np.clip(confidence, 0.0, 1.0)

    @staticmethod
    def find_best_match(query_features: np.ndarray,
                       database_features: np.ndarray,
                       distance_metric: str = 'euclidean') -> Tuple[int, float]:
        """
        Find the best matching face in the database.

        Args:
            query_features: Feature vector of query face
            database_features: Database of face features (n_samples, n_features)
            distance_metric: Distance metric to use

        Returns:
            Tuple of (best_match_index, best_distance)
        """
        if database_features.shape[0] == 0:
            raise ValueError("Database is empty")

        best_distance = float('inf')
        best_match_idx = 0

        for i, db_features in enumerate(database_features):
            if distance_metric == 'euclidean':
                distance = DistanceMetrics.euclidean_distance(query_features, db_features)
            elif distance_metric == 'cosine':
                distance = DistanceMetrics.cosine_distance(query_features, db_features)
            elif distance_metric == 'manhattan':
                distance = DistanceMetrics.manhattan_distance(query_features, db_features)
            else:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")

            if distance < best_distance:
                best_distance = distance
                best_match_idx = i

        return best_match_idx, best_distance

    @staticmethod
    def calculate_reconstruction_quality(original: np.ndarray,
                                       reconstructed: np.ndarray) -> dict:
        """
        Calculate quality metrics for face reconstruction.

        Args:
            original: Original face image
            reconstructed: Reconstructed face image

        Returns:
            Dictionary with quality metrics
        """
        original = np.asarray(original).flatten()
        reconstructed = np.asarray(reconstructed).flatten()

        if original.shape != reconstructed.shape:
            raise ValueError("Original and reconstructed must have same shape")

        # Mean Squared Error
        mse = np.mean((original - reconstructed) ** 2)

        # Peak Signal-to-Noise Ratio
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0  # Assuming 8-bit images
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        # Structural Similarity Index (simplified version)
        mean_orig = np.mean(original)
        mean_recon = np.mean(reconstructed)

        var_orig = np.var(original)
        var_recon = np.var(reconstructed)
        cov_orig_recon = np.cov(original, reconstructed)[0, 1]

        numerator = 2 * mean_orig * mean_recon + 1e-8
        denominator = mean_orig**2 + mean_recon**2 + 1e-8

        luminance = numerator / denominator

        numerator = 2 * cov_orig_recon + 1e-8
        denominator = var_orig + var_recon + 1e-8

        contrast = numerator / denominator

        ssim = luminance * contrast

        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'rmse': float(np.sqrt(mse))
        }


class OptimizationUtils:
    """
    Utility functions for optimization and performance improvements.

    This class provides tools for computational optimization, memory management,
    and performance analysis for the face recognition system.
    """

    @staticmethod
    def memory_efficient_matrix_multiply(A: np.ndarray,
                                      B: np.ndarray,
                                      chunk_size: Optional[int] = None) -> np.ndarray:
        """
        Perform memory-efficient matrix multiplication for large matrices.

        Args:
            A: First matrix
            B: Second matrix
            chunk_size: Size of chunks for processing (None for auto)

        Returns:
            Result of A @ B
        """
        A = np.asarray(A)
        B = np.asarray(B)

        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Cannot multiply: {A.shape} @ {B.shape}")

        # For small matrices, use standard multiplication
        if A.size < 1e6 and B.size < 1e6:
            return A @ B

        # Auto-determine chunk size if not provided
        if chunk_size is None:
            # Use available memory estimate
            chunk_size = max(100, min(1000, A.shape[0] // 10))

        n_rows = A.shape[0]
        n_cols = B.shape[1]
        result = np.zeros((n_rows, n_cols))

        for i in range(0, n_rows, chunk_size):
            end_idx = min(i + chunk_size, n_rows)
            result[i:end_idx] = A[i:end_idx] @ B

        return result

    @staticmethod
    def estimate_optimal_components(data_shape: Tuple[int, int],
                                  variance_threshold: float = 0.95) -> int:
        """
        Estimate optimal number of PCA components based on data properties.

        Args:
            data_shape: Shape of data (n_samples, n_features)
            variance_threshold: Desired variance to explain

        Returns:
            Estimated optimal number of components
        """
        n_samples, n_features = data_shape

        # Rule of thumb: minimum of samples and features
        max_possible = min(n_samples, n_features)

        # For face recognition, typically 50-200 components are sufficient
        # for AT&T dataset (40 subjects, 92x112 images)
        if n_features == 10304:  # AT&T face dataset size
            # Empirical estimates for AT&T dataset
            if variance_threshold >= 0.95:
                return min(150, max_possible)
            elif variance_threshold >= 0.90:
                return min(100, max_possible)
            else:
                return min(50, max_possible)

        # General heuristic: log2 of features for high-dimensional data
        heuristic = int(np.log2(n_features) * variance_threshold * 10)

        return min(heuristic, max_possible)


def main():
    """
    Demonstrate the mathematical utilities functionality.
    """
    print("MATHEMATICAL UTILITIES DEMONSTRATION")
    print("=" * 50)

    # Test vectors
    x1 = np.array([1.0, 2.0, 3.0, 4.0])
    x2 = np.array([1.5, 2.5, 2.8, 4.2])

    print(f"\nTest vectors:")
    print(f"x1: {x1}")
    print(f"x2: {x2}")

    # Distance metrics
    print("\n--- Distance Metrics ---")
    euclidean_dist = DistanceMetrics.euclidean_distance(x1, x2)
    manhattan_dist = DistanceMetrics.manhattan_distance(x1, x2)
    cosine_sim = DistanceMetrics.cosine_similarity(x1, x2)
    cosine_dist = DistanceMetrics.cosine_distance(x1, x2)

    print(f"Euclidean distance: {euclidean_dist:.4f}")
    print(f"Manhattan distance: {manhattan_dist:.4f}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print(f"Cosine distance: {cosine_dist:.4f}")

    # Normalization
    print("\n--- Normalization ---")
    data = np.random.randn(5, 3)
    l2_norm = NormalizationUtils.l2_normalize(data)
    min_max_norm = NormalizationUtils.min_max_normalize(data)
    z_score_norm = NormalizationUtils.z_score_normalize(data)

    print(f"Original data shape: {data.shape}")
    print(f"L2 normalized data:\n{l2_norm}")
    print(f"L2 norms after normalization: {np.linalg.norm(l2_norm, axis=1)}")

    # Matrix utilities
    print("\n--- Matrix Utilities ---")
    matrix = np.random.randn(3, 3)
    symmetric_matrix = (matrix + matrix.T) / 2

    print(f"Matrix is symmetric: {MatrixUtils.is_symmetric(symmetric_matrix)}")
    print(f"Matrix is positive semi-definite: {MatrixUtils.is_positive_semi_definite(symmetric_matrix)}")
    print(f"Condition number: {MatrixUtils.matrix_condition_number(symmetric_matrix):.4f}")

    # Face recognition utilities
    print("\n--- Face Recognition Utilities ---")
    confidence = FaceRecognitionUtils.calculate_confidence_score(
        distance=0.3, threshold=0.5, metric='euclidean'
    )
    print(f"Confidence score (distance=0.3, threshold=0.5): {confidence:.4f}")

    # Database matching simulation
    query = np.array([0.1, 0.2, 0.3])
    database = np.array([
        [0.1, 0.2, 0.3],  # Exact match
        [0.2, 0.3, 0.4],  # Close match
        [0.9, 0.8, 0.7]   # Far match
    ])

    best_idx, best_dist = FaceRecognitionUtils.find_best_match(
        query, database, distance_metric='euclidean'
    )
    print(f"Best match index: {best_idx}, distance: {best_dist:.4f}")

    # Optimization utilities
    print("\n--- Optimization Utilities ---")
    optimal_comp = OptimizationUtils.estimate_optimal_components(
        data_shape=(240, 10304),  # AT&T dataset like
        variance_threshold=0.95
    )
    print(f"Optimal components for AT&T dataset (95% variance): {optimal_comp}")

    print("\n✅ Mathematical utilities demonstration complete!")


if __name__ == "__main__":
    main()