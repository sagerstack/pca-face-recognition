"""
Advanced PCA Techniques for Face Recognition
Includes: Kernel PCA, Incremental PCA, and Optimizations
Author: AI Math Master's Project Team
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
import time
from typing import Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class KernelPCA:
    """
    Kernel PCA for Non-linear Dimensionality Reduction
    
    Mathematical Foundation:
    Instead of computing φ(x) explicitly, we use the kernel trick:
    k(x_i, x_j) = φ(x_i)^T φ(x_j)
    
    Common kernels:
    - Linear: k(x, y) = x^T y
    - RBF: k(x, y) = exp(-γ ||x - y||²)
    - Polynomial: k(x, y) = (γ x^T y + r)^d
    """
    
    def __init__(self, n_components: int = 50, kernel: str = 'rbf', 
                 gamma: float = None, degree: int = 3, coef0: float = 1):
        """
        Initialize Kernel PCA
        
        Args:
            n_components: Number of components to keep
            kernel: Type of kernel ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: Kernel coefficient for rbf, poly and sigmoid
            degree: Degree for polynomial kernel
            coef0: Independent term in polynomial and sigmoid kernels
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        self.X_fit_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.K_centered_ = None
        
    def _compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Compute kernel matrix K where K_ij = k(x_i, x_j)
        
        Mathematical formulations for different kernels:
        - Linear: K_ij = x_i^T x_j
        - RBF: K_ij = exp(-γ ||x_i - x_j||²)
        - Polynomial: K_ij = (γ x_i^T x_j + r)^d
        """
        if Y is None:
            Y = X
            
        if self.kernel == 'linear':
            K = X @ Y.T
            
        elif self.kernel == 'rbf':
            # Set default gamma if not specified
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]
            
            # Compute pairwise squared Euclidean distances
            if Y is X:
                # More efficient for symmetric case
                dists = squareform(pdist(X, 'sqeuclidean'))
            else:
                # For non-symmetric case
                XX = np.sum(X**2, axis=1)[:, np.newaxis]
                YY = np.sum(Y**2, axis=1)[np.newaxis, :]
                XY = X @ Y.T
                dists = XX + YY - 2*XY
            
            K = np.exp(-self.gamma * dists)
            
        elif self.kernel == 'poly':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]
            K = (self.gamma * (X @ Y.T) + self.coef0) ** self.degree
            
        elif self.kernel == 'sigmoid':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]
            K = np.tanh(self.gamma * (X @ Y.T) + self.coef0)
            
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
            
        return K
    
    def _center_kernel_matrix(self, K: np.ndarray, K_fit: np.ndarray = None) -> np.ndarray:
        """
        Center kernel matrix in feature space
        
        Mathematical formula:
        K_centered = K - 1_n K - K 1_n + 1_n K 1_n
        where 1_n is the n×n matrix with all entries equal to 1/n
        """
        n = K.shape[0]
        
        if K_fit is None:
            # Training phase: center the kernel matrix
            K_fit = K
            n_fit = n
        else:
            n_fit = K_fit.shape[0]
        
        # Compute column means of K_fit
        one_n_fit = np.ones((n_fit, n_fit)) / n_fit
        K_fit_col_means = np.mean(K_fit, axis=0)
        K_col_means = np.mean(K, axis=0)
        K_fit_mean = np.mean(K_fit)
        
        # Center the kernel matrix
        K_centered = K - K_col_means[np.newaxis, :] - K_fit_col_means[:, np.newaxis] + K_fit_mean
        
        return K_centered
    
    def fit(self, X: np.ndarray) -> 'KernelPCA':
        """
        Fit Kernel PCA model
        
        Steps:
        1. Compute kernel matrix K
        2. Center K in feature space
        3. Eigendecomposition of centered K
        4. Select top k eigenvectors
        """
        self.X_fit_ = X
        n_samples = X.shape[0]
        
        # Step 1: Compute kernel matrix
        print(f"Computing {self.kernel} kernel matrix...")
        K = self._compute_kernel_matrix(X)
        
        # Step 2: Center kernel matrix
        print("Centering kernel matrix in feature space...")
        self.K_centered_ = self._center_kernel_matrix(K)
        
        # Step 3: Eigendecomposition
        print("Performing eigendecomposition...")
        # For numerical stability, ensure matrix is symmetric
        self.K_centered_ = (self.K_centered_ + self.K_centered_.T) / 2
        
        eigenvalues, eigenvectors = eigh(self.K_centered_)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 4: Select top k components and normalize
        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.eigenvectors_ = eigenvectors[:, :self.n_components]
        
        # Normalize eigenvectors
        for i in range(self.n_components):
            self.eigenvectors_[:, i] /= np.sqrt(self.eigenvalues_[i])
        
        print(f"Kernel PCA fitting complete with {self.n_components} components")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted Kernel PCA model
        
        Mathematical operation:
        y = K(x, X_train) @ α
        where α are the normalized eigenvectors
        """
        # Compute kernel matrix between X and training data
        K = self._compute_kernel_matrix(X, self.X_fit_)
        
        # Center the kernel matrix
        K_centered = self._center_kernel_matrix(K, self._compute_kernel_matrix(self.X_fit_))
        
        # Project onto kernel principal components
        X_transformed = K_centered @ self.eigenvectors_
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)


class IncrementalPCA:
    """
    Incremental PCA for Large-scale Data
    
    Algorithm: Processes data in mini-batches to handle datasets that don't fit in memory
    Based on: "Incremental Learning for Robust Visual Tracking" by Ross et al.
    
    Mathematical Foundation:
    Updates the eigenspace incrementally using:
    1. Project new data onto current eigenspace
    2. Compute residual
    3. Update eigenspace using QR decomposition
    """
    
    def __init__(self, n_components: int = 50, batch_size: int = 100):
        """
        Initialize Incremental PCA
        
        Args:
            n_components: Number of components to keep
            batch_size: Size of mini-batches for processing
        """
        self.n_components = n_components
        self.batch_size = batch_size
        
        self.mean_ = None
        self.components_ = None
        self.n_samples_seen_ = 0
        self.var_ = None
        self.singular_values_ = None
        
    def partial_fit(self, X: np.ndarray) -> 'IncrementalPCA':
        """
        Incrementally fit the model with a batch of samples
        
        Mathematical update rules:
        1. Update mean: μ_new = (n*μ_old + m*μ_batch) / (n + m)
        2. Update components using incremental SVD
        """
        n_samples, n_features = X.shape
        
        if self.mean_ is None:
            # First batch: initialize
            self.mean_ = np.zeros(n_features)
            self.components_ = None
            
        # Update mean incrementally
        col_batch_mean = np.mean(X, axis=0)
        n_total = self.n_samples_seen_ + n_samples
        
        # Incremental mean update formula
        col_mean = (self.n_samples_seen_ * self.mean_ + 
                   n_samples * col_batch_mean) / n_total
        
        # Center the batch
        X_centered = X - col_batch_mean
        
        # Update mean correction for existing components
        if self.components_ is not None:
            mean_correction = np.sqrt(
                (self.n_samples_seen_ * n_samples) / n_total
            ) * (self.mean_ - col_batch_mean)
            
            # Project mean correction onto existing components
            mean_proj = mean_correction @ self.components_.T
            
        # Update components
        if self.components_ is None:
            # First batch: use standard SVD
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            U = U[:, :self.n_components]
            S = S[:self.n_components]
            Vt = Vt[:self.n_components]
            
            self.singular_values_ = S
            self.components_ = Vt
            
        else:
            # Incremental update using the algorithm from Ross et al.
            # Project batch onto existing components
            X_proj = X_centered @ self.components_.T
            
            # Compute residual
            X_residual = X_centered - X_proj @ self.components_
            
            # QR decomposition of residual
            Q, R = np.linalg.qr(X_residual, mode='reduced')
            
            # Construct augmented matrix
            if self.components_ is not None:
                B = np.block([
                    [np.diag(self.singular_values_), X_proj.T],
                    [np.zeros((R.shape[0], self.singular_values_.shape[0])), R]
                ])
                
                # Add mean correction
                B[:self.n_components, :self.n_components] += np.outer(
                    self.singular_values_, mean_proj
                )
            else:
                B = R
                
            # SVD of augmented matrix
            U_B, S_B, Vt_B = np.linalg.svd(B, full_matrices=False)
            
            # Update components
            U_B = U_B[:, :self.n_components]
            S_B = S_B[:self.n_components]
            Vt_B = Vt_B[:self.n_components]
            
            # Rotate components
            if self.components_ is not None:
                components_new = U_B[:self.components_.shape[0]] @ self.components_
                
                if R.shape[0] > 0:
                    Q_components = U_B[self.components_.shape[0]:] @ Q.T
                    components_new = np.vstack([components_new, Q_components])
                    
                self.components_ = Vt_B @ components_new
            else:
                self.components_ = Vt_B @ Q.T
                
            self.singular_values_ = S_B
        
        # Update statistics
        self.n_samples_seen_ = n_total
        self.mean_ = col_mean
        
        # Ensure components are normalized
        self.components_ = self.components_[:self.n_components]
        
        return self
    
    def fit(self, X: np.ndarray) -> 'IncrementalPCA':
        """
        Fit the model by processing data in batches
        """
        n_samples = X.shape[0]
        
        # Process in batches
        for batch_start in range(0, n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_samples)
            self.partial_fit(X[batch_start:batch_end])
            
            if (batch_end // self.batch_size) % 10 == 0:
                print(f"Processed {batch_end}/{n_samples} samples...")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the fitted model"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)


class OptimizedPCA:
    """
    Optimized PCA Implementation with Various Computational Tricks
    
    Optimizations:
    1. Randomized SVD for large datasets
    2. Efficient covariance computation
    3. Memory-efficient operations
    4. Parallel processing support
    """
    
    def __init__(self, n_components: int = 50, method: str = 'auto',
                 random_state: int = None):
        """
        Initialize Optimized PCA
        
        Args:
            n_components: Number of components
            method: 'auto', 'full', 'randomized', or 'incremental'
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def _randomized_svd(self, X: np.ndarray, n_components: int, 
                       n_oversamples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomized SVD using the algorithm from Halko et al. 2011
        
        Mathematical foundation:
        1. Generate random projection matrix Ω
        2. Form Y = X @ Ω
        3. Compute QR decomposition: Y = QR
        4. Form B = Q^T @ X
        5. Compute SVD of small matrix B
        """
        n_samples, n_features = X.shape
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Step 1: Random projection matrix
        n_random = n_components + n_oversamples
        omega = np.random.randn(n_features, n_random)
        
        # Step 2: Form Y = X @ Ω
        Y = X @ omega
        
        # Step 3: QR decomposition
        Q, _ = np.linalg.qr(Y, mode='reduced')
        
        # Power iteration for improved accuracy (optional)
        for _ in range(2):
            Q, _ = np.linalg.qr(X.T @ Q, mode='reduced')
            Q, _ = np.linalg.qr(X @ Q, mode='reduced')
        
        # Step 4: Form B = Q^T @ X
        B = Q.T @ X
        
        # Step 5: SVD of small matrix B
        U_small, S, Vt = np.linalg.svd(B, full_matrices=False)
        
        # Recover U for original matrix
        U = Q @ U_small
        
        # Return only requested components
        return U[:, :n_components], S[:n_components], Vt[:n_components]
    
    def _select_method(self, n_samples: int, n_features: int) -> str:
        """
        Automatically select the best method based on data size
        """
        if self.method != 'auto':
            return self.method
        
        # Heuristics for method selection
        if n_samples < 500 and n_features < 500:
            return 'full'
        elif max(n_samples, n_features) > 10000:
            return 'randomized'
        elif n_samples > 5000:
            return 'incremental'
        else:
            return 'full'
    
    def fit(self, X: np.ndarray) -> 'OptimizedPCA':
        """
        Fit PCA model using optimized method
        """
        n_samples, n_features = X.shape
        
        # Select method
        method = self._select_method(n_samples, n_features)
        print(f"Using {method} method for PCA computation...")
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        if method == 'full':
            # Standard SVD
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            U = U[:, :self.n_components]
            S = S[:self.n_components]
            Vt = Vt[:self.n_components]
            
        elif method == 'randomized':
            # Randomized SVD
            U, S, Vt = self._randomized_svd(X_centered, self.n_components)
            
        elif method == 'incremental':
            # Use incremental PCA
            inc_pca = IncrementalPCA(n_components=self.n_components)
            inc_pca.fit(X)
            self.components_ = inc_pca.components_
            self.explained_variance_ = inc_pca.singular_values_ ** 2 / (n_samples - 1)
            return self
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store results
        self.components_ = Vt
        self.explained_variance_ = S ** 2 / (n_samples - 1)
        
        # Calculate explained variance ratio
        total_var = np.sum(X_centered.var(axis=0))
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform"""
        return X_transformed @ self.components_ + self.mean_


class RobustPCA:
    """
    Robust PCA using RPCA (Robust Principal Component Analysis)
    
    Mathematical Model:
    Decomposes matrix X into low-rank L and sparse S components:
    X = L + S
    
    Solved via Principal Component Pursuit:
    minimize ||L||_* + λ||S||_1
    subject to X = L + S
    
    where ||·||_* is nuclear norm and ||·||_1 is L1 norm
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-7,
                 lambda_: float = None):
        """
        Initialize Robust PCA
        
        Args:
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
            lambda_: Regularization parameter (auto-computed if None)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_ = lambda_
        
        self.L_ = None  # Low-rank component
        self.S_ = None  # Sparse component
        
    def _soft_threshold(self, X: np.ndarray, tau: float) -> np.ndarray:
        """
        Soft thresholding operator
        S_τ(x) = sign(x) * max(|x| - τ, 0)
        """
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)
    
    def _singular_value_threshold(self, X: np.ndarray, tau: float) -> np.ndarray:
        """
        Singular value thresholding operator
        D_τ(X) = U * S_τ(Σ) * V^T
        """
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        S_thresh = self._soft_threshold(S, tau)
        return U @ np.diag(S_thresh) @ Vt
    
    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose X into low-rank and sparse components
        
        Uses Alternating Direction Method of Multipliers (ADMM)
        """
        n_samples, n_features = X.shape
        
        # Set lambda if not provided
        if self.lambda_ is None:
            self.lambda_ = 1.0 / np.sqrt(max(n_samples, n_features))
        
        print(f"Running Robust PCA with λ={self.lambda_:.4f}")
        
        # Initialize
        S = np.zeros_like(X)
        Y = np.zeros_like(X)
        mu = 1.25 / np.linalg.norm(X, 2)  # Step size
        mu_bar = mu * 1e7
        rho = 1.5  # Step size growth
        
        for iteration in range(self.max_iter):
            # Update L (low-rank component)
            L = self._singular_value_threshold(X - S + Y/mu, 1/mu)
            
            # Update S (sparse component)
            S = self._soft_threshold(X - L + Y/mu, self.lambda_/mu)
            
            # Update dual variable Y
            Z = X - L - S
            Y = Y + mu * Z
            
            # Update mu
            mu = min(mu * rho, mu_bar)
            
            # Check convergence
            err = np.linalg.norm(Z, 'fro') / np.linalg.norm(X, 'fro')
            if err < self.tol:
                print(f"Converged at iteration {iteration} with error {err:.2e}")
                break
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Error: {err:.2e}")
        
        self.L_ = L
        self.S_ = S
        
        # Perform PCA on the low-rank component
        U, S_vals, Vt = np.linalg.svd(L, full_matrices=False)
        
        return L, S


def compare_pca_methods(X: np.ndarray, n_components: int = 50):
    """
    Compare different PCA implementations
    """
    methods = {
        'Standard PCA': OptimizedPCA(n_components=n_components, method='full'),
        'Randomized PCA': OptimizedPCA(n_components=n_components, method='randomized'),
        'Kernel PCA (RBF)': KernelPCA(n_components=n_components, kernel='rbf'),
        'Incremental PCA': IncrementalPCA(n_components=n_components, batch_size=100)
    }
    
    results = {}
    
    for name, pca in methods.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Time the fitting process
        start_time = time.time()
        
        if name == 'Kernel PCA (RBF)':
            # Use subset for kernel PCA due to computational complexity
            X_subset = X[:min(1000, X.shape[0])]
            X_transformed = pca.fit_transform(X_subset)
        else:
            X_transformed = pca.fit_transform(X)
        
        fit_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'fit_time': fit_time,
            'transformed_shape': X_transformed.shape
        }
        
        print(f"Fit time: {fit_time:.3f} seconds")
        print(f"Transformed shape: {X_transformed.shape}")
    
    return results


def visualize_kernel_pca_comparison(X: np.ndarray):
    """
    Visualize the effect of different kernels in Kernel PCA
    """
    # Generate non-linear 2D data for visualization
    np.random.seed(42)
    from sklearn.datasets import make_circles, make_moons
    
    datasets = {
        'Circles': make_circles(n_samples=400, factor=0.3, noise=0.05)[0],
        'Moons': make_moons(n_samples=400, noise=0.05)[0]
    }
    
    kernels = ['linear', 'rbf', 'poly']
    
    fig, axes = plt.subplots(len(datasets), len(kernels) + 1, 
                             figsize=(15, 8))
    
    for i, (data_name, data) in enumerate(datasets.items()):
        # Original data
        axes[i, 0].scatter(data[:, 0], data[:, 1], c=range(len(data)), 
                          cmap='viridis', alpha=0.6)
        axes[i, 0].set_title(f'{data_name} - Original')
        axes[i, 0].set_xlabel('Feature 1')
        axes[i, 0].set_ylabel('Feature 2')
        
        # Apply different kernels
        for j, kernel in enumerate(kernels, 1):
            kpca = KernelPCA(n_components=2, kernel=kernel)
            data_transformed = kpca.fit_transform(data)
            
            axes[i, j].scatter(data_transformed[:, 0], data_transformed[:, 1],
                             c=range(len(data)), cmap='viridis', alpha=0.6)
            axes[i, j].set_title(f'{data_name} - {kernel.upper()} Kernel')
            axes[i, j].set_xlabel('KPC 1')
            axes[i, j].set_ylabel('KPC 2')
    
    plt.suptitle('Kernel PCA with Different Kernels', fontsize=16)
    plt.tight_layout()
    plt.show()


def demonstrate_robust_pca():
    """
    Demonstrate Robust PCA for handling corrupted data
    """
    # Create synthetic data with outliers
    np.random.seed(42)
    n_samples, n_features = 100, 50
    
    # Generate low-rank data
    rank = 10
    U = np.random.randn(n_samples, rank)
    V = np.random.randn(rank, n_features)
    L_true = U @ V
    
    # Add sparse corruptions (outliers)
    S_true = np.zeros((n_samples, n_features))
    n_corruptions = int(0.1 * n_samples * n_features)
    corruption_indices = np.random.choice(n_samples * n_features, 
                                        n_corruptions, replace=False)
    S_true.flat[corruption_indices] = np.random.randn(n_corruptions) * 10
    
    # Observed data
    X = L_true + S_true + np.random.randn(n_samples, n_features) * 0.1
    
    # Apply Robust PCA
    rpca = RobustPCA()
    L_recovered, S_recovered = rpca.fit_transform(X)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original components
    im1 = axes[0, 0].imshow(L_true, aspect='auto', cmap='coolwarm')
    axes[0, 0].set_title('True Low-rank Component')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(S_true, aspect='auto', cmap='coolwarm')
    axes[0, 1].set_title('True Sparse Component')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(X, aspect='auto', cmap='coolwarm')
    axes[0, 2].set_title('Observed Data (Corrupted)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Recovered components
    im4 = axes[1, 0].imshow(L_recovered, aspect='auto', cmap='coolwarm')
    axes[1, 0].set_title('Recovered Low-rank Component')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(S_recovered, aspect='auto', cmap='coolwarm')
    axes[1, 1].set_title('Recovered Sparse Component')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(L_recovered + S_recovered, aspect='auto', cmap='coolwarm')
    axes[1, 2].set_title('Reconstructed Data')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.suptitle('Robust PCA Decomposition', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print recovery errors
    print("\nRobust PCA Recovery Errors:")
    print(f"Low-rank recovery error: {np.linalg.norm(L_true - L_recovered, 'fro'):.4f}")
    print(f"Sparse recovery error: {np.linalg.norm(S_true - S_recovered, 'fro'):.4f}")


def main():
    """
    Main demonstration of advanced PCA techniques
    """
    print("="*60)
    print("ADVANCED PCA TECHNIQUES FOR FACE RECOGNITION")
    print("="*60)
    
    # Generate synthetic face-like data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 500
    
    # Create data with intrinsic low dimensionality
    intrinsic_dim = 50
    U = np.random.randn(n_samples, intrinsic_dim)
    V = np.random.randn(intrinsic_dim, n_features)
    X = U @ V + np.random.randn(n_samples, n_features) * 0.1
    
    print(f"\nSynthetic data shape: {X.shape}")
    print(f"Intrinsic dimensionality: {intrinsic_dim}")
    
    # 1. Compare different PCA methods
    print("\n" + "="*60)
    print("COMPARING PCA IMPLEMENTATIONS")
    print("="*60)
    results = compare_pca_methods(X, n_components=50)
    
    # 2. Demonstrate Kernel PCA
    print("\n" + "="*60)
    print("KERNEL PCA DEMONSTRATION")
    print("="*60)
    visualize_kernel_pca_comparison(X[:400, :2])
    
    # 3. Demonstrate Robust PCA
    print("\n" + "="*60)
    print("ROBUST PCA FOR OUTLIER HANDLING")
    print("="*60)
    demonstrate_robust_pca()
    
    # 4. Performance comparison plot
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Create performance comparison chart
    method_names = list(results.keys())
    fit_times = [results[m]['fit_time'] for m in method_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(method_names)), fit_times, color='skyblue')
    plt.xlabel('PCA Method')
    plt.ylabel('Fit Time (seconds)')
    plt.title('PCA Methods Performance Comparison')
    plt.xticks(range(len(method_names)), method_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, time in zip(bars, fit_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\nAdvanced PCA demonstration complete!")


if __name__ == "__main__":
    main()
