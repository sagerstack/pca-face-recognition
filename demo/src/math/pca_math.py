from typing import Dict, Tuple

import numpy as np


def compute_pca_svd(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA via economy SVD on centered data.

    Returns:
        mu: mean vector (D,)
        eigvals: eigenvalues sorted desc (D,)
        eigvecs: eigenvectors as columns (D, D)
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    # Economy SVD: Xc = U S Vt, covariance eigvals = (S^2)/(n-1)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / (X.shape[0] - 1)
    eigvecs = Vt.T  # columns are eigenvectors
    return mu, eigvals, eigvecs


def project(x: np.ndarray, mu: np.ndarray, eigvecs: np.ndarray, k: int) -> np.ndarray:
    """Project centered vector onto first k eigenvectors."""
    k = min(k, eigvecs.shape[1])
    return (x - mu) @ eigvecs[:, :k]


def reconstruct(z: np.ndarray, mu: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Reconstruct from PCA coefficients."""
    k = z.shape[-1]
    return z @ eigvecs[:, :k].T + mu


def explained_variance_ratio(eigvals: np.ndarray) -> np.ndarray:
    """Return explained variance ratio."""
    total = eigvals.sum()
    return eigvals / total if total > 0 else np.zeros_like(eigvals)


def build_templates(z: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Mean PCA vector per person."""
    templates: Dict[int, np.ndarray] = {}
    for pid in np.unique(labels):
        mask = labels == pid
        templates[int(pid)] = z[mask].mean(axis=0)
    return templates


def distances_to_templates(z_query: np.ndarray, templates: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Compute L2 distance from query vector to each template."""
    dists: Dict[int, float] = {}
    for pid, tvec in templates.items():
        dists[pid] = float(np.linalg.norm(z_query - tvec))
    return dists
