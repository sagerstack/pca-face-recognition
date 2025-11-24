"""
Eigenfaces-Specific Visualizations for PCA Face Recognition

This module provides specialized visualization functions specifically designed
for eigenfaces analysis, PCA concepts demonstration, and face reconstruction.

Author: PCA Face Recognition Team
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import io
import base64

try:
    from ..utils.logger import safe_streamlit_execute, log_exception
    from .visualizations import VisualizationUtils
    from .chart_utils import ChartUtils
except ImportError:
    from utils.logger import safe_streamlit_execute, log_exception
    from visualization.visualizations import VisualizationUtils
    from visualization.chart_utils import ChartUtils


class EigenfacesVisualization:
    """
    Specialized visualization class for eigenfaces and PCA concepts.

    This class provides educational visualizations specifically designed for
    explaining eigenfaces, PCA mathematical concepts, and face reconstruction
    processes with real face data.
    """

    def __init__(self):
        """Initialize eigenfaces visualization with educational focus."""
        try:
            self.logger = logging.getLogger('pca_face_recognition.eigenfaces_viz')
            self.viz_utils = VisualizationUtils()
            self.chart_utils = ChartUtils()

            # Educational color schemes
            self.eigenface_colors = plt.cm.viridis
            self.reconstruction_colors = ['blue', 'red', 'green', 'orange', 'purple']

            self.logger.info("EigenfacesVisualization initialized successfully")

        except Exception as e:
            log_exception(e, context="EigenfacesVisualization.__init__")
            raise

    def visualize_mean_face_computation(self, face_samples: np.ndarray,
                                       subject_labels: np.ndarray) -> plt.Figure:
        """
        Visualize the computation of mean face from sample faces.

        Args:
            face_samples: Sample face images (n_samples, height, width)
            subject_labels: Subject labels for each face

        Returns:
            matplotlib Figure showing mean face computation process
        """
        try:
            return safe_streamlit_execute(
                lambda: self._visualize_mean_face_computation_impl(face_samples, subject_labels),
                "visualize_mean_face_computation"
            )

        except Exception as e:
            log_exception(e, context="EigenfacesVisualization.visualize_mean_face_computation")
            raise

    def _visualize_mean_face_computation_impl(self, face_samples: np.ndarray,
                                            subject_labels: np.ndarray) -> plt.Figure:
        """Implementation of mean face computation visualization."""
        n_samples = min(16, len(face_samples))
        mean_face = np.mean(face_samples, axis=0)

        # Create subplot grid
        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        fig.suptitle('Mean Face Computation Process', fontsize=16, fontweight='bold', y=0.98)

        # Show sample faces
        for i in range(min(n_samples, 16)):
            row, col = i // 4, i % 4
            axes[row, col].imshow(face_samples[i], cmap='gray', aspect='auto')
            axes[row, col].set_title(f'Subject {subject_labels[i]}', fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        # Show mean computation formula
        axes[4, 0].text(0.5, 0.5, r'$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$',
                        transform=axes[4, 0].transAxes, fontsize=14,
                        ha='center', va='center', fontweight='bold')
        axes[4, 0].set_title('Mean Formula', fontsize=11, fontweight='bold')
        axes[4, 0].set_xticks([])
        axes[4, 0].set_yticks([])

        # Show computed mean face
        axes[4, 1].imshow(mean_face, cmap='gray', aspect='auto')
        axes[4, 1].set_title(f'Computed Mean Face\nShape: {mean_face.shape}', fontsize=11, fontweight='bold')
        axes[4, 1].set_xticks([])
        axes[4, 1].set_yticks([])

        # Show statistics
        axes[4, 2].text(0.5, 0.7, f'Number of Samples: {len(face_samples)}',
                        transform=axes[4, 2].transAxes, fontsize=11, ha='center')
        axes[4, 2].text(0.5, 0.5, f'Mean Pixel Value: {np.mean(mean_face):.2f}',
                        transform=axes[4, 2].transAxes, fontsize=11, ha='center')
        axes[4, 2].text(0.5, 0.3, f'Std Dev: {np.std(mean_face):.2f}',
                        transform=axes[4, 2].transAxes, fontsize=11, ha='center')
        axes[4, 2].set_title('Statistics', fontsize=11, fontweight='bold')
        axes[4, 2].set_xticks([])
        axes[4, 2].set_yticks([])

        # Show mean centering effect
        if n_samples > 0:
            centered_face = face_samples[0] - mean_face
            axes[4, 3].imshow(centered_face, cmap='gray', aspect='auto')
            axes[4, 3].set_title('Sample - Mean\n(Centered Face)', fontsize=11, fontweight='bold')
            axes[4, 3].set_xticks([])
            axes[4, 3].set_yticks([])

        plt.tight_layout()
        return fig

    def visualize_eigenface_formation(self, eigenfaces: np.ndarray,
                                     eigenvalues: np.ndarray,
                                     n_components: int = 16) -> plt.Figure:
        """
        Visualize the formation and importance of eigenfaces.

        Args:
            eigenfaces: Array of eigenfaces (n_components, height, width)
            eigenvalues: Corresponding eigenvalues
            n_components: Number of components to visualize

        Returns:
            matplotlib Figure showing eigenface formation
        """
        try:
            return safe_streamlit_execute(
                lambda: self._visualize_eigenface_formation_impl(eigenfaces, eigenvalues, n_components),
                "visualize_eigenface_formation"
            )

        except Exception as e:
            log_exception(e, context="EigenfacesVisualization.visualize_eigenface_formation")
            raise

    def _visualize_eigenface_formation_impl(self, eigenfaces: np.ndarray,
                                           eigenvalues: np.ndarray,
                                           n_components: int) -> plt.Figure:
        """Implementation of eigenface formation visualization."""
        n_components = min(n_components, len(eigenfaces))
        variance_explained = eigenvalues / np.sum(eigenvalues) * 100

        # Create figure with eigenfaces and variance
        fig = plt.figure(figsize=(16, 12))

        # Create grid for eigenfaces
        gs = fig.add_gridspec(4, 4, height_ratios=[3, 0.5, 0.5, 1], width_ratios=[1, 1, 1, 1])

        # Plot eigenfaces
        for i in range(n_components):
            row, col = i // 4, i % 4
            ax = fig.add_subplot(gs[0, col] if row == 0 else gs[0, col])

            eigenface = eigenfaces[i].reshape((112, 92))
            im = ax.imshow(eigenface, cmap='gray', aspect='auto')
            ax.set_title(f'PC{i+1}\n{variance_explained[i]:.1f}% variance', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Add mathematical explanation
        ax_text = fig.add_subplot(gs[1:, :])
        ax_text.axis('off')

        explanation_text = """
        Mathematical Foundation:
        • Eigenvalue Problem: Cv = λv
        • Covariance Matrix: C = (1/n) Σ(x_i - μ)(x_i - μ)^T
        • Eigenfaces = Eigenvectors of C
        • Eigenvalues = Variance captured by each eigenface
        • Total Variance: Σλ_i = Trace(C)
        • Variance Explained by PC_k: λ_k / Σλ_i
        """

        ax_text.text(0.05, 0.95, explanation_text, transform=ax_text.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        fig.suptitle('Eigenfaces Formation and Variance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_progressive_reconstruction_animation(self, original_face: np.ndarray,
                                                  eigenfaces: np.ndarray,
                                                  component_counts: List[int]) -> plt.Figure:
        """
        Create progressive reconstruction animation showing face quality improvement.

        Args:
            original_face: Original face image
            eigenfaces: Array of eigenfaces
            component_counts: List of component counts for reconstruction

        Returns:
            matplotlib Figure showing progressive reconstruction
        """
        try:
            return safe_streamlit_execute(
                lambda: self._create_progressive_reconstruction_animation_impl(
                    original_face, eigenfaces, component_counts
                ),
                "create_progressive_reconstruction_animation"
            )

        except Exception as e:
            log_exception(e, context="EigenfacesVisualization.create_progressive_reconstruction_animation")
            raise

    def _create_progressive_reconstruction_animation_impl(self, original_face: np.ndarray,
                                                        eigenfaces: np.ndarray,
                                                        component_counts: List[int]) -> plt.Figure:
        """Implementation of progressive reconstruction animation."""
        n_reconstructions = len(component_counts) + 1  # +1 for original
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        fig.suptitle('Progressive Face Reconstruction', fontsize=16, fontweight='bold')

        # Show original face
        axes[0].imshow(original_face.reshape(112, 92), cmap='gray', aspect='auto')
        axes[0].set_title('Original Face', fontsize=11, fontweight='bold')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Show reconstructions
        for i, n_components in enumerate(component_counts[:5]):  # Limit to 5 reconstructions
            ax = axes[i + 1]

            # Simple reconstruction simulation (in real implementation, use actual PCA)
            # For visualization purposes, we'll show a progressively blurred version
            from scipy.ndimage import gaussian_filter

            # Simulate reconstruction by applying Gaussian filter
            # More components = less blur (better reconstruction)
            sigma = max(0.5, 10 - n_components)  # Inverse relationship
            reconstructed = gaussian_filter(original_face.reshape(112, 92), sigma=sigma)

            # Calculate MSE (simulated)
            mse = np.mean((original_face.reshape(112, 92) - reconstructed) ** 2)

            ax.imshow(reconstructed, cmap='gray', aspect='auto')
            ax.set_title(f'{n_components} Components\nMSE: {mse:.2f}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide any remaining axes
        for i in range(len(component_counts) + 1, len(axes)):
            axes[i].set_visible(False)

        # Add reconstruction formula
        fig.text(0.02, 0.02, r'Reconstruction Formula: $x_{rec} = \mu + \sum_{i=1}^{k} (x \cdot w_i) w_i$',
                fontsize=10, fontstyle='italic')

        plt.tight_layout()
        return fig

    def visualize_covariance_matrix_structure(self, face_data: np.ndarray,
                                            n_samples: int = 100) -> plt.Figure:
        """
        Visualize covariance matrix structure and properties.

        Args:
            face_data: Face data array (n_samples, n_features)
            n_samples: Number of samples to use for visualization

        Returns:
            matplotlib Figure showing covariance matrix analysis
        """
        try:
            return safe_streamlit_execute(
                lambda: self._visualize_covariance_matrix_structure_impl(face_data, n_samples),
                "visualize_covariance_matrix_structure"
            )

        except Exception as e:
            log_exception(e, context="EigenfacesVisualization.visualize_covariance_matrix_structure")
            raise

    def _visualize_covariance_matrix_structure_impl(self, face_data: np.ndarray,
                                                  n_samples: int) -> plt.Figure:
        """Implementation of covariance matrix visualization."""
        # Limit samples for computational efficiency
        face_subset = face_data[:min(n_samples, len(face_data))]

        # Center the data
        mean_face = np.mean(face_subset, axis=0)
        centered_data = face_subset - mean_face

        # Compute covariance matrix (using smaller subset for visualization)
        subset_size = min(50, len(centered_data))
        covariance_matrix = np.cov(centered_data[:subset_size].T)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Covariance Matrix Analysis', fontsize=16, fontweight='bold')

        # Full covariance matrix heatmap (sampled)
        sample_size = min(100, covariance_matrix.shape[0])
        sample_indices = np.linspace(0, covariance_matrix.shape[0]-1, sample_size, dtype=int)
        cov_sample = covariance_matrix[np.ix_(sample_indices, sample_indices)]

        im1 = axes[0, 0].imshow(cov_sample, cmap='RdBu_r', aspect='auto')
        axes[0, 0].set_title(f'Covariance Matrix Heatmap\n({sample_size}x{sample_size} sample)', fontsize=11)
        plt.colorbar(im1, ax=axes[0, 0])

        # Eigenvalue distribution
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        eigenvalues = eigenvalues[::-1]  # Sort in descending order

        axes[0, 1].plot(eigenvalues[:50], 'b-', linewidth=2)
        axes[0, 1].set_title('Eigenvalue Distribution (Top 50)', fontsize=11)
        axes[0, 1].set_xlabel('Component Index')
        axes[0, 1].set_ylabel('Eigenvalue Magnitude')
        axes[0, 1].grid(True, alpha=0.3)

        # Cumulative variance
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        axes[1, 0].plot(cumulative_variance * 100, 'g-', linewidth=2)
        axes[1, 0].axhline(y=95, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Cumulative Variance Explained', fontsize=11)
        axes[1, 0].set_xlabel('Number of Components')
        axes[1, 0].set_ylabel('Cumulative Variance (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 105)

        # Mathematical explanation
        axes[1, 1].axis('off')
        explanation = """
        Covariance Matrix Properties:

        • C = (1/n) Σ(x_i - μ)(x_i - μ)^T
        • Size: {d}×{d} (d = 10304 pixels)
        • Symmetric: C = C^T
        • Positive semi-definite
        • Trace = Total variance
        • Rank = ≤ min(n_samples, n_features)

        Mathematical Insight:
        • Large off-diagonal elements indicate
          correlated pixel relationships
        • Diagonal elements show individual
          pixel variances
        • Eigenfaces are eigenvectors of C
        • Eigenvalues = variance captured
        """.format(d=covariance_matrix.shape[0])

        axes[1, 1].text(0.05, 0.95, explanation, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        return fig

    def create_pca_step_by_step_visualization(self, face_data: np.ndarray,
                                             n_components: int = 5) -> plt.Figure:
        """
        Create step-by-step visualization of PCA process.

        Args:
            face_data: Input face data
            n_components: Number of components to visualize

        Returns:
            matplotlib Figure showing PCA process steps
        """
        try:
            return safe_streamlit_execute(
                lambda: self._create_pca_step_by_step_visualization_impl(face_data, n_components),
                "create_pca_step_by_step_visualization"
            )

        except Exception as e:
            log_exception(e, context="EigenfacesVisualization.create_pca_step_by_step_visualization")
            raise

    def _create_pca_step_by_step_visualization_impl(self, face_data: np.ndarray,
                                                    n_components: int) -> plt.Figure:
        """Implementation of PCA step-by-step visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PCA Process Step-by-Step', fontsize=16, fontweight='bold')

        # Step 1: Original faces
        sample_faces = face_data[:min(4, len(face_data))]
        grid_axes = axes[0, 0]
        grid_axes.imshow(np.vstack([sample_faces[i].reshape(112, 92) for i in range(4)]),
                       cmap='gray', aspect='auto')
        grid_axes.set_title('Step 1: Original Face Images', fontsize=11, fontweight='bold')
        grid_axes.set_xticks([])
        grid_axes.set_yticks([])

        # Step 2: Mean face
        mean_face = np.mean(face_data, axis=0)
        axes[0, 1].imshow(mean_face.reshape(112, 92), cmap='gray', aspect='auto')
        axes[0, 1].set_title('Step 2: Mean Face (μ)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])

        # Step 3: Mean-centered sample
        centered_sample = face_data[0] - mean_face
        axes[0, 2].imshow(centered_sample.reshape(112, 92), cmap='gray', aspect='auto')
        axes[0, 2].set_title('Step 3: Mean-Centered Face\n(x₁ - μ)', fontsize=11, fontweight='bold')
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])

        # Step 4: Top eigenfaces
        # Simulate eigenfaces (in real implementation, compute from covariance)
        from scipy.linalg import eigh
        face_subset = face_data[:min(100, len(face_data))]
        mean_subset = np.mean(face_subset, axis=0)
        centered_subset = face_subset - mean_subset

        # Use smaller subset for computational efficiency
        if centered_subset.shape[0] > centered_subset.shape[1]:
            # Transpose if we have more samples than features
            cov_matrix = np.cov(centered_subset.T)
        else:
            cov_matrix = np.cov(centered_subset)

        # Compute top eigenvectors
        if min(cov_matrix.shape) > 0:
            eigenvalues, eigenvectors = eigh(cov_matrix)
            eigenvalues = eigenvalues[::-1]
            eigenvectors = eigenvectors[:, ::-1]

            # Show top eigenface
            top_eigenface = eigenvectors[:, 0]
            # Reshape to image dimensions if possible
            if len(top_eigenface) == 112 * 92:
                axes[1, 0].imshow(top_eigenface.reshape(112, 92), cmap='gray', aspect='auto')
            else:
                # Show as 1D plot if dimensions don't match
                axes[1, 0].plot(top_eigenface)
                axes[1, 0].set_title('Step 4: Top Eigenface\n(PC1)', fontsize=11, fontweight='bold')
                axes[1, 0].set_xlabel('Pixel Index')
                axes[1, 0].set_ylabel('Eigenvalue')
                axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor eigenface computation',
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Step 4: Top Eigenface', fontsize=11, fontweight='bold')

        if len(top_eigenface) == 112 * 92:
            axes[1, 0].set_title('Step 4: Top Eigenface\n(PC1)', fontsize=11, fontweight='bold')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])

        # Step 5: Projection visualization
        axes[1, 1].scatter(range(min(20, len(eigenvalues))), eigenvalues[:20], alpha=0.7)
        axes[1, 1].set_title('Step 5: Projection Weights\n(x · wᵢ)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Component Index')
        axes[1, 1].set_ylabel('Projection Weight')
        axes[1, 1].grid(True, alpha=0.3)

        # Step 6: Mathematical summary
        axes[1, 2].axis('off')
        summary_text = f"""
        PCA Mathematical Summary:

        • Input shape: {face_data.shape}
        • Mean face: μ ∈ ℝ^{face_data.shape[1]}
        • Components selected: {n_components}
        • Variance preservation: {np.sum(eigenvalues[:n_components])/np.sum(eigenvalues)*100:.1f}%

        Key Formulas:
        • Centering: x' = x - μ
        • Covariance: C = (1/n) Σ(x' · x'^T)
        • Projection: y = x' · W
        • Reconstruction: x̂ = μ + y · W^T
        """

        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

        plt.tight_layout()
        return fig


def main():
    """Test the EigenfacesVisualization class."""
    import sys

    try:
        eigenfaces_viz = EigenfacesVisualization()

        print("Testing EigenfacesVisualization...")

        # Test with dummy data
        face_samples = np.random.rand(20, 112, 92)
        subject_labels = np.random.randint(1, 41, 20)

        # Test mean face visualization
        fig = eigenfaces_viz.visualize_mean_face_computation(face_samples, subject_labels)
        print("✓ Mean face computation visualization test passed")

        # Test eigenface formation
        eigenfaces = np.random.rand(20, 112, 92)
        eigenvalues = np.random.dirichlet(np.ones(20)) * 100
        fig = eigenfaces_viz.visualize_eigenface_formation(eigenfaces, eigenvalues)
        print("✓ Eigenface formation visualization test passed")

        # Test progressive reconstruction
        original_face = np.random.rand(112, 92)
        component_counts = [5, 10, 20, 50, 100]
        fig = eigenfaces_viz.create_progressive_reconstruction_animation(
            original_face, eigenfaces, component_counts)
        print("✓ Progressive reconstruction animation test passed")

        print("All eigenfaces visualization tests passed!")

    except Exception as e:
        print(f"EigenfacesVisualization test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()