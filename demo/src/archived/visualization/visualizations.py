"""
Visualization Utilities for PCA Face Recognition

This module provides comprehensive plotting and chart generation utilities
for visualizing PCA concepts, eigenfaces, reconstruction results, and performance metrics.

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
except ImportError:
    from utils.logger import safe_streamlit_execute, log_exception


class VisualizationUtils:
    """
    Comprehensive visualization utilities for PCA face recognition system.

    This class provides static methods for creating various types of visualizations
    including eigenfaces galleries, reconstruction comparisons, performance metrics,
    and educational charts for PCA concepts.
    """

    def __init__(self):
        """Initialize visualization utilities with logging."""
        try:
            self.logger = logging.getLogger('pca_face_recognition.visualization')

            # Set matplotlib and seaborn style for educational clarity
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")

            # Configure matplotlib for better quality
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['legend.fontsize'] = 9
            plt.rcParams['figure.titlesize'] = 14

            self.logger.info("VisualizationUtils initialized successfully")

        except Exception as e:
            log_exception(e, context="VisualizationUtils.__init__")
            raise

    @staticmethod
    def plot_eigenfaces(eigenfaces: np.ndarray, n_faces: int = 16,
                       title: str = "Eigenfaces Gallery") -> plt.Figure:
        """
        Display eigenfaces gallery with component numbers and mathematical annotations.

        Args:
            eigenfaces: Array of eigenfaces (n_components, height, width)
            n_faces: Number of eigenfaces to display
            title: Plot title

        Returns:
            matplotlib Figure object

        Raises:
            ValueError: If eigenfaces array is invalid
        """
        try:
            safe_streamlit_execute(
                lambda: VisualizationUtils._plot_eigenfaces_impl(eigenfaces, n_faces, title),
                "plot_eigenfaces"
            )

        except Exception as e:
            log_exception(e, context="VisualizationUtils.plot_eigenfaces")
            raise

    @staticmethod
    def _plot_eigenfaces_impl(eigenfaces: np.ndarray, n_faces: int, title: str) -> plt.Figure:
        """Implementation of eigenfaces plotting."""
        if len(eigenfaces.shape) != 3:
            raise ValueError("Eigenfaces must be 3D array (n_components, height, width)")

        if n_faces > len(eigenfaces):
            n_faces = len(eigenfaces)

        # Calculate grid dimensions
        cols = min(4, n_faces)
        rows = (n_faces + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        for i in range(n_faces):
            row, col = i // cols, i % cols
            ax = axes[row, col]

            # Display eigenface
            eigenface = eigenfaces[i].reshape((112, 92))  # AT&T dimensions
            im = ax.imshow(eigenface, cmap='gray', aspect='auto')
            ax.set_title(f'Eigenface {i+1}\nComponent PC{i+1}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add variance explanation if eigenvalues are available
            if hasattr(eigenfaces, 'explained_variance_ratio_'):
                variance_ratio = eigenfaces.explained_variance_ratio_[i] * 100
                ax.set_title(f'Eigenface {i+1}\n({variance_ratio:.1f}% variance)', fontsize=9)

        # Hide unused subplots
        for i in range(n_faces, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_reconstruction_comparison(original: np.ndarray,
                                    reconstructed: np.ndarray,
                                    n_samples: int = 8,
                                    component_counts: Optional[List[int]] = None) -> plt.Figure:
        """
        Compare original and reconstructed images side-by-side.

        Args:
            original: Original images array (n_samples, height, width)
            reconstructed: Reconstructed images array (n_samples, height, width)
            n_samples: Number of samples to display
            component_counts: List of component counts used for reconstruction

        Returns:
            matplotlib Figure object
        """
        try:
            return safe_streamlit_execute(
                lambda: VisualizationUtils._plot_reconstruction_comparison_impl(
                    original, reconstructed, n_samples, component_counts
                ),
                "plot_reconstruction_comparison"
            )

        except Exception as e:
            log_exception(e, context="VisualizationUtils.plot_reconstruction_comparison")
            raise

    @staticmethod
    def _plot_reconstruction_comparison_impl(original: np.ndarray, reconstructed: np.ndarray,
                                           n_samples: int, component_counts: Optional[List[int]]) -> plt.Figure:
        """Implementation of reconstruction comparison plotting."""
        n_samples = min(n_samples, len(original), len(reconstructed))

        if len(original.shape) != 3:
            original = original.reshape(-1, 112, 92)
        if len(reconstructed.shape) != 3:
            reconstructed = reconstructed.reshape(-1, 112, 92)

        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Original vs. Reconstructed Images', fontsize=16, fontweight='bold')

        for i in range(n_samples):
            # Original image
            axes[i, 0].imshow(original[i], cmap='gray', aspect='auto')
            axes[i, 0].set_title(f'Original Image {i+1}', fontsize=10)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            # Reconstructed image
            axes[i, 1].imshow(reconstructed[i], cmap='gray', aspect='auto')

            # Calculate MSE for this sample
            mse = np.mean((original[i] - reconstructed[i]) ** 2)

            title = f'Reconstructed {i+1}'
            if component_counts and i < len(component_counts):
                title += f'\n({component_counts[i]} components)'
            title += f'\nMSE: {mse:.2f}'

            axes[i, 1].set_title(title, fontsize=10)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_variance_explained(variance_ratio: np.ndarray,
                              threshold: float = 0.95,
                              show_components: bool = True) -> plt.Figure:
        """
        Plot cumulative variance explained by principal components.

        Args:
            variance_ratio: Array of variance ratios for each component
            threshold: Variance threshold to highlight (default 0.95)
            show_components: Whether to show individual component contributions

        Returns:
            matplotlib Figure object
        """
        try:
            return safe_streamlit_execute(
                lambda: VisualizationUtils._plot_variance_explained_impl(
                    variance_ratio, threshold, show_components
                ),
                "plot_variance_explained"
            )

        except Exception as e:
            log_exception(e, context="VisualizationUtils.plot_variance_explained")
            raise

    @staticmethod
    def _plot_variance_explained_impl(variance_ratio: np.ndarray,
                                     threshold: float, show_components: bool) -> plt.Figure:
        """Implementation of variance explained plotting."""
        cumulative_variance = np.cumsum(variance_ratio)
        n_components = len(variance_ratio)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [1, 1]})

        # Cumulative variance plot
        ax1.plot(range(1, n_components + 1), cumulative_variance * 100,
                'b-', linewidth=2, label='Cumulative Variance')
        ax1.axhline(y=threshold * 100, color='r', linestyle='--',
                   label=f'{threshold * 100:.0f}% Threshold')

        # Find components needed for threshold
        threshold_idx = np.argmax(cumulative_variance >= threshold) + 1
        ax1.axvline(x=threshold_idx, color='g', linestyle='--', alpha=0.7,
                   label=f'{threshold_idx} Components')

        ax1.fill_between(range(1, n_components + 1), 0, cumulative_variance * 100,
                        alpha=0.3, color='blue')

        ax1.set_xlabel('Number of Principal Components')
        ax1.set_ylabel('Cumulative Variance Explained (%)')
        ax1.set_title('Cumulative Variance Explained by Principal Components')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, min(n_components, threshold_idx * 2))
        ax1.set_ylim(0, 105)

        # Individual component variance plot
        if show_components:
            ax2.bar(range(1, min(50, n_components) + 1),
                   variance_ratio[:50] * 100, alpha=0.7, color='skyblue')
            ax2.set_xlabel('Principal Component Number')
            ax2.set_ylabel('Individual Variance Explained (%)')
            ax2.set_title('Individual Component Variance Contribution')
            ax2.grid(True, alpha=0.3)

            # Highlight threshold components
            if threshold_idx <= 50:
                ax2.axvline(x=threshold_idx, color='g', linestyle='--', alpha=0.7)
                ax2.text(threshold_idx + 1, max(variance_ratio[:50] * 100) * 0.9,
                        f'{threshold_idx} components\nfor {threshold * 100:.0f}% variance',
                        fontsize=9, ha='left')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            normalize: bool = False) -> plt.Figure:
        """
        Plot confusion matrix for classification results.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names
            normalize: Whether to normalize the confusion matrix

        Returns:
            matplotlib Figure object
        """
        try:
            return safe_streamlit_execute(
                lambda: VisualizationUtils._plot_confusion_matrix_impl(
                    y_true, y_pred, class_names, normalize
                ),
                "plot_confusion_matrix"
            )

        except Exception as e:
            log_exception(e, context="VisualizationUtils.plot_confusion_matrix")
            raise

    @staticmethod
    def _plot_confusion_matrix_impl(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: Optional[List[str]], normalize: bool) -> plt.Figure:
        """Implementation of confusion matrix plotting."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        if class_names is None:
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            class_names = [f'Subject {label}' for label in unique_labels]

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_accuracy_vs_components(component_counts: List[int],
                                   accuracies: List[float],
                                   plot_type: str = 'line') -> plt.Figure:
        """
        Plot accuracy versus number of components.

        Args:
            component_counts: List of component counts
            accuracies: List of corresponding accuracies
            plot_type: Type of plot ('line' or 'scatter')

        Returns:
            matplotlib Figure object
        """
        try:
            return safe_streamlit_execute(
                lambda: VisualizationUtils._plot_accuracy_vs_components_impl(
                    component_counts, accuracies, plot_type
                ),
                "plot_accuracy_vs_components"
            )

        except Exception as e:
            log_exception(e, context="VisualizationUtils.plot_accuracy_vs_components")
            raise

    @staticmethod
    def _plot_accuracy_vs_components_impl(component_counts: List[int],
                                         accuracies: List[float], plot_type: str) -> plt.Figure:
        """Implementation of accuracy vs components plotting."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == 'scatter':
            ax.scatter(component_counts, accuracies, s=100, alpha=0.7, color='blue')
        else:
            ax.plot(component_counts, accuracies, 'bo-', linewidth=2, markersize=8)

        ax.fill_between(component_counts, 0, accuracies, alpha=0.3, color='blue')

        ax.set_xlabel('Number of Principal Components', fontsize=12)
        ax.set_ylabel('Recognition Accuracy (%)', fontsize=12)
        ax.set_title('Recognition Accuracy vs. Number of Components', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add best accuracy annotation
        best_idx = np.argmax(accuracies)
        best_components = component_counts[best_idx]
        best_accuracy = accuracies[best_idx]

        ax.annotate(f'Best: {best_accuracy:.1f}%\nat {best_components} components',
                   xy=(best_components, best_accuracy),
                   xytext=(best_components + 10, best_accuracy - 5),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=10)

        ax.set_ylim(0, 105)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_mean_face(mean_face: np.ndarray, title: str = "Mean Face") -> plt.Figure:
        """
        Plot the computed mean face with mathematical annotations.

        Args:
            mean_face: Mean face array (height, width)
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        try:
            return safe_streamlit_execute(
                lambda: VisualizationUtils._plot_mean_face_impl(mean_face, title),
                "plot_mean_face"
            )

        except Exception as e:
            log_exception(e, context="VisualizationUtils.plot_mean_face")
            raise

    @staticmethod
    def _plot_mean_face_impl(mean_face: np.ndarray, title: str) -> plt.Figure:
        """Implementation of mean face plotting."""
        if len(mean_face.shape) == 1:
            mean_face = mean_face.reshape(112, 92)  # AT&T dimensions

        fig, ax = plt.subplots(figsize=(6, 8))

        im = ax.imshow(mean_face, cmap='gray', aspect='auto')
        ax.set_title(f'{title}\nμ = (1/n) Σᵢ₌₁ⁿ xᵢ', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pixel Intensity', rotation=270, labelpad=15)

        # Add mathematical annotation
        ax.text(0.02, 0.98, f'Mean Face\nShape: {mean_face.shape}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_interactive_variance_plot(variance_ratio: np.ndarray) -> go.Figure:
        """
        Create interactive Plotly variance explained plot.

        Args:
            variance_ratio: Array of variance ratios for each component

        Returns:
            Plotly Figure object
        """
        try:
            return safe_streamlit_execute(
                lambda: VisualizationUtils._create_interactive_variance_plot_impl(variance_ratio),
                "create_interactive_variance_plot"
            )

        except Exception as e:
            log_exception(e, context="VisualizationUtils.create_interactive_variance_plot")
            raise

    @staticmethod
    def _create_interactive_variance_plot_impl(variance_ratio: np.ndarray) -> go.Figure:
        """Implementation of interactive variance plot."""
        cumulative_variance = np.cumsum(variance_ratio)
        n_components = len(variance_ratio)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Variance Explained', 'Individual Component Variance'),
            vertical_spacing=0.1
        )

        # Cumulative variance trace
        fig.add_trace(
            go.Scatter(
                x=list(range(1, n_components + 1)),
                y=cumulative_variance * 100,
                mode='lines',
                name='Cumulative Variance',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Component %{x}</b><br>Variance: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Individual component variance trace
        fig.add_trace(
            go.Bar(
                x=list(range(1, min(50, n_components) + 1)),
                y=variance_ratio[:50] * 100,
                name='Individual Variance',
                marker_color='lightblue',
                hovertemplate='<b>Component %{x}</b><br>Variance: %{y:.3f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # Add 95% threshold line
        fig.add_hline(y=95, line_dash="dash", line_color="red",
                    annotation_text="95% Threshold", row=1, col=1)

        fig.update_xaxes(title_text="Number of Components", row=1, col=1)
        fig.update_xaxes(title_text="Principal Component", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Variance (%)", row=1, col=1)
        fig.update_yaxes(title_text="Variance (%)", row=2, col=1)

        fig.update_layout(
            title="Interactive Variance Explained Analysis",
            height=700,
            showlegend=True
        )

        return fig

    @staticmethod
    def fig_to_base64(fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string for Streamlit display.

        Args:
            fig: matplotlib Figure object

        Returns:
            Base64 encoded image string
        """
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            return image_base64

        except Exception as e:
            log_exception(e, context="VisualizationUtils.fig_to_base64")
            return ""

    @staticmethod
    def create_pca_mathematical_workflow_diagram() -> plt.Figure:
        """
        Create a mathematical workflow diagram for PCA process.

        Returns:
            matplotlib Figure object showing PCA mathematical workflow
        """
        try:
            return safe_streamlit_execute(
                VisualizationUtils._create_pca_mathematical_workflow_diagram_impl,
                "create_pca_mathematical_workflow_diagram"
            )

        except Exception as e:
            log_exception(e, context="VisualizationUtils.create_pca_mathematical_workflow_diagram")
            raise

    @staticmethod
    def _create_pca_mathematical_workflow_diagram_impl() -> plt.Figure:
        """Implementation of PCA mathematical workflow diagram."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create workflow boxes and arrows
        workflow_steps = [
            "Raw Face Images\nX ∈ ℝ^(n×10304)",
            "Mean Centering\nX_centered = X - μ",
            "Covariance Matrix\nC = cov(X_centered^T)",
            "Eigenvalue Decomposition\nCw = λw",
            "Select Components\nTop k eigenvectors",
            "Projection & Reconstruction\nx_rec = μ + Σ(x·w_i)·w_i"
        ]

        positions = [(0.5, 0.9), (0.5, 0.75), (0.5, 0.6), (0.5, 0.45), (0.5, 0.3), (0.5, 0.15)]

        for i, (step, pos) in enumerate(zip(workflow_steps, positions)):
            # Draw box
            rect = plt.Rectangle((pos[0] - 0.2, pos[1] - 0.06), 0.4, 0.12,
                               facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            # Add text
            ax.text(pos[0], pos[1], step, ha='center', va='center',
                   fontsize=9, fontweight='bold', wrap=True)

            # Add arrow
            if i < len(workflow_steps) - 1:
                next_pos = positions[i + 1]
                ax.arrow(pos[0], pos[1] - 0.06, 0, next_pos[1] - pos[1] + 0.06,
                        head_width=0.02, head_length=0.02, fc='black', ec='black')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('PCA Mathematical Workflow for Face Recognition',
                   fontsize=14, fontweight='bold', y=0.98)

        return fig


def main():
    """Test the VisualizationUtils class."""
    import sys

    try:
        viz_utils = VisualizationUtils()

        # Test with dummy data
        print("Testing VisualizationUtils...")

        # Test eigenfaces plotting
        eigenfaces = np.random.rand(16, 112, 92)
        fig = VisualizationUtils.plot_eigenfaces(eigenfaces, n_faces=8)
        print("✓ Eigenfaces plotting test passed")

        # Test variance plotting
        variance_ratio = np.random.dirichlet(np.ones(50))
        fig = VisualizationUtils.plot_variance_explained(variance_ratio)
        print("✓ Variance explained plotting test passed")

        # Test reconstruction comparison
        original = np.random.rand(5, 112, 92)
        reconstructed = original + np.random.normal(0, 0.1, original.shape)
        fig = VisualizationUtils.plot_reconstruction_comparison(original, reconstructed)
        print("✓ Reconstruction comparison test passed")

        print("All visualization tests passed!")

    except Exception as e:
        print(f"VisualizationUtils test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()