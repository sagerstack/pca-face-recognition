"""
Chart and Plotting Utilities for PCA Face Recognition

This module provides helper functions for creating and customizing charts
and plots for the PCA face recognition system with educational focus.

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


class ChartUtils:
    """
    Utility functions for creating and customizing charts with educational annotations.

    This class provides static helper methods for common chart operations,
    styling, and educational annotations specific to PCA face recognition.
    """

    def __init__(self):
        """Initialize chart utilities with educational styling."""
        try:
            self.logger = logging.getLogger('pca_face_recognition.chart_utils')

            # Educational color palette
            self.colors = {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'accent': '#F18F01',
                'success': '#C73E1D',
                'warning': '#F4A261',
                'info': '#264653'
            }

            # Chart styling configuration
            self.chart_style = {
                'font.family': 'Arial, sans-serif',
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.spines.top': False,
                'axes.spines.right': False
            }

            # Configure matplotlib
            plt.rcParams.update(self.chart_style)
            sns.set_palette("husl")

            self.logger.info("ChartUtils initialized with educational styling")

        except Exception as e:
            log_exception(e, context="ChartUtils.__init__")
            raise

    @staticmethod
    def setup_matplotlib_style(style: str = 'seaborn-v0_8-whitegrid',
                              dpi: int = 300,
                              figsize: Tuple[int, int] = (10, 6)):
        """
        Setup matplotlib styling for educational clarity.

        Args:
            style: matplotlib style to use
            dpi: figure resolution
            figsize: default figure size
        """
        try:
            plt.style.use(style)
            plt.rcParams['figure.dpi'] = dpi
            plt.rcParams['savefig.dpi'] = dpi
            plt.rcParams['figure.figsize'] = figsize

        except Exception as e:
            log_exception(e, context="ChartUtils.setup_matplotlib_style")
            raise

    @staticmethod
    def add_mathematical_annotation(ax, text: str, xy: Tuple[float, float],
                                   xytext: Optional[Tuple[float, float]] = None,
                                   fontsize: int = 10,
                                   bbox: bool = True):
        """
        Add mathematical annotation to plot with educational styling.

        Args:
            ax: matplotlib axes object
            text: annotation text (can include LaTeX)
            xy: position to point to
            xytext: position of text box
            fontsize: font size for annotation
            bbox: whether to include bounding box
        """
        try:
            if xytext is None:
                xytext = (xy[0] + 0.1, xy[1] + 0.1)

            bbox_props = None
            if bbox:
                bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                alpha=0.8, edgecolor="gray")

            ax.annotate(text, xy=xy, xytext=xytext,
                       fontsize=fontsize, ha='left', va='bottom',
                       bbox=bbox_props, arrowprops=dict(arrowstyle='->', color='red'))

        except Exception as e:
            log_exception(e, context="ChartUtils.add_mathematical_annotation")

    @staticmethod
    def create_subplot_grid(n_plots: int, max_cols: int = 3,
                           figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create subplot grid with optimal layout.

        Args:
            n_plots: number of plots needed
            max_cols: maximum columns per row
            figsize: optional figure size

        Returns:
            Tuple of figure and axes array
        """
        try:
            cols = min(max_cols, n_plots)
            rows = (n_plots + cols - 1) // cols

            if figsize is None:
                figsize = (cols * 4, rows * 3)

            fig, axes = plt.subplots(rows, cols, figsize=figsize)

            # Handle single subplot case
            if n_plots == 1:
                axes = np.array([axes])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)

            return fig, axes

        except Exception as e:
            log_exception(e, context="ChartUtils.create_subplot_grid")
            raise

    @staticmethod
    def format_axis_for_educational(ax, xlabel: str, ylabel: str, title: str,
                                   add_grid: bool = True,
                                   remove_spines: bool = True):
        """
        Format axes for educational clarity.

        Args:
            ax: matplotlib axes object
            xlabel: x-axis label
            ylabel: y-axis label
            title: plot title
            add_grid: whether to add grid
            remove_spines: whether to remove top/right spines
        """
        try:
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

            if add_grid:
                ax.grid(True, alpha=0.3, linestyle='--')

            if remove_spines:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            # Adjust tick parameters
            ax.tick_params(axis='both', which='major', labelsize=10)

        except Exception as e:
            log_exception(e, context="ChartUtils.format_axis_for_educational")

    @staticmethod
    def create_comparison_plot(data_list: List[np.ndarray],
                              labels: List[str],
                              title: str,
                              n_cols: int = 4,
                              cmap: str = 'gray') -> plt.Figure:
        """
        Create side-by-side comparison plot for multiple datasets.

        Args:
            data_list: list of image arrays to compare
            labels: list of labels for each image
            title: plot title
            n_cols: number of columns in subplot grid
            cmap: colormap for images

        Returns:
            matplotlib Figure object
        """
        try:
            n_images = len(data_list)
            fig, axes = ChartUtils.create_subplot_grid(n_images, max_cols=n_cols,
                                                   figsize=(n_cols * 3, (n_images // n_cols + 1) * 3))

            if n_images == 1:
                axes = axes.reshape(1, 1)
            elif len(axes.shape) == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

            for i, (data, label) in enumerate(zip(data_list, labels)):
                row, col = i // n_cols, i % n_cols

                if len(data.shape) == 1:
                    # Assume AT&T dimensions if flattened
                    data = data.reshape(112, 92)

                axes[row, col].imshow(data, cmap=cmap, aspect='auto')
                axes[row, col].set_title(label, fontsize=10)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

            # Hide unused subplots
            for i in range(n_images, len(axes.flat)):
                axes.flat[i].set_visible(False)

            plt.tight_layout()
            return fig

        except Exception as e:
            log_exception(e, context="ChartUtils.create_comparison_plot")
            raise

    @staticmethod
    def add_progress_indicator(ax, current_step: int, total_steps: int,
                               step_labels: Optional[List[str]] = None):
        """
        Add progress indicator to plot.

        Args:
            ax: matplotlib axes object
            current_step: current step number (0-indexed)
            total_steps: total number of steps
            step_labels: optional labels for each step
        """
        try:
            progress = (current_step + 1) / total_steps

            # Create progress bar
            ax.barh(0, progress, height=0.1, color='lightblue', edgecolor='black')
            ax.barh(0, (current_step + 1) / total_steps, height=0.1,
                   color='steelblue', edgecolor='black')

            # Add step markers
            for i in range(total_steps):
                x_pos = (i + 0.5) / total_steps
                ax.plot(x_pos, 0.05, 'ro', markersize=8)

            # Add labels
            if step_labels and current_step < len(step_labels):
                ax.text(0.5, 0.2, f"Step {current_step + 1}/{total_steps}: {step_labels[current_step]}",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.set_xlim(0, 1)
            ax.set_ylim(-0.1, 0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        except Exception as e:
            log_exception(e, context="ChartUtils.add_progress_indicator")

    @staticmethod
    def create_interactive_slider_plot(x_data: np.ndarray,
                                      y_data_list: List[np.ndarray],
                                      labels: List[str],
                                      title: str,
                                      x_label: str = "X",
                                      y_label: str = "Y") -> go.Figure:
        """
        Create interactive plot with slider for comparing multiple datasets.

        Args:
            x_data: x-axis data
            y_data_list: list of y-axis data arrays
            labels: list of labels for each dataset
            title: plot title
            x_label: x-axis label
            y_label: y-axis label

        Returns:
            Plotly Figure object with slider
        """
        try:
            fig = go.Figure()

            # Add traces for each dataset
            for i, (y_data, label) in enumerate(zip(y_data_list, labels)):
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines',
                        name=label,
                        line=dict(width=2),
                        visible=(i == 0)  # Only first trace visible initially
                    )
                )

            # Create slider
            steps = []
            for i in range(len(y_data_list)):
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(y_data_list)}],
                    label=labels[i]
                )
                step["args"][0]["visible"][i] = True
                steps.append(step)

            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Dataset: "},
                pad={"t": 50},
                steps=steps
            )]

            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                sliders=sliders,
                height=500
            )

            return fig

        except Exception as e:
            log_exception(e, context="ChartUtils.create_interactive_slider_plot")
            raise

    @staticmethod
    def add_formula_box(ax, formula: str, description: str,
                        position: Tuple[float, float] = (0.02, 0.98),
                        fontsize: int = 10):
        """
        Add formula box with explanation to plot.

        Args:
            ax: matplotlib axes object
            formula: mathematical formula (LaTeX supported)
            description: description of the formula
            position: position for the box (x, y)
            fontsize: font size
        """
        try:
            text = f"${formula}$\n\n{description}"

            ax.text(position[0], position[1], text,
                   transform=ax.transAxes, fontsize=fontsize,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                             edgecolor="gray", alpha=0.9),
                   wrap=True)

        except Exception as e:
            log_exception(e, context="ChartUtils.add_formula_box")

    @staticmethod
    def save_figure_for_streamlit(fig: plt.Figure,
                                  dpi: int = 150,
                                  bbox_inches: str = 'tight') -> str:
        """
        Save matplotlib figure for display in Streamlit.

        Args:
            fig: matplotlib Figure object
            dpi: resolution for saved figure
            bbox_inches: bbox_inches parameter

        Returns:
            Base64 encoded image string
        """
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=dpi, bbox_inches=bbox_inches,
                       facecolor='white', edgecolor='none')
            buffer.seek(0)

            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            log_exception(e, context="ChartUtils.save_figure_for_streamlit")
            return ""

    @staticmethod
    def create_performance_dashboard(metrics: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive performance dashboard.

        Args:
            metrics: Dictionary containing performance metrics

        Returns:
            Plotly Figure object with multiple subplots
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy vs Components', 'Processing Time',
                               'Memory Usage', 'Recognition Performance'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "area"}, {"type": "table"}]]
            )

            # Accuracy plot
            if 'component_counts' in metrics and 'accuracies' in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=metrics['component_counts'],
                        y=metrics['accuracies'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )

            # Processing time bar chart
            if 'processing_times' in metrics:
                operations = list(metrics['processing_times'].keys())
                times = list(metrics['processing_times'].values())

                fig.add_trace(
                    go.Bar(x=operations, y=times, name='Processing Time'),
                    row=1, col=2
                )

            # Memory usage area chart
            if 'memory_usage' in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(metrics['memory_usage']))),
                        y=metrics['memory_usage'],
                        mode='lines',
                        fill='tonexty',
                        name='Memory Usage'
                    ),
                    row=2, col=1
                )

            # Performance summary table
            if 'summary_stats' in metrics:
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Metric', 'Value']),
                        cells=dict(values=[
                            list(metrics['summary_stats'].keys()),
                            list(metrics['summary_stats'].values())
                        ])
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                title="PCA Face Recognition Performance Dashboard",
                height=800,
                showlegend=False
            )

            return fig

        except Exception as e:
            log_exception(e, context="ChartUtils.create_performance_dashboard")
            raise


def main():
    """Test the ChartUtils class."""
    import sys

    try:
        chart_utils = ChartUtils()

        print("Testing ChartUtils...")

        # Test subplot grid creation
        fig, axes = chart_utils.create_subplot_grid(6, max_cols=3)
        print("✓ Subplot grid creation test passed")

        # Test comparison plot
        test_images = [np.random.rand(112, 92) for _ in range(4)]
        labels = [f'Image {i+1}' for i in range(4)]
        fig = chart_utils.create_comparison_plot(test_images, labels, "Test Comparison")
        print("✓ Comparison plot test passed")

        # Test formula box
        fig, ax = plt.subplots(figsize=(8, 6))
        chart_utils.add_formula_box(ax, "C = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\mu)(x_i - \\mu)^T",
                                   "Covariance matrix formula")
        print("✓ Formula box test passed")

        print("All chart utility tests passed!")

    except Exception as e:
        print(f"ChartUtils test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()