"""
Face Recognition Page - PCA Face Recognition Demo

This page implements face recognition workflow for testing the trained PCA model
on held-out test images from the AT&T dataset with mathematical analysis and visual interpretation.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
import pandas as pd

# Import handling for exec() context
import sys
import os

# Always use absolute imports to work with Streamlit exec() context
# Get current file path - handle exec() context
current_file = globals().get('__file__')
if current_file:
    src_path = os.path.join(os.path.dirname(current_file), '..')
else:
    # When executed via exec(), use working directory
    src_path = os.path.join(os.getcwd(), 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from core.pca import PCA
    from core.eigenfaces_recognizer import EigenfacesRecognizer
    from processing.dataset_loader import DatasetLoader
    from utils.logger import log_exception, get_error_logger
    from visualization.visualizations import VisualizationUtils
    from config import DATASET_CONFIG
    logger = get_error_logger()
except ImportError as e:
    # Fallback - basic error handling without logging
    st.error(f"Error importing modules: {e}")
    st.error("Please make sure all required modules are available")
    st.stop()


def main():
    """Main function for Face Recognition page."""
    # Fix exec() context issue by setting __name__ global
    if '__name__' not in globals():
        globals()['__name__'] = '__main__'

    try:
        st.title("🔍 Face Recognition")
        st.markdown("PCA-based Face Recognition with Mathematical Analysis")
        logger.logger.info("Face Recognition page loaded")

        # Check if trained PCA model exists
        if 'pca_model' not in st.session_state or st.session_state.pca_model is None:
            st.error("❌ No trained PCA model found!")
            st.warning("⚠️ Please complete the Eigenfaces workflow first to train a model.")
            st.info("Go to the **Eigenfaces** page and train the PCA model on the AT&T dataset.")
            logger.logger.warning("Face Recognition accessed without trained model")
            return

        # Check if test data exists
        if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
            st.error("❌ No test data found!")
            st.warning("⚠️ Please complete the Eigenfaces workflow first to generate test data.")
            logger.logger.warning("Face Recognition accessed without test data")
            return

        # Initialize visualization utilities
        try:
            viz_utils = VisualizationUtils()
        except Exception as e:
            log_exception(e, context="Initializing visualization utilities")
            viz_utils = None

        # Initialize Eigenfaces Recognizer
        try:
            recognizer = EigenfacesRecognizer(
                pca=st.session_state.pca_model,
                distance_metric=st.session_state.training_params.get('distance_metric', 'Euclidean')
            )
            logger.logger.info("Eigenfaces recognizer initialized successfully")
        except Exception as e:
            log_exception(e, context="Initializing Eigenfaces recognizer")
            st.error("❌ Error initializing face recognition system")
            st.error("Please check the log file for detailed information.")
            return

        # Create tabs for face recognition workflow
        tab_names = [
            "Model Overview & Test Dataset",
            "Single Face Recognition",
            "Batch Recognition Testing",
            "Performance Analysis & Confusion Matrix"
        ]

        tabs = st.tabs(tab_names)

        # Tab 1: Model Overview & Test Dataset
        with tabs[0]:
            _model_overview_tab(recognizer)

        # Tab 2: Single Face Recognition
        with tabs[1]:
            _single_recognition_tab(recognizer, viz_utils)

        # Tab 3: Batch Recognition Testing
        with tabs[2]:
            _batch_recognition_tab(recognizer, viz_utils)

        # Tab 4: Performance Analysis & Confusion Matrix
        with tabs[3]:
            _performance_analysis_tab(recognizer, viz_utils)

    except Exception as e:
        log_exception(e, context="Face Recognition main")
        st.error("❌ An error occurred in the Face Recognition page")
        st.error("Please check the log file for detailed information.")


def _model_overview_tab(recognizer: EigenfacesRecognizer):
    """Model overview and test dataset information tab."""
    try:
        st.header("📊 Model Overview & Test Dataset")
        st.info("Review trained model and understand test dataset structure")
        logger.logger.info("Model Overview & Test Dataset tab loaded")

        # Model information
        st.subheader("🤖 Trained PCA Model Information")
        col1, col2 = st.columns(2)

        pca = st.session_state.pca_model
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        with col1:
            st.markdown("**Model Parameters:**")
            st.write(f"- **PCA Components**: {pca.n_components}")
            st.write(f"- **Training Samples**: {X_train.shape[0]}")
            st.write(f"- **Original Dimensions**: {X_train.shape[1]} (92×112)")
            st.write(f"- **Distance Metric**: {st.session_state.training_params.get('distance_metric', 'Euclidean')}")
            st.write(f"- **Compression Ratio**: {pca.n_components/X_train.shape[1]:.6f}")

        with col2:
            st.markdown("**Performance Metrics:**")
            st.write(f"- **Variance Explained**: {np.sum(pca.explained_variance_ratio_):.4f}")
            st.write(f"- **Reconstruction MSE**: {pca.get_reconstruction_error(X_train):.6f}")
            st.write(f"- **Optimal Components (95%)**: {pca.get_optimal_components(0.95)}")
            st.write(f"- **Largest Eigenvalue**: {pca.explained_variance_[0]:.2f}")
            st.write(f"- **Smallest Eigenvalue**: {pca.explained_variance_[-1]:.4f}")

        # Test dataset information
        st.subheader("🧪 Test Dataset Overview")

        # Calculate test dataset statistics
        unique_subjects_test = np.unique(y_test)
        test_samples_per_subject = {}
        for subject in unique_subjects_test:
            count = np.sum(y_test == subject)
            test_samples_per_subject[subject] = count

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Test Dataset Structure:**")
            st.write(f"- **Total Test Samples**: {len(y_test)}")
            st.write(f"- **Unique Subjects**: {len(unique_subjects_test)}")
            st.write(f"- **Subjects Range**: {min(unique_subjects_test)} to {max(unique_subjects_test)}")
            avg_per_subject = len(y_test) / len(unique_subjects_test)
            st.write(f"- **Avg Samples/Subject**: {avg_per_subject:.1f}")

            # Show samples per subject distribution
            st.markdown("**Samples per Subject:**")
            for subject in sorted(unique_subjects_test)[:10]:  # Show first 10
                st.write(f"  - Subject {subject}: {test_samples_per_subject[subject]} samples")
            if len(unique_subjects_test) > 10:
                st.write(f"  - ... and {len(unique_subjects_test) - 10} more subjects")

        with col2:
            st.markdown("**Recognition Task Complexity:**")
            st.write(f"- **Multi-class Classification**: {len(unique_subjects_test)} classes")
            st.write(f"- **One-vs-All Recognition**: Each subject vs others")
            st.write(f"- **Feature Space**: {pca.n_components} dimensions")
            st.write(f"- **Total Comparisons/Query**: {len(unique_subjects_test)}")

            # Mathematical complexity
            st.markdown("**Computational Complexity:**")
            st.write(f"- **Projection**: O({pca.n_components} × {X_train.shape[1]})")
            st.write(f"- **Distance Calculation**: O({pca.n_components}) per subject")
            st.write(f"- **Total per Query**: O({len(unique_subjects_test) * pca.n_components})")

        # Visual sample of test dataset
        st.subheader("👥 Test Dataset Sample")

        # Show sample test images
        n_samples = min(8, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Sample Test Images from AT&T Dataset', fontsize=16, fontweight='bold')

        for i, idx in enumerate(sample_indices):
            row, col = i // 4, i % 4
            test_face = X_test[idx].reshape(112, 92)
            axes[row, col].imshow(test_face, cmap='gray')
            axes[row, col].set_title(f'Test Sample\nSubject {y_test[idx]}')
            axes[row, col].axis('off')

        plt.tight_layout()
        st.pyplot(fig)

        # Mathematical foundation
        st.subheader("📐 Face Recognition Mathematics")
        with st.expander("Mathematical Foundation", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Recognition Process:**")
                st.latex(r"y_{test} = W^T(x_{test} - \mu)")
                st.markdown("*Project test face to eigenface space*")

                st.latex(r"d(x_{test}, s_i) = \|y_{test} - y_{s_i}\|^2")
                st.markdown("*Calculate distance to each subject's mean in eigenface space*")

                st.latex(r"predicted = \arg\min_{i} d(x_{test}, s_i)")
                st.markdown("*Select subject with minimum distance*")

            with col2:
                st.markdown("**Distance Metrics:**")
                current_metric = st.session_state.training_params.get('distance_metric', 'Euclidean')

                if current_metric == "Euclidean":
                    st.latex(r"d_{euclidean}(a,b) = \sqrt{\sum_{i=1}^{k} (a_i - b_i)^2}")
                else:
                    st.latex(r"d_{cosine}(a,b) = 1 - \frac{a \cdot b}{\|a\| \|b\|}")

                st.markdown("**Confidence Score:**")
                st.latex(r"confidence = 1 - \frac{d_{min}}{d_{max}}")
                st.markdown("*Normalized distance-based confidence*")

        st.success("✅ Model and test dataset overview completed!")
        logger.logger.info("Model overview and test dataset analysis completed successfully")

    except Exception as e:
        log_exception(e, context="Model Overview & Test Dataset tab")
        st.error("❌ Error loading model overview and test dataset")


def _single_recognition_tab(recognizer: EigenfacesRecognizer, viz_utils: Optional[VisualizationUtils]):
    """Single face recognition testing tab."""
    try:
        st.header("🎯 Single Face Recognition")
        st.info("Test individual face recognition with detailed analysis")
        logger.logger.info("Single Face Recognition tab loaded")

        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Test sample selection
        st.subheader("📸 Test Sample Selection")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Subject selection
            available_subjects = sorted(np.unique(y_test))
            selected_subject = st.selectbox(
                "Select Test Subject:",
                options=available_subjects,
                help="Choose a subject to test recognition for",
                key="test_subject_select"
            )

        with col2:
            # Sample index within subject
            subject_indices = np.where(y_test == selected_subject)[0]
            sample_options = [f"Sample {i+1} (Index {idx})" for i, idx in enumerate(subject_indices)]
            selected_sample_str = st.selectbox(
                "Select Sample:",
                options=sample_options,
                help="Choose a specific sample of the selected subject",
                key="test_sample_select"
            )
            selected_index = int(selected_sample_str.split("(Index ")[1].split(")")[0])

        with col3:
            st.markdown("**Selection Info:**")
            st.write(f"**Subject**: {selected_subject}")
            st.write(f"**Index**: {selected_index}")
            st.write(f"**Total subject samples**: {len(subject_indices)}")

        # Recognition button
        st.subheader("🔍 Recognition Testing")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Face Recognition", type="primary", key="single_recognition_button"):
                with st.spinner("Performing face recognition..."):
                    try:
                        # Get test image
                        test_image = X_test[selected_index:selected_index+1]
                        true_label = y_test[selected_index]

                        # Perform recognition
                        predicted_label, confidence, distances = recognizer.recognize(test_image[0])

                        # Store results in session state
                        st.session_state.last_recognition_result = {
                            'test_image': test_image[0],
                            'true_label': true_label,
                            'predicted_label': predicted_label,
                            'confidence': confidence,
                            'distances': distances,
                            'selected_index': selected_index
                        }

                        st.success("✅ Face recognition completed!")
                        logger.logger.info(f"Single recognition completed: true={true_label}, predicted={predicted_label}, confidence={confidence:.4f}")

                    except Exception as e:
                        log_exception(e, context="Single face recognition")
                        st.error("❌ Face recognition failed")
                        st.error("Please check the log file for detailed information.")

        # Display results if available
        if 'last_recognition_result' in st.session_state:
            result = st.session_state.last_recognition_result

            st.subheader("📋 Recognition Results")

            # Results summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("True Subject", result['true_label'])

            with col2:
                st.metric("Predicted Subject", result['predicted_label'])

            with col3:
                is_correct = result['true_label'] == result['predicted_label']
                st.metric("Correct", "✅ Yes" if is_correct else "❌ No")

            with col4:
                st.metric("Confidence", f"{result['confidence']:.4f}")

            # Visual analysis
            st.subheader("👁️ Visual Analysis")
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                st.markdown("**Test Image:**")
                fig, ax = plt.subplots(figsize=(4, 5))
                test_face = result['test_image'].reshape(112, 92)
                ax.imshow(test_face, cmap='gray')
                ax.set_title(f'Test Image\nSubject {result["true_label"]}')
                ax.axis('off')
                st.pyplot(fig)

            with col2:
                st.markdown("**Distance Analysis:**")
                distances = result['distances']

                # Create distance analysis chart
                subjects = list(distances.keys())
                distance_values = list(distances.values())

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(len(subjects)), distance_values)

                # Highlight the predicted and true subjects
                for i, subject in enumerate(subjects):
                    if subject == result['predicted_label']:
                        bars[i].set_color('green' if is_correct else 'red')
                        bars[i].set_alpha(0.7)
                        ax.text(i, distance_values[i] + max(distance_values)*0.01,
                               'PREDICTED' if subject == result['predicted_label'] else '',
                               ha='center', fontweight='bold', fontsize=8)

                ax.set_xlabel('Subject ID')
                ax.set_ylabel('Distance (Lower is Better)')
                ax.set_title(f'Distance to All Subjects (True: {result["true_label"]}, Predicted: {result["predicted_label"]})')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(len(subjects)))
                ax.set_xticklabels(subjects, rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

            with col3:
                st.markdown("**Top 5 Candidates:**")
                sorted_distances = sorted(distances.items(), key=lambda x: x[1])
                for i, (subject, distance) in enumerate(sorted_distances[:5]):
                    emoji = "🏆" if subject == result['predicted_label'] else "🥈" if subject == result['true_label'] else "👤"
                    st.write(f"{i+1}. {emoji} Subject {subject}: {distance:.4f}")

            # Mathematical analysis
            st.subheader("📊 Mathematical Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Distance Statistics:**")
                distance_vals = list(distances.values())
                st.write(f"- **Min Distance**: {min(distance_vals):.6f}")
                st.write(f"- **Max Distance**: {max(distance_vals):.6f}")
                st.write(f"- **Mean Distance**: {np.mean(distance_vals):.6f}")
                st.write(f"- **Std Deviation**: {np.std(distance_vals):.6f}")

                predicted_distance = distances[result['predicted_label']]
                st.write(f"- **Predicted Subject Distance**: {predicted_distance:.6f}")
                st.write(f"- **Distance Percentile**: {(predicted_distance - min(distance_vals))/(max(distance_vals)-min(distance_vals))*100:.1f}%")

            with col2:
                st.markdown("**Recognition Confidence:**")
                confidence = result['confidence']
                st.write(f"- **Normalized Confidence**: {confidence:.4f}")

                # Confidence interpretation
                if confidence > 0.8:
                    st.write("🟢 **High Confidence** - Very reliable prediction")
                elif confidence > 0.6:
                    st.write("🟡 **Medium Confidence** - Moderately reliable prediction")
                else:
                    st.write("🔴 **Low Confidence** - Prediction should be verified")

                st.write(f"- **Recognition Correct**: {'✅ Yes' if is_correct else '❌ No'}")

        st.info("💡 **Tip**: Select different subjects and samples to test the recognition system across various scenarios.")

    except Exception as e:
        log_exception(e, context="Single Face Recognition tab")
        st.error("❌ Error in single face recognition")


def _batch_recognition_tab(recognizer: EigenfacesRecognizer, viz_utils: Optional[VisualizationUtils]):
    """Batch face recognition testing tab."""
    try:
        st.header("📦 Batch Recognition Testing")
        st.info("Test recognition on multiple samples with comprehensive analysis")
        logger.logger.info("Batch Recognition Testing tab loaded")

        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Batch configuration
        st.subheader("⚙️ Batch Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Sample size selection
            max_batch_size = min(50, len(X_test))
            batch_size = st.slider(
                "Number of Test Samples:",
                min_value=5,
                max_value=max_batch_size,
                value=min(20, len(X_test)),
                step=5,
                help="Number of test samples to evaluate",
                key="batch_size_slider"
            )

        with col2:
            # Sampling strategy
            sampling_strategy = st.selectbox(
                "Sampling Strategy:",
                options=["Random", "Balanced (per subject)", "Sequential"],
                index=0,
                help="How to select test samples",
                key="sampling_strategy_select"
            )

        with col3:
            # Subjects to include (for balanced sampling)
            available_subjects = sorted(np.unique(y_test))
            if sampling_strategy == "Balanced (per subject)":
                n_subjects = st.slider(
                    "Subjects to Include:",
                    min_value=2,
                    max_value=len(available_subjects),
                    value=min(10, len(available_subjects)),
                    step=1,
                    help="Number of subjects to include in balanced sampling",
                    key="n_subjects_slider"
                )
            else:
                n_subjects = len(available_subjects)

        # Batch recognition button
        st.subheader("🚀 Batch Recognition")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔥 Run Batch Recognition", type="primary", key="batch_recognition_button"):
                with st.spinner("Performing batch face recognition..."):
                    try:
                        # Select test samples based on strategy
                        if sampling_strategy == "Random":
                            selected_indices = np.random.choice(len(X_test), batch_size, replace=False)
                        elif sampling_strategy == "Balanced (per subject)":
                            selected_subjects = available_subjects[:n_subjects]
                            samples_per_subject = batch_size // n_subjects
                            selected_indices = []

                            for subject in selected_subjects:
                                subject_indices = np.where(y_test == subject)[0]
                                if len(subject_indices) >= samples_per_subject:
                                    chosen = np.random.choice(subject_indices, samples_per_subject, replace=False)
                                    selected_indices.extend(chosen)

                            # Adjust to exact batch size
                            if len(selected_indices) > batch_size:
                                selected_indices = selected_indices[:batch_size]
                            elif len(selected_indices) < batch_size:
                                remaining = batch_size - len(selected_indices)
                                other_indices = [i for i in range(len(X_test)) if i not in selected_indices]
                                if other_indices:
                                    additional = np.random.choice(other_indices, min(remaining, len(other_indices)), replace=False)
                                    selected_indices.extend(additional)
                        else:  # Sequential
                            selected_indices = list(range(min(batch_size, len(X_test))))

                        # Perform batch recognition
                        batch_results = []
                        correct_predictions = 0

                        for idx in selected_indices:
                            test_image = X_test[idx]
                            true_label = y_test[idx]

                            predicted_label, confidence, distances = recognizer.recognize(test_image)

                            batch_results.append({
                                'index': idx,
                                'true_label': true_label,
                                'predicted_label': predicted_label,
                                'confidence': confidence,
                                'correct': true_label == predicted_label,
                                'min_distance': min(distances.values())
                            })

                            if true_label == predicted_label:
                                correct_predictions += 1

                        # Calculate overall metrics
                        accuracy = correct_predictions / len(batch_results)
                        avg_confidence = np.mean([r['confidence'] for r in batch_results])

                        # Store results in session state
                        st.session_state.last_batch_result = {
                            'results': batch_results,
                            'accuracy': accuracy,
                            'avg_confidence': avg_confidence,
                            'batch_size': len(batch_results),
                            'correct_predictions': correct_predictions,
                            'sampling_strategy': sampling_strategy,
                            'selected_indices': selected_indices
                        }

                        st.success(f"✅ Batch recognition completed! Accuracy: {accuracy:.4f}")
                        logger.logger.info(f"Batch recognition completed: {len(batch_results)} samples, accuracy={accuracy:.4f}")

                    except Exception as e:
                        log_exception(e, context="Batch face recognition")
                        st.error("❌ Batch recognition failed")
                        st.error("Please check the log file for detailed information.")

        # Display batch results if available
        if 'last_batch_result' in st.session_state:
            batch_result = st.session_state.last_batch_result
            results = batch_result['results']

            st.subheader("📊 Batch Recognition Results")

            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Samples Tested", batch_result['batch_size'])

            with col2:
                st.metric("Accuracy", f"{batch_result['accuracy']:.4f}")

            with col3:
                st.metric("Correct Predictions", f"{batch_result['correct_predictions']}/{batch_result['batch_size']}")

            with col4:
                st.metric("Avg Confidence", f"{batch_result['avg_confidence']:.4f}")

            # Results breakdown by subject
            st.subheader("📈 Results Analysis by Subject")

            # Create per-subject analysis
            subject_stats = {}
            for result in results:
                subject = result['true_label']
                if subject not in subject_stats:
                    subject_stats[subject] = {'correct': 0, 'total': 0, 'confidences': []}

                subject_stats[subject]['total'] += 1
                if result['correct']:
                    subject_stats[subject]['correct'] += 1
                subject_stats[subject]['confidences'].append(result['confidence'])

            # Create summary table
            summary_data = []
            for subject in sorted(subject_stats.keys()):
                stats = subject_stats[subject]
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                avg_conf = np.mean(stats['confidences'])

                summary_data.append({
                    'Subject': subject,
                    'Total': stats['total'],
                    'Correct': stats['correct'],
                    'Accuracy': f"{accuracy:.4f}",
                    'Avg Confidence': f"{avg_conf:.4f}"
                })

            # Display as DataFrame
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Visual analysis
            if viz_utils:
                st.subheader("🎨 Visual Analysis")

                # Create two plots: confidence distribution and accuracy by confidence
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Confidence distribution
                confidences = [r['confidence'] for r in results]
                correct_confidences = [r['confidence'] for r in results if r['correct']]
                incorrect_confidences = [r['confidence'] for r in results if not r['correct']]

                ax1.hist(correct_confidences, bins=15, alpha=0.7, label='Correct Predictions', color='green')
                if incorrect_confidences:
                    ax1.hist(incorrect_confidences, bins=10, alpha=0.7, label='Incorrect Predictions', color='red')
                ax1.set_xlabel('Confidence Score')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Confidence Score Distribution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Accuracy vs confidence threshold
                confidence_thresholds = np.linspace(0, 1, 21)
                accuracies = []

                for threshold in confidence_thresholds:
                    filtered_results = [r for r in results if r['confidence'] >= threshold]
                    if filtered_results:
                        acc = sum(1 for r in filtered_results if r['correct']) / len(filtered_results)
                        accuracies.append(acc)
                    else:
                        accuracies.append(0)

                ax2.plot(confidence_thresholds, accuracies, 'bo-', linewidth=2, markersize=6)
                ax2.set_xlabel('Confidence Threshold')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Accuracy vs Confidence Threshold')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=batch_result['accuracy'], color='r', linestyle='--',
                          label=f'Overall Accuracy ({batch_result["accuracy"]:.3f})')
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)

            # Sample results visualization
            st.subheader("🖼️ Sample Recognition Results")

            # Show a grid of sample results
            n_samples_to_show = min(12, len(results))
            sample_indices = np.random.choice(len(results), n_samples_to_show, replace=False)

            cols = 4
            rows = (n_samples_to_show + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            fig.suptitle('Sample Recognition Results', fontsize=16, fontweight='bold')

            if rows == 1:
                axes = axes.reshape(1, -1)

            for i, result_idx in enumerate(sample_indices):
                result = results[result_idx]
                row, col = i // cols, i % cols

                # Get the actual test image
                test_image = X_test[result['index']]
                face_img = test_image.reshape(112, 92)

                axes[row, col].imshow(face_img, cmap='gray')

                # Create title with prediction info
                title = f"True: {result['true_label']}\nPred: {result['predicted_label']}"
                if result['correct']:
                    axes[row, col].set_title(f"{title}\n✅ {result['confidence']:.3f}", color='green')
                else:
                    axes[row, col].set_title(f"{title}\n❌ {result['confidence']:.3f}", color='red')

                axes[row, col].axis('off')

            # Hide unused subplots
            for i in range(n_samples_to_show, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')

            plt.tight_layout()
            st.pyplot(fig)

            # Detailed results table
            st.subheader("📋 Detailed Results")
            detailed_data = []
            for result in results:
                detailed_data.append({
                    'Index': result['index'],
                    'True Subject': result['true_label'],
                    'Predicted Subject': result['predicted_label'],
                    'Correct': '✅ Yes' if result['correct'] else '❌ No',
                    'Confidence': f"{result['confidence']:.6f}",
                    'Min Distance': f"{result['min_distance']:.6f}"
                })

            detailed_df = pd.DataFrame(detailed_data)
            st.dataframe(detailed_df, use_container_width=True)

            # Export option
            st.subheader("💾 Export Results")
            if st.button("📥 Export Batch Results to CSV", key="export_batch_button"):
                try:
                    csv_data = pd.DataFrame(detailed_data)
                    csv_str = csv_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_str,
                        file_name=f"batch_recognition_results_{batch_size}_samples.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    log_exception(e, context="Exporting batch results")
                    st.error("❌ Failed to export results")

        st.info("💡 **Tip**: Use different sampling strategies to test recognition performance across various scenarios.")

    except Exception as e:
        log_exception(e, context="Batch Recognition Testing tab")
        st.error("❌ Error in batch recognition testing")


def _performance_analysis_tab(recognizer: EigenfacesRecognizer, viz_utils: Optional[VisualizationUtils]):
    """Performance analysis and confusion matrix tab."""
    try:
        st.header("📈 Performance Analysis & Confusion Matrix")
        st.info("Comprehensive evaluation of face recognition performance")
        logger.logger.info("Performance Analysis & Confusion Matrix tab loaded")

        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Performance evaluation configuration
        st.subheader("⚙️ Evaluation Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            evaluation_size = st.slider(
                "Evaluation Set Size:",
                min_value=10,
                max_value=min(100, len(X_test)),
                value=min(50, len(X_test)),
                step=10,
                help="Number of test samples for comprehensive evaluation",
                key="evaluation_size_slider"
            )

        with col2:
            eval_strategy = st.selectbox(
                "Evaluation Strategy:",
                options=["Random Sample", "Balanced per Subject", "All Test Data"],
                index=0,
                help="Strategy for selecting evaluation samples",
                key="eval_strategy_select"
            )

        with col3:
            include_confidence_analysis = st.checkbox(
                "Include Confidence Analysis",
                value=True,
                help="Generate confidence-based performance metrics",
                key="confidence_analysis_checkbox"
            )

        # Run evaluation button
        st.subheader("🚀 Run Performance Evaluation")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 Run Full Evaluation", type="primary", key="performance_eval_button"):
                with st.spinner("Performing comprehensive performance evaluation..."):
                    try:
                        # Select evaluation samples
                        if eval_strategy == "Random Sample":
                            eval_indices = np.random.choice(len(X_test), evaluation_size, replace=False)
                        elif eval_strategy == "Balanced per Subject":
                            available_subjects = sorted(np.unique(y_test))
                            samples_per_subject = evaluation_size // len(available_subjects)
                            eval_indices = []

                            for subject in available_subjects:
                                subject_indices = np.where(y_test == subject)[0]
                                if len(subject_indices) >= samples_per_subject:
                                    chosen = np.random.choice(subject_indices, samples_per_subject, replace=False)
                                    eval_indices.extend(chosen)
                        else:  # All Test Data
                            eval_indices = list(range(len(X_test)))

                        # Perform evaluation
                        y_true = []
                        y_pred = []
                        confidences = []
                        all_distances = {}

                        for idx in eval_indices:
                            test_image = X_test[idx]
                            true_label = y_test[idx]

                            predicted_label, confidence, distances = recognizer.recognize(test_image)

                            y_true.append(true_label)
                            y_pred.append(predicted_label)
                            confidences.append(confidence)

                            if idx not in all_distances:
                                all_distances[idx] = distances

                        # Calculate metrics
                        y_true = np.array(y_true)
                        y_pred = np.array(y_pred)
                        confidences = np.array(confidences)

                        # Overall accuracy
                        overall_accuracy = np.mean(y_true == y_pred)

                        # Per-subject accuracy
                        unique_subjects = sorted(np.unique(np.concatenate([y_true, y_pred])))
                        per_subject_accuracy = {}
                        per_subject_precision = {}
                        per_subject_recall = {}
                        per_subject_f1 = {}

                        for subject in unique_subjects:
                            # True Positive, False Positive, False Negative for this subject
                            tp = np.sum((y_true == subject) & (y_pred == subject))
                            fp = np.sum((y_true != subject) & (y_pred == subject))
                            fn = np.sum((y_true == subject) & (y_pred != subject))

                            per_subject_accuracy[subject] = np.sum((y_true == subject) & (y_pred == subject)) / np.sum(y_true == subject) if np.sum(y_true == subject) > 0 else 0
                            per_subject_precision[subject] = tp / (tp + fp) if (tp + fp) > 0 else 0
                            per_subject_recall[subject] = tp / (tp + fn) if (tp + fn) > 0 else 0
                            per_subject_f1[subject] = 2 * (per_subject_precision[subject] * per_subject_recall[subject]) / (per_subject_precision[subject] + per_subject_recall[subject]) if (per_subject_precision[subject] + per_subject_recall[subject]) > 0 else 0

                        # Confidence analysis
                        if include_confidence_analysis:
                            # Accuracy at different confidence thresholds
                            confidence_thresholds = np.linspace(0, 1, 21)
                            threshold_accuracies = []
                            threshold_counts = []

                            for threshold in confidence_thresholds:
                                mask = confidences >= threshold
                                if np.sum(mask) > 0:
                                    acc = np.mean(y_true[mask] == y_pred[mask])
                                    threshold_accuracies.append(acc)
                                    threshold_counts.append(np.sum(mask))
                                else:
                                    threshold_accuracies.append(0)
                                    threshold_counts.append(0)

                        # Store results
                        evaluation_results = {
                            'y_true': y_true,
                            'y_pred': y_pred,
                            'confidences': confidences,
                            'overall_accuracy': overall_accuracy,
                            'per_subject_accuracy': per_subject_accuracy,
                            'per_subject_precision': per_subject_precision,
                            'per_subject_recall': per_subject_recall,
                            'per_subject_f1': per_subject_f1,
                            'unique_subjects': unique_subjects,
                            'eval_indices': eval_indices,
                            'confidence_thresholds': confidence_thresholds if include_confidence_analysis else None,
                            'threshold_accuracies': threshold_accuracies if include_confidence_analysis else None,
                            'threshold_counts': threshold_counts if include_confidence_analysis else None
                        }

                        st.session_state.last_evaluation_result = evaluation_results

                        st.success(f"✅ Performance evaluation completed! Overall Accuracy: {overall_accuracy:.4f}")
                        logger.logger.info(f"Performance evaluation completed: {len(eval_indices)} samples, accuracy={overall_accuracy:.4f}")

                    except Exception as e:
                        log_exception(e, context="Performance evaluation")
                        st.error("❌ Performance evaluation failed")
                        st.error("Please check the log file for detailed information.")

        # Display evaluation results if available
        if 'last_evaluation_result' in st.session_state:
            eval_result = st.session_state.last_evaluation_result

            st.subheader("📊 Overall Performance Metrics")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Overall Accuracy", f"{eval_result['overall_accuracy']:.4f}")

            with col2:
                st.metric("Samples Evaluated", len(eval_result['y_true']))

            with col3:
                st.metric("Correct Predictions", f"{np.sum(eval_result['y_true'] == eval_result['y_pred'])}")

            with col4:
                st.metric("Avg Confidence", f"{np.mean(eval_result['confidences']):.4f}")

            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")

            try:
                from sklearn.metrics import confusion_matrix, classification_report

                # Create confusion matrix
                cm = confusion_matrix(eval_result['y_true'], eval_result['y_pred'], labels=eval_result['unique_subjects'])

                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(12, 10))

                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=eval_result['unique_subjects'],
                           yticklabels=eval_result['unique_subjects'],
                           ax=ax)

                ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
                ax.set_xlabel('Predicted Subject', fontsize=12)
                ax.set_ylabel('True Subject', fontsize=12)

                plt.tight_layout()
                st.pyplot(fig)

                # Per-subject performance metrics
                st.subheader("📈 Per-Subject Performance Analysis")

                # Create performance table
                performance_data = []
                for subject in eval_result['unique_subjects']:
                    performance_data.append({
                        'Subject': subject,
                        'Accuracy': f"{eval_result['per_subject_accuracy'][subject]:.4f}",
                        'Precision': f"{eval_result['per_subject_precision'][subject]:.4f}",
                        'Recall': f"{eval_result['per_subject_recall'][subject]:.4f}",
                        'F1-Score': f"{eval_result['per_subject_f1'][subject]:.4f}"
                    })

                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, use_container_width=True)

                # Performance visualization
                if viz_utils:
                    st.subheader("🎨 Performance Visualization")

                    # Create multiple performance plots
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle('Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

                    # 1. Per-subject accuracy
                    subjects = list(eval_result['per_subject_accuracy'].keys())
                    accuracies = list(eval_result['per_subject_accuracy'].values())

                    ax1.bar(range(len(subjects)), accuracies, color='skyblue', edgecolor='black')
                    ax1.set_xlabel('Subject ID')
                    ax1.set_ylabel('Accuracy')
                    ax1.set_title('Per-Subject Recognition Accuracy')
                    ax1.set_xticks(range(len(subjects)))
                    ax1.set_xticklabels(subjects, rotation=45)
                    ax1.grid(True, alpha=0.3)
                    ax1.axhline(y=eval_result['overall_accuracy'], color='r', linestyle='--',
                              label=f'Overall ({eval_result["overall_accuracy"]:.3f})')
                    ax1.legend()

                    # 2. Precision vs Recall scatter
                    precisions = list(eval_result['per_subject_precision'].values())
                    recalls = list(eval_result['per_subject_recall'].values())

                    scatter = ax2.scatter(precisions, recalls, c=subjects, cmap='viridis', s=100, alpha=0.7)
                    ax2.set_xlabel('Precision')
                    ax2.set_ylabel('Recall')
                    ax2.set_title('Precision vs Recall by Subject')
                    ax2.grid(True, alpha=0.3)
                    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)

                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax2)
                    cbar.set_label('Subject ID')

                    # 3. F1-Scores
                    f1_scores = list(eval_result['per_subject_f1'].values())

                    ax3.bar(range(len(subjects)), f1_scores, color='lightgreen', edgecolor='black')
                    ax3.set_xlabel('Subject ID')
                    ax3.set_ylabel('F1-Score')
                    ax3.set_title('Per-Subject F1-Scores')
                    ax3.set_xticks(range(len(subjects)))
                    ax3.set_xticklabels(subjects, rotation=45)
                    ax3.grid(True, alpha=0.3)
                    ax3.axhline(y=np.mean(f1_scores), color='r', linestyle='--',
                              label=f'Average ({np.mean(f1_scores):.3f})')
                    ax3.legend()

                    # 4. Classification difficulty (error analysis)
                    error_rates = [1 - acc for acc in accuracies]

                    colors = ['green' if er < 0.1 else 'orange' if er < 0.2 else 'red' for er in error_rates]
                    ax4.bar(range(len(subjects)), error_rates, color=colors, edgecolor='black', alpha=0.7)
                    ax4.set_xlabel('Subject ID')
                    ax4.set_ylabel('Error Rate')
                    ax4.set_title('Classification Difficulty by Subject')
                    ax4.set_xticks(range(len(subjects)))
                    ax4.set_xticklabels(subjects, rotation=45)
                    ax4.grid(True, alpha=0.3)

                    # Add legend for colors
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='green', alpha=0.7, label='Easy (<10% errors)'),
                        Patch(facecolor='orange', alpha=0.7, label='Medium (10-20% errors)'),
                        Patch(facecolor='red', alpha=0.7, label='Hard (>20% errors)')
                    ]
                    ax4.legend(handles=legend_elements, loc='upper right')

                    plt.tight_layout()
                    st.pyplot(fig)

                # Confidence analysis
                if include_confidence_analysis and eval_result['threshold_accuracies']:
                    st.subheader("🎯 Confidence-Based Analysis")

                    # Confidence threshold analysis
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # Accuracy vs confidence threshold
                    ax1.plot(eval_result['confidence_thresholds'], eval_result['threshold_accuracies'],
                            'bo-', linewidth=2, markersize=6)
                    ax1.set_xlabel('Confidence Threshold')
                    ax1.set_ylabel('Accuracy')
                    ax1.set_title('Accuracy vs Confidence Threshold')
                    ax1.grid(True, alpha=0.3)
                    ax1.axhline(y=eval_result['overall_accuracy'], color='r', linestyle='--',
                              label=f'Overall ({eval_result["overall_accuracy"]:.3f})')
                    ax1.legend()

                    # Sample count vs confidence threshold
                    ax2.plot(eval_result['confidence_thresholds'], eval_result['threshold_counts'],
                            'go-', linewidth=2, markersize=6)
                    ax2.set_xlabel('Confidence Threshold')
                    ax2.set_ylabel('Number of Samples')
                    ax2.set_title('Sample Count vs Confidence Threshold')
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Confidence statistics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Confidence Threshold Analysis:**")
                        for threshold in [0.5, 0.7, 0.8, 0.9]:
                            idx = np.argmin(np.abs(eval_result['confidence_thresholds'] - threshold))
                            acc = eval_result['threshold_accuracies'][idx]
                            count = eval_result['threshold_counts'][idx]
                            st.write(f"- **Threshold {threshold:.1f}**: Accuracy {acc:.4f}, {count} samples")

                    with col2:
                        st.markdown("**Confidence Distribution:**")
                        correct_confidences = eval_result['confidences'][eval_result['y_true'] == eval_result['y_pred']]
                        incorrect_confidences = eval_result['confidences'][eval_result['y_true'] != eval_result['y_pred']]

                        st.write(f"- **Correct predictions**: Mean {np.mean(correct_confidences):.4f}, Std {np.std(correct_confidences):.4f}")
                        st.write(f"- **Incorrect predictions**: Mean {np.mean(incorrect_confidences):.4f}, Std {np.std(incorrect_confidences):.4f}")

                # Mathematical interpretation
                st.subheader("📐 Mathematical Interpretation")
                with st.expander("Statistical Analysis", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Classification Statistics:**")
                        st.write(f"- **Total Predictions**: {len(eval_result['y_true'])}")
                        st.write(f"- **Correct Predictions**: {np.sum(eval_result['y_true'] == eval_result['y_pred'])}")
                        st.write(f"- **Overall Accuracy**: {eval_result['overall_accuracy']:.4f}")
                        st.write(f"- **Error Rate**: {1 - eval_result['overall_accuracy']:.4f}")

                        # Class balance
                        unique, counts = np.unique(eval_result['y_true'], return_counts=True)
                        st.write(f"- **Class Balance**: {np.min(counts)}/{np.max(counts)} = {np.min(counts)/np.max(counts):.3f}")

                    with col2:
                        st.markdown("**Statistical Significance:**")

                        # Simple statistical test (accuracy vs random baseline)
                        n_classes = len(eval_result['unique_subjects'])
                        random_baseline = 1 / n_classes
                        z_score = (eval_result['overall_accuracy'] - random_baseline) / \
                                 np.sqrt(eval_result['overall_accuracy'] * (1 - eval_result['overall_accuracy']) / len(eval_result['y_true']))

                        st.write(f"- **Random Baseline**: {random_baseline:.4f}")
                        st.write(f"- **Z-Score**: {z_score:.2f}")
                        st.write(f"- **Statistical Significance**: {'High' if abs(z_score) > 2.58 else 'Medium' if abs(z_score) > 1.96 else 'Low'}")

                # Export results
                st.subheader("💾 Export Evaluation Results")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📥 Export Performance Metrics", key="export_performance_button"):
                        try:
                            # Create comprehensive performance report
                            report_data = {
                                'overall_accuracy': eval_result['overall_accuracy'],
                                'total_samples': len(eval_result['y_true']),
                                'correct_predictions': np.sum(eval_result['y_true'] == eval_result['y_pred']),
                                'random_baseline': 1 / len(eval_result['unique_subjects']),
                                'z_score': z_score,
                                'per_subject_metrics': performance_data
                            }

                            import json
                            report_json = json.dumps(report_data, indent=2)
                            st.download_button(
                                label="Download Performance Report (JSON)",
                                data=report_json,
                                file_name=f"performance_report_{len(eval_result['y_true'])}_samples.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            log_exception(e, context="Exporting performance report")
                            st.error("❌ Failed to export performance report")

                with col2:
                    if st.button("📥 Export Confusion Matrix", key="export_confusion_button"):
                        try:
                            # Export confusion matrix as CSV
                            cm_df = pd.DataFrame(cm,
                                                index=eval_result['unique_subjects'],
                                                columns=eval_result['unique_subjects'])
                            cm_csv = cm_df.to_csv()
                            st.download_button(
                                label="Download Confusion Matrix (CSV)",
                                data=cm_csv,
                                file_name=f"confusion_matrix_{len(eval_result['y_true'])}_samples.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            log_exception(e, context="Exporting confusion matrix")
                            st.error("❌ Failed to export confusion matrix")

            except ImportError:
                st.warning("⚠️ Scikit-learn not available for advanced metrics. Basic performance shown only.")

        st.info("💡 **Tip**: Use larger evaluation sets for more reliable performance estimates.")

    except Exception as e:
        log_exception(e, context="Performance Analysis & Confusion Matrix tab")
        st.error("❌ Error in performance analysis")


# Handle both direct execution and Streamlit exec() context
if 'main' in globals() and callable(main):
    # If main function exists, run it
    main()