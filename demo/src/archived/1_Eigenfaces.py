"""
Eigenfaces Page - PCA Face Recognition Demo

This page implements the 4-tab Eigenfaces workflow with mathematics + visual interpretation:
Tab 1: Dataset Loading & Understanding (Mathematical Concept: Data Representation & Labeling)
Tab 2: PCA Configuration & Mathematical Foundations (Mathematical Concept: Mean Centering & Covariance Matrix)
Tab 3: Eigenface Generation & Principal Components (Mathematical Concept: Eigenvalue Problem & Principal Components)
Tab 4: Facial Reconstruction & Mathematical Interpretation (Mathematical Concept: Projection & Reconstruction)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Inject Inconsolata font from Google Fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;600&display=swap" rel="stylesheet">

<style>
body, .stApp, .streamlit-container {
    font-family: 'Inconsolata', 'Courier New', monospace !important;
}

.stMarkdown, .stText, .stCaption {
    font-family: 'Inconsolata', 'Courier New', monospace !important;
}

.stCode, code, pre {
    font-family: 'Inconsolata', 'Courier New', monospace !important;
}
</style>
""", unsafe_allow_html=True)

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
    from processing.dataset_loader import DatasetLoader
    from utils.logger import log_exception, get_error_logger
    from config import DATASET_CONFIG
    logger = get_error_logger()
except ImportError as e:
    # Fallback - basic error handling without logging
    st.error(f"Error importing modules: {e}")
    st.error("Please make sure all required modules are available")
    st.stop()


def main():
    """Main function for Model Training page."""
    # Fix exec() context issue by setting __name__ global
    if '__name__' not in globals():
        globals()['__name__'] = '__main__'

    # Fix applied: Handle exec() context and absolute imports
    try:

        # Initialize session state if not exists
        if 'pca_model' not in st.session_state:
            st.session_state.pca_model = None
        if 'training_params' not in st.session_state:
            st.session_state.training_params = {}

        # Create Streamlit tabs for 4-tab workflow
        tab_names = [
            "Raw Dataset",
            "PCA Configuration & Mathematical Foundations",
            "Eigenface Generation & Principal Components",
            "Facial Reconstruction & Mathematical Interpretation"
        ]

        tabs = st.tabs(tab_names)

        # Tab 1: Dataset Loading & Understanding
        with tabs[0]:
            _dataset_loading_tab()

        # Tab 2: PCA Configuration & Mathematical Foundations
        with tabs[1]:
            _pca_mathematical_foundations_tab()

        # Tab 3: Eigenface Generation & Principal Components
        with tabs[2]:
            _eigenface_generation_tab()

        # Tab 4: Facial Reconstruction & Mathematical Interpretation
        with tabs[3]:
            _facial_reconstruction_tab()

    except Exception as e:
        log_exception(e, context="Eigenfaces main")
        st.error("❌ An error occurred in the Eigenfaces page")
        st.error("Please check the log file for detailed information.")


def _dataset_loading_tab():
    """Dataset Loading & Understanding tab with mathematical concepts and card-based UX."""
    try:
        st.subheader("📊 AT&T Dataset Stats & Configuration")
        logger.logger.info("Dataset Loading & Understanding tab loaded")

        # Initialize comprehensive dataset for cross-tab consistency
        if 'full_dataset' not in st.session_state:
            try:
                dataset_path = DATASET_CONFIG["dataset_path"]
                dataset_loader = DatasetLoader(dataset_path)
                X, y = dataset_loader.load_att_dataset()

                # Organize full dataset by subject
                full_dataset = {}
                for i in range(min(len(X), len(y))):
                    subject = y[i]
                    if subject not in full_dataset:
                        full_dataset[subject] = []
                    full_dataset[subject].append({
                        'image': X[i],
                        'index': i,
                        'flattened': X[i].flatten()
                    })

                # Sort subjects and ensure exactly 10 images per subject
                sorted_subjects = sorted(full_dataset.keys())
                organized_dataset = {}
                for subject in sorted_subjects:
                    images = full_dataset[subject]
                    if len(images) >= 10:
                        organized_dataset[subject] = images[:10]  # Take first 10 images

                st.session_state['full_dataset'] = organized_dataset
                st.session_state['sorted_subjects'] = sorted(organized_dataset.keys())

                # Also maintain demo subjects for reconstruction (first image of first 5 subjects)
                demo_subjects = []
                for i, subject in enumerate(sorted(organized_dataset.keys())[:5]):
                    demo_subjects.append({
                        'subject': subject,
                        'image': organized_dataset[subject][0]['image'],
                        'index': organized_dataset[subject][0]['index'],
                        'flattened': organized_dataset[subject][0]['flattened']
                    })

                st.session_state['demo_subjects'] = demo_subjects
                logger.logger.info(f"Initialized dataset with {len(organized_dataset)} subjects for consistency")

            except Exception as e:
                logger.logger.warning(f"Could not initialize dataset: {e}")
                st.session_state['full_dataset'] = {}
                st.session_state['sorted_subjects'] = []
                st.session_state['demo_subjects'] = []

        # Card-based dataset statistics first (before slider)
        # Calculate statistics using default training_size (will be updated after slider)
        if 'training_size' not in st.session_state:
            training_size = 6
        else:
            training_size = st.session_state['training_size']

        total_images = 400
        training_images = 40 * training_size
        test_images = 40 * (10 - training_size)
        training_ratio = training_size / 10
        flattened_size = 92 * 112
        training_matrix = f"{training_images} × {flattened_size}"
        test_matrix = f"{test_images} × {flattened_size}"

        # Single row layout with all 6 cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            # Card 1: Number of Subjects
            st.markdown("""
            <div style="background-color: #e3f2fd; padding: 12px; border-radius: 8px; border-left: 4px solid #90caf9;">
                <h6 style="margin: 0 0 5px 0; font-size: 16px;">👥 <strong>Subjects</strong></h6>
                <p style="font-size: 20px; font-weight: bold; margin: 3px 0;">40</p>
                <small style="font-size: 14px;">10 images each</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Card 2: Image Size
            st.markdown("""
            <div style="background-color: #e8f5e8; padding: 12px; border-radius: 8px; border-left: 4px solid #81c784;">
                <h6 style="margin: 0 0 5px 0; font-size: 16px;">📐 <strong>Size</strong></h6>
                <p style="font-size: 20px; font-weight: bold; margin: 3px 0;">92×112</p>
                <small style="font-size: 14px;">Grayscale</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Card 3: Vector Size
            st.markdown("""
            <div style="background-color: #fff8e1; padding: 12px; border-radius: 8px; border-left: 4px solid #ffb74d;">
                <h6 style="margin: 0 0 5px 0; font-size: 16px;">🔢 <strong>Vector</strong></h6>
                <p style="font-size: 20px; font-weight: bold; margin: 3px 0;">10,304</p>
                <small style="font-size: 14px;">Flattened</small>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # Card 4: Training Ratio
            st.markdown(f"""
            <div style="background-color: #fce4ec; padding: 12px; border-radius: 8px; border-left: 4px solid #f06292;">
                <h6 style="margin: 0 0 5px 0; font-size: 16px;">📊 <strong>Training Ratio</strong></h6>
                <p style="font-size: 20px; font-weight: bold; margin: 3px 0;">{training_ratio:.0%}</p>
                <small style="font-size: 14px;">{training_images}/{test_images}</small>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            # Card 5: Training Matrix
            st.markdown(f"""
            <div style="background-color: #f3e5f5; padding: 12px; border-radius: 8px; border-left: 4px solid #ba68c8; color: #333; text-align: center;">
                <h6 style="margin: 0 0 5px 0; font-size: 16px;">🧮 <strong>Train</strong></h6>
                <p style="font-size: 20px; font-weight: bold; margin: 3px 0;">{training_images}×10304</p>
                <small style="font-size: 14px;">Matrix</small>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            # Card 6: Test Matrix
            st.markdown(f"""
            <div style="background-color: #e0f2f1; padding: 12px; border-radius: 8px; border-left: 4px solid #4db6ac; color: #333; text-align: center;">
                <h6 style="margin: 0 0 5px 0; font-size: 16px;">🧪 <strong>Test</strong></h6>
                <p style="font-size: 20px; font-weight: bold; margin: 3px 0;">{test_images}×10304</p>
                <small style="font-size: 14px;">Matrix</small>
            </div>
            """, unsafe_allow_html=True)

        # Small training size slider below cards - left justified and compact
        st.markdown("---")
     
        # Create 2 equal columns
        colA, colB = st.columns(2)
        st.columns([0.3,0.7])  # Adjust column ratio for better alignment

        # Add content to first column
        with colA:
            training_size = st.slider(
                "Training images per subject:",
                min_value=1,
                max_value=9,
                value=training_size,
                step=1,
                help="Controls training/test split ratio",
                key="training_size_slider"
            )
    # Add content to second column  
        with colB:
            st.latex(rf"\text{{Data Matrix: }} X_{{train}} \in \mathbb{{R}}^{{n \times 10304}},\; n = 40 \times {training_size}")

        # Column Ratio Options:



        # Store training parameters
        st.session_state.training_params['training_size'] = training_size
        st.session_state['training_size'] = training_size


        # Initialize session state for pagination if not exists
        if 'training_images_page' not in st.session_state:
            st.session_state.training_images_page = 1

        # Check if dataset exists
        dataset_path = DATASET_CONFIG["dataset_path"]
        if os.path.exists(dataset_path):
            try:
                # Load sample data for first 10 subjects
                from src.processing.dataset_loader import DatasetLoader
                loader = DatasetLoader(dataset_path)
                X, y = loader.load_att_dataset()

                # Filter to first 5 subjects for display
                first_5_subjects = np.unique(y)[:5]
                mask = np.isin(y, first_5_subjects)
                X_filtered = X[mask]
                y_filtered = y[mask]

                # Display all 5 subjects with their images
                st.subheader("📸 Training Images (First 5 Subjects)")

                # Process each subject
                for subject_idx, current_subject in enumerate(first_5_subjects):
                    # Subject header
                    st.markdown(f"Subject {current_subject:02d}")

                    # Get images for this subject
                    subject_mask = y_filtered == current_subject
                    subject_images = X_filtered[subject_mask]

                    # Display images in 10-column grid (all 10 images per subject in one row)
                    cols = st.columns(10)

                    for i, img in enumerate(subject_images):
                        with cols[i]:
                            # Normalize image to [0, 255] uint8 for Streamlit display
                            img_display = (img).astype(np.uint8)
                            st.image(img_display, caption=f"{i+1}", width=100)  # Increased width for better visibility

            except Exception as e:
                st.error(f"Error loading training images: {str(e)}")
        else:
            st.warning("⚠️ Dataset not found. Please ensure the AT&T dataset is properly extracted.")

    except Exception as e:
        log_exception(e, context="Dataset Loading & Understanding tab")
        st.error("❌ Error loading dataset configuration")

    # Workflow activation section
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Confirm Dataset Configuration", type="primary", key="confirm_dataset_config"):
            # Validate dataset exists
            dataset_path = DATASET_CONFIG["dataset_path"]
            if os.path.exists(dataset_path):
                st.session_state['dataset_configured'] = True
                st.session_state['training_size'] = training_size
                st.success("✅ Dataset configuration confirmed! Please click on **Tab 2** above to continue to PCA Configuration.")
            else:
                st.error("❌ AT&T Dataset not found. Please ensure the dataset is properly extracted.")
                st.info(f"Expected path: {dataset_path}")



def _pca_mathematical_foundations_tab():
    """Covariance Matrix Computation tab with actual mathematical implementation."""
    try:
        st.header("📊 Covariance Matrix Computation")
        st.info("Mathematical Concept: Mean Centering & Covariance Matrix C = (X-μ)ᵀ(X-μ)/(n-1)")
        logger.logger.info("Covariance Matrix Computation tab loaded")

        
        # Check if dataset is ready
        if not st.session_state.get('dataset_configured', False):
            st.warning("⚠️ Please complete Dataset Configuration in Tab 1 first")
            st.info("Go to Tab 1 and click '✅ Confirm Dataset Configuration'")
            return

        # PCA configuration parameters
        st.subheader("🔧 PCA Configuration Parameters")

        # Number of components with mathematical context
        n_components = st.slider(
            "Number of Principal Components (k):",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="k = number of eigenvectors with largest eigenvalues to keep. Controls compression ratio: 10304 → k",
            key="n_components_slider"
        )

        # Distance metric selection
        distance_metric = st.selectbox(
            "Distance Metric for Recognition:",
            options=["Euclidean", "Cosine"],
            index=0,
            help="Distance function d(x,y) for measuring face similarity in eigenspace",
            key="distance_metric_select"
        )

        # Store parameters
        st.session_state.training_params['n_components'] = n_components
        st.session_state.training_params['distance_metric'] = distance_metric
        st.session_state['n_components'] = n_components  # Direct access for Tab 3
        st.session_state['distance_metric'] = distance_metric  # Direct access for Tab 3

        # Covariance Matrix Computation Interface
        st.subheader("🔬 Compute Covariance Matrix")
        st.markdown("**Step 1: Mean Centering**")
        st.latex(r"X_{centered} = X - \mu \quad \text{where} \quad \mu = \frac{1}{n}\sum_{i=1}^{n} x_i")

        st.markdown("**Step 2: Covariance Matrix Computation**")
        st.latex(r"C = \frac{1}{n-1}(X_{centered})^T \cdot X_{centered}")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧮 Compute Covariance Matrix", type="primary", key="compute_covariance"):
                with st.spinner("🔄 Computing covariance matrix from AT&T dataset..."):
                    try:
                        # Fix exec() context issue by setting __name__ global
                        if '__name__' not in globals():
                            globals()['__name__'] = '__main__'

                        # Load dataset (DatasetLoader already imported at top)
                        dataset_path = "data/ATnT"
                        dataset_loader = DatasetLoader(dataset_path)
                        X, y = dataset_loader.load_att_dataset()

                        # Get training parameters
                        training_size = st.session_state.training_params.get('training_size', 6)
                        X_train, X_test, y_train, y_test = dataset_loader.train_test_split(X, y, train_size=training_size)

                        # Flatten images for PCA
                        X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)

                        # Step 1: Mean centering
                        mean_face = np.mean(X_train_flat, axis=0)
                        X_centered = X_train_flat - mean_face

                        # Step 2: Compute covariance matrix
                        n_samples, n_features = X_centered.shape
                        # Use the trick: C = (X_centered @ X_centered.T) / (n_samples - 1)
                        # This gives a smaller matrix when n_samples < n_features
                        if n_samples < n_features:
                            cov_matrix_small = (X_centered @ X_centered.T) / (n_samples - 1)
                            cov_matrix_size = f"{n_samples} × {n_samples}"
                        else:
                            cov_matrix_small = (X_centered.T @ X_centered) / (n_samples - 1)
                            cov_matrix_size = f"{n_features} × {n_features}"

                        # Store results for next tab
                        st.session_state['X_centered'] = X_centered
                        st.session_state['mean_face'] = mean_face
                        st.session_state['cov_matrix_small'] = cov_matrix_small
                        st.session_state['cov_matrix_size'] = cov_matrix_size
                        st.session_state['n_samples'] = n_samples
                        st.session_state['n_features'] = n_features
                        st.session_state['y_train'] = y_train
                        st.session_state['X_train_original'] = X_train

                        st.success("✅ Covariance matrix computed successfully!")
                        st.info(f"📊 **Covariance Matrix Size**: {cov_matrix_size}")
                        logger.logger.info(f"Covariance matrix computed: {cov_matrix_size}")

                    except Exception as e:
                        log_exception(e, context="Covariance matrix computation")
                        st.error("❌ Covariance matrix computation failed")
                        st.error("Please check the log file for detailed information.")
                        return

        # Show computed results
        if 'cov_matrix_small' in st.session_state:
            st.subheader("📈 Covariance Matrix Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Matrix Properties:**")
                cov_matrix = st.session_state['cov_matrix_small']
                st.write(f"- **Shape**: {cov_matrix.shape}")
                st.write(f"- **Trace**: {np.trace(cov_matrix):.2f}")
                st.write(f"- **Determinant**: {np.linalg.det(cov_matrix):.2e}")
                st.write(f"- **Rank**: {np.linalg.matrix_rank(cov_matrix)}")
                st.write(f"- **Frobenius norm**: {np.linalg.norm(cov_matrix, 'fro'):.2f}")
                st.write(f"- **Symmetry check**: {np.allclose(cov_matrix, cov_matrix.T)}")

            with col2:
                st.markdown("**Statistical Properties:**")
                eigenvals = np.linalg.eigvalsh(cov_matrix)
                st.write(f"- **Min eigenvalue**: {np.min(eigenvals):.2e}")
                st.write(f"- **Max eigenvalue**: {np.max(eigenvals):.2e}")
                st.write(f"- **Eigenvalue sum**: {np.sum(eigenvals):.2f}")
                st.write(f"- **Condition number**: {np.max(eigenvals)/max(np.min(eigenvals), 1e-10):.2e}")
                st.write(f"- **Positive definite**: {np.all(eigenvals > 0)}")

            # Visualize covariance matrix structure
            st.subheader("🎨 Covariance Matrix Visualization")
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Covariance matrix heatmap
                im1 = ax1.imshow(cov_matrix, cmap='RdBu_r', aspect='auto')
                ax1.set_title(f'Covariance Matrix ({cov_matrix.shape[0]}×{cov_matrix.shape[1]})')
                ax1.set_xlabel('Component Index')
                ax1.set_ylabel('Component Index')
                plt.colorbar(im1, ax=ax1, shrink=0.6)

                # Eigenvalue spectrum
                eigenvals = np.linalg.eigvalsh(cov_matrix)
                eigenvals = eigenvals[::-1]  # Sort descending
                ax2.plot(eigenvals[:50], 'b-', linewidth=2, marker='o', markersize=3)
                ax2.set_title('Eigenvalue Spectrum (First 50)')
                ax2.set_xlabel('Eigenvalue Index')
                ax2.set_ylabel('Eigenvalue Magnitude')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Visualization error: {e}")

            # Show mean face
            st.subheader("👤 Mean Face Visualization")
            try:
                mean_face = st.session_state['mean_face'].reshape(112, 92)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(mean_face, cmap='gray')
                ax.set_title('Computed Mean Face μ')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Mean face visualization error: {e}")

            # Mathematical foundations section
            with st.expander("📚 Mathematical Foundations", expanded=False):
                st.markdown("**Covariance Matrix Properties:**")
                st.latex(r"C_{ij} = \text{cov}(X_i, X_j) = \frac{1}{n-1}\sum_{k=1}^{n}(X_{ki} - \mu_i)(X_{kj} - \mu_j)")
                st.markdown("**Key Properties:**")
                st.write("- **Symmetric**: C = Cᵀ")
                st.write("- **Positive semi-definite**: vᵀCv ≥ 0 for any vector v")
                st.write("- **Diagonal elements**: Variances of individual features")
                st.write("- **Off-diagonal elements**: Covariances between feature pairs")

                st.markdown("**Eigenvalue Problem Preview:**")
                st.latex(r"C\vec{v} = \lambda\vec{v}")
                st.markdown("Where λ are eigenvalues (variance captured) and v are eigenvectors (eigenfaces)")

    except Exception as e:
        log_exception(e, context="Covariance Matrix Computation tab")
        st.error("❌ Error loading covariance matrix computation")

    # Workflow activation section
    st.markdown("---")
    st.subheader("🚀 Activate Next Step")

    # Check if covariance matrix is computed
    cov_ready = 'cov_matrix_small' in st.session_state

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not cov_ready:
            st.warning("⚠️ Please compute the covariance matrix first")
            st.info("Click the '🧮 Compute Covariance Matrix' button above")
        else:
            if st.button("📊 Confirm Covariance Matrix Computation", type="primary", key="confirm_covariance"):
                st.session_state['covariance_computed'] = True
                st.success("✅ Covariance matrix computation confirmed! You can now proceed to Tab 3.")

    # Navigation hint
    if cov_ready:
        st.info("👆 **Next Step**: After confirming covariance matrix computation above, click 'Eigenface Generation & Principal Components' tab")
    else:
        st.info("👆 **Next Step**: Compute the covariance matrix using the button above, then confirm to proceed to Tab 3")


def _eigenface_generation_tab():
    """Eigenvalue Decomposition & Eigenface Generation tab with actual mathematical implementation."""
    try:
        st.header("🧠 Eigenvalue Decomposition & Eigenface Generation")
        st.info("Mathematical Concept: Solve C·v = λ·v to extract eigenfaces from covariance matrix")
        logger.logger.info("Eigenvalue Decomposition & Eigenface Generation tab loaded")

        
        # Check if prerequisites are met
        dataset_ready = st.session_state.get('dataset_configured', False)
        cov_ready = st.session_state.get('covariance_computed', False)

        if not dataset_ready or not cov_ready:
            st.warning("⚠️ Please complete the prerequisite steps first:")
            if not dataset_ready:
                st.error("❌ Tab 1: Click '✅ Confirm Dataset Configuration' button")
            if not cov_ready:
                st.error("❌ Tab 2: Compute and confirm covariance matrix")
            return

        # Eigenvalue decomposition interface
        st.subheader("🔬 Perform Eigenvalue Decomposition")
        st.markdown("**Step 1: Solve Eigenvalue Problem**")
        st.latex(r"C\vec{v} = \lambda\vec{v}")
        st.markdown("*Where C is the covariance matrix from Tab 2, λ are eigenvalues, and v are eigenvectors (eigenfaces)*")

        st.markdown("**Step 2: Extract Principal Components**")
        st.latex(r"W = [\vec{v}_1, \vec{v}_2, ..., \vec{v}_k] \quad \text{where} \quad \lambda_1 \geq \lambda_2 \geq ... \geq \lambda_k")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚡ Perform Eigenvalue Decomposition", type="primary", key="perform_eigen_decomposition"):
                with st.spinner("🔄 Solving eigenvalue problem and extracting eigenfaces..."):
                    try:
                        # Get pre-computed data from Tab 2
                        X_centered = st.session_state['X_centered']
                        cov_matrix_small = st.session_state['cov_matrix_small']
                        mean_face = st.session_state['mean_face']
                        n_samples = st.session_state['n_samples']
                        n_features = st.session_state['n_features']
                        n_components = st.session_state['n_components']
                        y_train = st.session_state['y_train']
                        X_train_original = st.session_state['X_train_original']

                        # Step 1: Perform eigenvalue decomposition on the smaller covariance matrix
                        from scipy.linalg import eigh
                        eigenvalues_small, eigenvectors_small = eigh(cov_matrix_small)

                        # Step 2: Sort eigenvalues and eigenvectors in descending order
                        idx = np.argsort(eigenvalues_small)[::-1]
                        eigenvalues_small = eigenvalues_small[idx]
                        eigenvectors_small = eigenvectors_small[:, idx]

                        # Step 3: Convert to actual eigenvectors of the original covariance matrix
                        if n_samples < n_features:
                            # Case: We computed (X @ X.T), need to convert eigenvectors
                            actual_eigenvectors = (X_centered.T @ eigenvectors_small) / np.sqrt(eigenvalues_small * (n_samples - 1))
                        else:
                            # Case: We computed (X.T @ X), eigenvectors are already in correct space
                            actual_eigenvectors = eigenvectors_small

                        # Step 4: Normalize eigenvectors
                        actual_eigenvectors = actual_eigenvectors / np.linalg.norm(actual_eigenvectors, axis=0)

                        # Step 5: Select top k components
                        k = min(n_components, actual_eigenvectors.shape[1])
                        eigenfaces = actual_eigenvectors[:, :k]
                        selected_eigenvalues = eigenvalues_small[:k]

                        # Step 6: Compute projected data
                        X_projected = X_centered @ eigenfaces

                        # Store results for next tab
                        st.session_state['eigenfaces'] = eigenfaces
                        st.session_state['eigenvalues'] = selected_eigenvalues
                        st.session_state['X_projected'] = X_projected
                        st.session_state['X_centered'] = X_centered
                        st.session_state['mean_face'] = mean_face
                        st.session_state['n_components'] = k
                        st.session_state['y_train'] = y_train
                        st.session_state['X_train_original'] = X_train_original

                        # Compute explained variance ratio
                        total_variance = np.sum(eigenvalues_small)
                        explained_variance_ratio = selected_eigenvalues / total_variance
                        cumulative_variance = np.cumsum(explained_variance_ratio)

                        st.session_state['explained_variance_ratio'] = explained_variance_ratio
                        st.session_state['cumulative_variance'] = cumulative_variance

                        st.success("✅ Eigenvalue decomposition completed successfully!")
                        st.info(f"🎯 **Extracted {k} eigenfaces** from covariance matrix")
                        logger.logger.info(f"Eigenvalue decomposition completed: {k} eigenfaces extracted")

                    except Exception as e:
                        log_exception(e, context="Eigenvalue decomposition")
                        st.error("❌ Eigenvalue decomposition failed")
                        st.error("Please check the log file for detailed information.")
                        return

        # Show eigenface analysis results
        if 'eigenfaces' in st.session_state:
            st.subheader("🎭 Eigenface Analysis & Visualization")

            eigenfaces = st.session_state['eigenfaces']
            eigenvalues = st.session_state['eigenvalues']
            explained_variance_ratio = st.session_state['explained_variance_ratio']
            cumulative_variance = st.session_state['cumulative_variance']

            # Eigenvalue analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Eigenvalue Statistics:**")
                st.write(f"- **Number of eigenfaces**: {len(eigenvalues)}")
                st.write(f"- **Largest eigenvalue**: {eigenvalues[0]:.2e}")
                st.write(f"- **Smallest eigenvalue**: {eigenvalues[-1]:.2e}")
                st.write(f"- **Eigenvalue sum**: {np.sum(eigenvalues):.2f}")
                st.write(f"- **Condition number**: {eigenvalues[0]/eigenvalues[-1]:.2e}")

            with col2:
                st.markdown("**Variance Explained:**")
                st.write(f"- **Top component variance**: {explained_variance_ratio[0]*100:.2f}%")
                st.write(f"- **Total variance captured**: {np.sum(explained_variance_ratio)*100:.2f}%")
                st.write(f"- **95% variance components**: {np.argmax(cumulative_variance >= 0.95) + 1}")
                st.write(f"- **99% variance components**: {np.argmax(cumulative_variance >= 0.99) + 1}")

            # Mathematical interpretation of eigenfaces
            st.subheader("🧮 Understanding Eigenfaces")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px;">
                    <h5 style="margin-top: 0;">📐 Mathematical Interpretation</h5>
                    <p style="margin-bottom: 10px;"><strong>An eigenface is a principal component vector that represents directions of maximum variance in the face space.</strong></p>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>Eigenvector</strong>: Unit vector in face space</li>
                        <li><strong>Eigenvalue</strong>: Amount of variance captured</li>
                        <li><strong>Orthogonal</strong>: Each eigenface is perpendicular to others</li>
                        <li><strong>Basis</strong>: Forms coordinate system for faces</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                # Mathematical formula
                st.latex(r"C \cdot v_i = \lambda_i \cdot v_i")
                st.markdown(r"*C: Covariance matrix, $v_i$: ith eigenvector (eigenface), $\lambda_i$: ith eigenvalue*")

            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px;">
                    <h5 style="margin-top: 0;">🎨 Visual Interpretation</h5>
                    <p style="margin-bottom: 10px;"><strong>Each eigenface captures specific facial patterns that vary across all subjects:</strong></p>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>Eigenface 1</strong>: Overall face shape & lighting</li>
                        <li><strong>Eigenface 2</strong>: Horizontal features (eyes, mouth)</li>
                        <li><strong>Eigenface 3</strong>: Vertical features (nose, symmetry)</li>
                        <li><strong>Higher eigenfaces</strong>: Fine details & textures</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                # Visual representation concept
                st.markdown("*Eigenfaces look like ghostly faces because they represent statistical patterns, not actual individuals*")

            st.markdown("---")

            # Dataset-wide clarification
            st.markdown("""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; margin: 15px 0;">
                <strong>🔍 Important:</strong> Eigenfaces are generated from <strong>ALL training images (240+ faces from 40 subjects)</strong>,
                not from individual subjects. Each eigenface represents universal facial patterns shared across the entire dataset.
            </div>
            """, unsafe_allow_html=True)

            # Visualize top eigenfaces
            st.subheader("🖼️ Top Eigenfaces Visualization")
            n_faces_to_show = min(16, len(eigenvalues))
            n_cols = 4
            n_rows = (n_faces_to_show + n_cols - 1) // n_cols

            try:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
                fig.suptitle(f'Top {n_faces_to_show} Eigenfaces (Ordered by Eigenvalue)', fontsize=14, fontweight='bold')

                if n_rows == 1:
                    axes = axes.reshape(1, -1)

                for i in range(n_faces_to_show):
                    row, col = i // n_cols, i % n_cols
                    eigenface_img = eigenfaces[:, i].reshape(112, 92)

                    # Normalize for visualization
                    eigenface_img = (eigenface_img - eigenface_img.min()) / (eigenface_img.max() - eigenface_img.min())

                    axes[row, col].imshow(eigenface_img, cmap='gray')
                    axes[row, col].set_title(f'Eigenface {i+1}\nλ = {eigenvalues[i]:.2e}\nVar: {explained_variance_ratio[i]*100:.1f}%',
                                           fontsize=8)
                    axes[row, col].axis('off')

                # Hide unused subplots
                for i in range(n_faces_to_show, n_rows * n_cols):
                    row, col = i // n_cols, i % n_cols
                    axes[row, col].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Eigenface visualization error: {e}")

            # Eigenvalue spectrum plot
            st.subheader("📊 Eigenvalue Spectrum Analysis")
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Eigenvalue magnitude
                ax1.plot(eigenvalues[:50], 'b-', linewidth=2, marker='o', markersize=4)
                ax1.set_title('Eigenvalue Magnitude Spectrum (First 50)')
                ax1.set_xlabel('Eigenvalue Index')
                ax1.set_ylabel('Eigenvalue (log scale)')
                ax1.set_yscale('log')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=eigenvalues[-1] * 10, color='r', linestyle='--', alpha=0.7, label='10× smallest eigenvalue')
                ax1.legend()

                # Cumulative variance
                ax2.plot(cumulative_variance, 'g-', linewidth=2, marker='s', markersize=4)
                ax2.set_title('Cumulative Variance Explained')
                ax2.set_xlabel('Number of Components')
                ax2.set_ylabel('Cumulative Variance Ratio')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='90% variance')
                ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% variance')
                ax2.axhline(y=0.99, color='purple', linestyle='--', alpha=0.7, label='99% variance')
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Spectrum analysis error: {e}")

            # Mathematical foundations section
            with st.expander("📚 Mathematical Foundations", expanded=False):
                st.markdown("**Eigenvalue Problem Solution:**")
                st.latex(r"C\vec{v}_i = \lambda_i \vec{v}_i")
                st.markdown("Where each eigenface $\vec{v}_i$ captures a principal component of facial variation")

                st.markdown("**Orthogonality Property:**")
                st.latex(r"\vec{v}_i \cdot \vec{v}_j = 0 \quad \text{for} \quad i \neq j")
                st.markdown("Eigenfaces are orthogonal vectors in the high-dimensional face space")

                st.markdown("**Projection Formula:**")
                st.latex(r"\vec{y} = W^T (x - \mu) \quad \text{where} \quad W = [\vec{v}_1, \vec{v}_2, ..., \vec{v}_k]")
                st.markdown("Projects any face onto the eigenface subspace for recognition")

    except Exception as e:
        log_exception(e, context="Eigenvalue Decomposition & Eigenface Generation tab")
        st.error("❌ Error loading eigenface generation")

    # Workflow activation section
    st.markdown("---")
    st.subheader("🚀 Activate Next Step")

    # Check if eigenfaces are computed
    eigenfaces_ready = 'eigenfaces' in st.session_state

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not eigenfaces_ready:
            st.warning("⚠️ Please perform eigenvalue decomposition first")
            st.info("Click the '⚡ Perform Eigenvalue Decomposition' button above")
        else:
            if st.button("🧠 Confirm Eigenface Generation", type="primary", key="confirm_eigenface_generation"):
                st.session_state['eigenfaces_generated'] = True
                st.success("✅ Eigenface generation confirmed! You can now proceed to Tab 4.")

    # Navigation hint
    if eigenfaces_ready:
        st.info("👆 **Next Step**: After confirming eigenface generation above, click 'Facial Reconstruction & Mathematical Interpretation' tab")
    else:
        st.info("👆 **Next Step**: Perform eigenvalue decomposition using the button above, then confirm to proceed to Tab 4")


def _facial_reconstruction_tab():
    """Facial Reconstruction & Mathematical Interpretation tab with actual projection and reconstruction."""
    try:
        st.header("🔄 Facial Reconstruction & Mathematical Interpretation")
        st.info("Mathematical Concept: Projection y = Wᵀ(x-μ) and Reconstruction x̂ = μ + Wy")
        logger.logger.info("Facial Reconstruction & Mathematical Interpretation tab loaded")

        # Check if prerequisites are met
        eigenfaces_ready = st.session_state.get('eigenfaces_generated', False)

        if not eigenfaces_ready:
            st.warning("⚠️ Please complete the prerequisite steps first:")
            st.error("❌ Tab 3: Click '🧠 Confirm Eigenface Generation' button")
            return

        # Get stored data from previous tabs
        eigenfaces = st.session_state['eigenfaces']
        X_centered = st.session_state['X_centered']
        mean_face = st.session_state['mean_face']
        X_projected = st.session_state['X_projected']
        y_train = st.session_state['y_train']
        X_train_original = st.session_state['X_train_original']
        cumulative_variance = st.session_state['cumulative_variance']

        st.subheader("🎭 Facial Reconstruction Analysis")

        # Number of components for reconstruction analysis
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            n_reconstruction_components = st.slider(
                "Number of Components for Reconstruction:",
                min_value=1,
                max_value=min(eigenfaces.shape[1], 50),
                value=min(eigenfaces.shape[1], 20),
                step=1,
                help="Number of eigenfaces to use for reconstruction"
            )

        # Subject selection for consistent reconstruction
        st.subheader("👤 Subject Selection for Reconstruction")
        demo_subjects = st.session_state.get('demo_subjects', [])

        if demo_subjects:
            subject_options = [f"Subject {subj['subject']}" for subj in demo_subjects]
            selected_subject_name = st.selectbox(
                "Select subject for reconstruction:",
                options=subject_options,
                index=0,
                help="Choose the same subject shown in Dataset Preview for consistency"
            )
            selected_index = subject_options.index(selected_subject_name)
        else:
            st.warning("⚠️ No demo subjects available. Please ensure dataset is loaded in Tab 1.")
            selected_index = 0

        # Perform reconstruction
        st.subheader("🔄 Test Facial Reconstruction")

        try:
            # Use selected demo subject for reconstruction
            demo_subjects = st.session_state.get('demo_subjects', [])
            if demo_subjects and selected_index < len(demo_subjects):
                # Use the selected demo subject
                selected_demo = demo_subjects[selected_index]
                sample_idx = selected_demo['index']
                selected_subject = selected_demo['subject']
                original_face = X_centered[sample_idx]
                original_with_mean = original_face + mean_face

                # Show which subject is being used
                st.info(f"🎯 **Using Subject {selected_subject}** for reconstruction (consistent with Dataset Preview)")
            else:
                # Fallback to first sample
                sample_idx = 0
                selected_subject = "Unknown"
                original_face = X_centered[sample_idx]
                original_with_mean = original_face + mean_face
                st.warning("⚠️ Using fallback sample - demo subjects not available")

            # Project to eigenface space
            projected = X_projected[sample_idx][:n_reconstruction_components]

            # Reconstruct using selected eigenfaces
            eigenfaces_subset = eigenfaces[:, :n_reconstruction_components]
            reconstructed_centered = eigenfaces_subset @ projected
            reconstructed_face = reconstructed_centered + mean_face

            # Calculate reconstruction error
            reconstruction_error = np.mean((original_face - reconstructed_centered) ** 2)
            variance_preserved = cumulative_variance[n_reconstruction_components - 1] if n_reconstruction_components <= len(cumulative_variance) else 1.0

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Original Face**")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.imshow(original_with_mean.reshape(112, 92), cmap='gray')
                ax1.set_title('Original Face')
                ax1.axis('off')
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                st.markdown(f"**Reconstructed Face**\n({n_reconstruction_components} components)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.imshow(reconstructed_face.reshape(112, 92), cmap='gray')
                ax2.set_title(f'Reconstructed Face\nVariance: {variance_preserved*100:.1f}%')
                ax2.axis('off')
                st.pyplot(fig2)
                plt.close(fig2)

            with col3:
                st.markdown("**Reconstruction Error**")
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                error_face = np.abs(original_face - reconstructed_centered)
                ax3.imshow(error_face.reshape(112, 92), cmap='hot')
                ax3.set_title(f'MSE: {reconstruction_error:.2f}')
                ax3.axis('off')
                st.pyplot(fig3)
                plt.close(fig3)

            # Reconstruction metrics
            st.subheader("📊 Reconstruction Quality Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Compression Metrics:**")
                original_dims = 112 * 92  # 10304 pixels
                compressed_dims = n_reconstruction_components
                compression_ratio = compressed_dims / original_dims

                st.write(f"- **Original dimensions**: {original_dims} pixels")
                st.write(f"- **Compressed dimensions**: {compressed_dims} components")
                st.write(f"- **Compression ratio**: {compression_ratio:.6f}")
                st.write(f"- **Space reduction**: {(1 - compression_ratio)*100:.2f}%")
                st.write(f"- **Reconstruction MSE**: {reconstruction_error:.4f}")

            with col2:
                st.markdown("**Quality Metrics:**")
                st.write(f"- **Variance preserved**: {variance_preserved*100:.1f}%")
                st.write(f"- **Signal-to-noise ratio**: {10 * np.log10(np.var(original_face) / max(reconstruction_error, 1e-10)):.2f} dB")
                st.write(f"- **Peak signal-to-noise**: {10 * np.log10(1.0 / max(reconstruction_error / (np.max(original_face) - np.min(original_face)), 1e-10)):.2f} dB")
                st.write(f"- **Structural similarity**: {(1 - reconstruction_error / np.var(original_face)) * 100:.1f}%")

            # Mathematical foundations section
            with st.expander("📚 Mathematical Foundations", expanded=False):
                st.markdown("**Projection to Eigenface Space:**")
                st.latex(r"\vec{y} = W^T (x - \mu)")
                st.markdown("*Where W contains the selected eigenfaces and μ is the mean face*")

                st.markdown("**Reconstruction Formula:**")
                st.latex(r"\hat{x} = \mu + W \vec{y}")
                st.markdown("*Reconstructs the face by projecting back to the original space*")

                st.markdown("**Reconstruction Error:**")
                st.latex(r"E = \frac{1}{d} \sum_{i=1}^{d} (x_i - \hat{x}_i)^2")
                st.markdown("*Mean squared error between original and reconstructed pixels*")

        except Exception as e:
            log_exception(e, context="Facial reconstruction analysis")
            st.error(f"❌ Reconstruction analysis failed: {e}")

    except Exception as e:
        log_exception(e, context="Facial Reconstruction & Mathematical Interpretation tab")
        st.error("❌ Error loading facial reconstruction")




# Handle both direct execution and Streamlit exec() context
if 'main' in globals() and callable(main):
    # If main function exists, run it
    main()
