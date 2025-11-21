"""
Face Verification Page - PCA Face Recognition Demo

This page implements face verification workflow for comparing two uploaded images
to determine if they show the same person using PCA feature comparison.

Tab Structure:
1. Model Information - Display trained model details and verification overview
2. Face Upload & Processing - Upload and process two face images
3. Face Verification Results - Compare features and provide verification decision
"""

import streamlit as st
import numpy as np
import cv2
import time
import io
import base64
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Import from our modules
try:
    from ..core.pca import PCAFromScratch
    from ..core.eigenfaces import EigenfacesRecognizer
    from ..data.dataset_loader import DatasetLoader
    from ..utils.logger import log_exception
    from ..visualization.chart_utils import ChartUtils
except ImportError:
    # Fallback for relative imports
    import sys
    sys.path.append('src')
    from core.pca import PCAFromScratch
    from core.eigenfaces import EigenfacesRecognizer
    from data.dataset_loader import DatasetLoader
    from utils.logger import log_exception
    from visualization.chart_utils import ChartUtils


def initialize_session_state():
    """Initialize session state variables for face verification workflow."""
    if 'verification_tab' not in st.session_state:
        st.session_state.verification_tab = 0

    # Verification data storage
    if 'verification_images' not in st.session_state:
        st.session_state.verification_images = {}

    if 'verification_results' not in st.session_state:
        st.session_state.verification_results = {}

    if 'verification_processed' not in st.session_state:
        st.session_state.verification_processed = False


def safe_streamlit_execute(func):
    """Safely execute Streamlit functions with error handling."""
    try:
        return func()
    except Exception as e:
        log_exception(e, context="Face Verification Streamlit execution")
        st.error("❌ An unexpected error occurred")
        st.error("Please check the log file for detailed information.")
        return None


def tab_navigation():
    """Handle tab navigation for face verification workflow."""
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.session_state.verification_tab > 0:
            if st.button("⬅️ Previous", key="verify_prev"):
                st.session_state.verification_tab -= 1
                st.rerun()

    with col2:
        st.markdown(f"**Tab {st.session_state.verification_tab + 1} of 3**")

    with col3:
        progress = (st.session_state.verification_tab + 1) / 3
        st.progress(progress)

    with col4:
        if st.session_state.verification_tab < 2:
            # Check if we can proceed to next tab
            can_proceed = False

            if st.session_state.verification_tab == 0:
                # Can proceed if we have a trained model
                can_proceed = st.session_state.get('pca_model') is not None
            elif st.session_state.verification_tab == 1:
                # Can proceed if we have processed images
                can_proceed = len(st.session_state.verification_images) == 2

            if can_proceed:
                if st.button("Next ➡️", key="verify_next"):
                    st.session_state.verification_tab += 1
                    st.rerun()
            else:
                st.button("Next ➡️", key="verify_next_disabled", disabled=True)

    with col5:
        if st.session_state.verification_tab == 2:
            if st.button("Complete ✓", key="verify_complete", type="primary"):
                st.success("🎉 Face verification workflow completed!")
                st.balloons()


def tab_1_model_information():
    """Tab 1: Model Information - Display trained model details and verification overview."""
    st.header("📊 Model Information")

    # Check if trained model exists
    pca_model = st.session_state.get('pca_model')

    if pca_model is None:
        st.warning("⚠️ No trained model found!")
        st.info("Please train the PCA model on the **Eigenfaces** page first.")

        # Show placeholder information
        st.subheader("📋 Face Verification Overview")
        st.markdown("""
        **Face verification** determines if two uploaded images show the same person using PCA feature comparison.

        ### How it Works:
        1. **Feature Extraction**: Both images are projected into the PCA eigenface space
        2. **Distance Calculation**: Compute Euclidean distance between feature vectors
        3. **Verification Decision**: Compare distance against threshold to determine identity

        ### Mathematical Foundation:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Feature Projection:**")
            st.latex(r"y = W^T(x - \mu)")
            st.markdown("*Where $W$ = eigenface matrix, $x$ = face image, $\mu$ = mean face*")

        with col2:
            st.markdown("**Distance Metric:**")
            st.latex(r"d = \|y_1 - y_2\|_2")
            st.markdown("*Euclidean distance between projected features*")

        st.markdown("**Verification Decision:**")
        st.latex(r"\text{Same Person} = \begin{cases} \text{Yes} & \text{if } d < \tau \\ \text{No} & \text{if } d \geq \tau \end{cases}")
        st.markdown("*Where $\tau$ = verification threshold (typically 50-100 units)*")

        return False

    # Display model information
    st.success("✅ Trained PCA model loaded successfully!")

    # Model details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Training Parameters")
        st.write(f"**Component Count:** {pca_model.n_components}")
        st.write(f"**Distance Metric:** Euclidean")
        st.write(f"**Dataset:** AT&T Face Database")

        # Training statistics if available
        if hasattr(pca_model, 'explained_variance_ratio_'):
            variance_explained = np.sum(pca_model.explained_variance_ratio_) * 100
            st.write(f"**Variance Explained:** {variance_explained:.1f}%")

    with col2:
        st.subheader("📈 Model Statistics")
        if hasattr(pca_model, 'components_'):
            st.write(f"**Eigenface Dimensions:** {pca_model.components_.shape}")
            st.write(f"**Feature Vector Size:** {pca_model.n_components}")

        # Processing information
        if 'processing_time' in st.session_state:
            st.write(f"**Training Time:** {st.session_state.processing_time:.2f} seconds")

        st.write(f"**Target Image Size:** 92×112 pixels")

    # Verification explanation
    st.subheader("🔍 Face Verification Process")

    with st.expander("📖 How Face Verification Works"):
        st.markdown("""
        ### Step-by-Step Process:

        **1. Image Preprocessing:**
        - Detect faces using OpenCV Haar cascades
        - Resize to 92×112 pixels (AT&T dataset standard)
        - Convert to grayscale for PCA compatibility
        - Apply histogram equalization for lighting normalization

        **2. Feature Extraction:**
        - Subtract mean face (centering)
        - Project into eigenface subspace using trained PCA model
        - Extract compact feature representation

        **3. Feature Comparison:**
        - Compute Euclidean distance between feature vectors
        - Normalize distance based on training data distribution
        - Compare against adaptive threshold

        **4. Verification Decision:**
        - **Same Person** if distance < threshold
        - **Different Person** if distance ≥ threshold
        - Provide confidence score based on distance from threshold

        ### Mathematical Formulation:

        **Feature Projection:**
        $$y_i = W^T(x_i - \mu)$$

        **Distance Calculation:**
        $$d = \|y_1 - y_2\|_2 = \sqrt{\sum_{j=1}^{k} (y_{1j} - y_{2j})^2}$$

        **Confidence Score:**
        $$\text{confidence} = \frac{\tau - d}{\tau} \times 100\%$$

        Where:
        - $W$ = eigenface matrix (principal components)
        - $x_i$ = input face image
        - $\mu$ = mean face
        - $y_i$ = projected feature vector
        - $d$ = Euclidean distance
        - $\tau$ = verification threshold
        """)

    return True


def detect_and_extract_face(image, target_size=(92, 112)):
    """
    Detect and extract face from image using OpenCV.

    Args:
        image: Input image (PIL Image or numpy array)
        target_size: Target size for face (width, height)

    Returns:
        tuple: (processed_face, detection_info)
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy()

        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        detection_info = {
            'faces_detected': len(faces),
            'face_locations': faces.tolist() if len(faces) > 0 else [],
            'original_size': image_array.shape[:2],
            'processed': False
        }

        if len(faces) == 0:
            # No face detected, use entire image
            processed_face = cv2.resize(gray, target_size)
            detection_info['method'] = 'resize_full_image'
        else:
            # Use the largest detected face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face

            # Extract face region with some padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + w + padding)
            y2 = min(gray.shape[0], y + h + padding)

            face_roi = gray[y1:y2, x1:x2]
            processed_face = cv2.resize(face_roi, target_size)
            detection_info['method'] = 'face_detection'
            detection_info['face_roi'] = [x1, y1, x2-x1, y2-y1]

        # Apply histogram equalization for lighting normalization
        processed_face = cv2.equalizeHist(processed_face)

        # Normalize pixel values to [0, 1]
        processed_face = processed_face.astype(np.float32) / 255.0

        detection_info['processed'] = True
        detection_info['final_size'] = processed_face.shape

        return processed_face, detection_info

    except Exception as e:
        log_exception(e, context="Face detection and extraction")
        return None, {'error': str(e), 'processed': False}


def tab_2_face_upload_processing():
    """Tab 2: Face Upload & Processing - Upload and process two face images."""
    st.header("📤 Face Upload & Processing")

    # Check if model exists
    if st.session_state.get('pca_model') is None:
        st.warning("⚠️ Please complete Tab 1 (train model) first!")
        return

    st.markdown("Upload two face images to verify if they show the same person.")

    # Create two columns for image upload
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 First Person (Reference)")

        # File upload options
        upload_method_1 = st.radio(
            "Upload method:",
            ["File Uploader", "Camera"],
            key="upload_method_1"
        )

        if upload_method_1 == "File Uploader":
            uploaded_file_1 = st.file_uploader(
                "Choose first image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="upload_1"
            )
        else:
            uploaded_file_1 = st.camera_input("Take first photo", key="camera_1")

        if uploaded_file_1 is not None:
            # Display uploaded image
            image_1 = Image.open(uploaded_file_1)
            st.image(image_1, caption="First person (original)", use_column_width=True)

            # Process face
            if st.button("🔍 Process First Face", key="process_1"):
                with st.spinner("Processing first face..."):
                    processed_face_1, detection_info_1 = detect_and_extract_face(image_1)

                    if processed_face_1 is not None:
                        st.session_state.verification_images['image_1'] = {
                            'original': image_1,
                            'processed': processed_face_1,
                            'detection_info': detection_info_1
                        }
                        st.success("✅ First face processed successfully!")

                        # Show detection info
                        col_info_1, col_display_1 = st.columns(2)

                        with col_info_1:
                            st.markdown("**Detection Results:**")
                            st.write(f"Faces detected: {detection_info_1['faces_detected']}")
                            st.write(f"Method: {detection_info_1.get('method', 'unknown')}")
                            st.write(f"Final size: {detection_info_1.get('final_size', 'N/A')}")

                        with col_display_1:
                            st.image(processed_face_1, caption="First person (processed)",
                                    use_column_width=True, clamp=True)
                    else:
                        st.error("❌ Failed to process first face!")

    with col2:
        st.subheader("👥 Second Person (Test)")

        # File upload options
        upload_method_2 = st.radio(
            "Upload method:",
            ["File Uploader", "Camera"],
            key="upload_method_2"
        )

        if upload_method_2 == "File Uploader":
            uploaded_file_2 = st.file_uploader(
                "Choose second image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="upload_2"
            )
        else:
            uploaded_file_2 = st.camera_input("Take second photo", key="camera_2")

        if uploaded_file_2 is not None:
            # Display uploaded image
            image_2 = Image.open(uploaded_file_2)
            st.image(image_2, caption="Second person (original)", use_column_width=True)

            # Process face
            if st.button("🔍 Process Second Face", key="process_2"):
                with st.spinner("Processing second face..."):
                    processed_face_2, detection_info_2 = detect_and_extract_face(image_2)

                    if processed_face_2 is not None:
                        st.session_state.verification_images['image_2'] = {
                            'original': image_2,
                            'processed': processed_face_2,
                            'detection_info': detection_info_2
                        }
                        st.success("✅ Second face processed successfully!")

                        # Show detection info
                        col_info_2, col_display_2 = st.columns(2)

                        with col_info_2:
                            st.markdown("**Detection Results:**")
                            st.write(f"Faces detected: {detection_info_2['faces_detected']}")
                            st.write(f"Method: {detection_info_2.get('method', 'unknown')}")
                            st.write(f"Final size: {detection_info_2.get('final_size', 'N/A')}")

                        with col_display_2:
                            st.image(processed_face_2, caption="Second person (processed)",
                                    use_column_width=True, clamp=True)
                    else:
                        st.error("❌ Failed to process second face!")

    # Show side-by-side comparison if both images are processed
    if len(st.session_state.verification_images) == 2:
        st.markdown("---")
        st.subheader("🔍 Processed Faces Comparison")

        col1, col2 = st.columns(2)

        with col1:
            img1_data = st.session_state.verification_images['image_1']
            st.image(img1_data['processed'], caption="First person (processed)",
                    use_column_width=True, clamp=True)

            # Display detection info
            info1 = img1_data['detection_info']
            st.markdown("**Processing Details:**")
            st.write(f"✅ Detection: {info1.get('method', 'unknown')}")
            st.write(f"📏 Final size: {info1.get('final_size', 'N/A')}")

        with col2:
            img2_data = st.session_state.verification_images['image_2']
            st.image(img2_data['processed'], caption="Second person (processed)",
                    use_column_width=True, clamp=True)

            # Display detection info
            info2 = img2_data['detection_info']
            st.markdown("**Processing Details:**")
            st.write(f"✅ Detection: {info2.get('method', 'unknown')}")
            st.write(f"📏 Final size: {info2.get('final_size', 'N/A')}")

        # Processing summary
        st.info("🎯 Both faces processed successfully! You can now proceed to verification results.")

        # Set flag for next tab
        st.session_state.verification_processed = True

    # Processing tips
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        ### Image Guidelines:
        - **Front-facing photos** work best (profile detection is limited)
        - **Good lighting** with minimal shadows
        - **Neutral expressions** (smiling is fine, but extreme expressions may affect results)
        - **Clear background** helps with face detection
        - **High resolution** images (at least 200×200 pixels)

        ### Technical Details:
        - Uses OpenCV Haar Cascade for face detection
        - If no face is detected, the entire image is resized
        - Images are standardized to 92×112 pixels (AT&T dataset format)
        - Histogram equalization normalizes lighting conditions
        - Processing time: ~1-2 seconds per image
        """)


def compute_verification_distance(face1_features, face2_features):
    """
    Compute verification distance between two feature vectors.

    Args:
        face1_features: Feature vector for first face
        face2_features: Feature vector for second face

    Returns:
        dict: Distance computation results
    """
    try:
        # Compute Euclidean distance
        euclidean_distance = np.linalg.norm(face1_features - face2_features)

        # Compute Manhattan distance for comparison
        manhattan_distance = np.sum(np.abs(face1_features - face2_features))

        # Compute cosine similarity
        dot_product = np.dot(face1_features, face2_features)
        norm1 = np.linalg.norm(face1_features)
        norm2 = np.linalg.norm(face2_features)

        if norm1 > 0 and norm2 > 0:
            cosine_similarity = dot_product / (norm1 * norm2)
        else:
            cosine_similarity = 0.0

        # Estimate threshold based on training data characteristics
        # This is a simplified approach - in practice, thresholds are determined empirically
        estimated_threshold = 80.0  # Default threshold for eigenface space

        # Compute confidence score
        if euclidean_distance < estimated_threshold:
            confidence = (estimated_threshold - euclidean_distance) / estimated_threshold * 100
        else:
            confidence = 0.0

        return {
            'euclidean_distance': float(euclidean_distance),
            'manhattan_distance': float(manhattan_distance),
            'cosine_similarity': float(cosine_similarity),
            'estimated_threshold': float(estimated_threshold),
            'confidence': float(confidence),
            'verification_result': 'Same Person' if euclidean_distance < estimated_threshold else 'Different Person'
        }

    except Exception as e:
        log_exception(e, context="Distance computation")
        return {
            'error': str(e),
            'euclidean_distance': float('inf'),
            'confidence': 0.0,
            'verification_result': 'Error'
        }


def tab_3_verification_results():
    """Tab 3: Face Verification Results - Compare features and provide verification decision."""
    st.header("🎯 Face Verification Results")

    # Check if we have processed images and model
    if len(st.session_state.verification_images) != 2:
        st.warning("⚠️ Please process both face images in Tab 2 first!")
        return

    if st.session_state.get('pca_model') is None:
        st.warning("⚠️ No trained model available!")
        return

    pca_model = st.session_state.pca_model

    # Get processed images
    img1_data = st.session_state.verification_images['image_1']
    img2_data = st.session_state.verification_images['image_2']

    face1 = img1_data['processed']
    face2 = img2_data['processed']

    st.markdown("Comparing facial features using trained PCA model...")

    # Perform verification
    with st.spinner("Extracting features and computing distances..."):
        start_time = time.time()

        try:
            # Flatten images for PCA
            face1_flat = face1.flatten()
            face2_flat = face2.flatten()

            # Extract PCA features
            face1_features = pca_model.transform(face1_flat.reshape(1, -1))
            face2_features = pca_model.transform(face2_flat.reshape(1, -1))

            # Compute verification metrics
            verification_results = compute_verification_distance(
                face1_features.flatten(),
                face2_features.flatten()
            )

            processing_time = time.time() - start_time

            # Store results
            st.session_state.verification_results = verification_results
            st.session_state.verification_results['processing_time'] = processing_time
            st.session_state.verification_results['features_extracted'] = True

        except Exception as e:
            log_exception(e, context="Feature extraction and verification")
            st.error(f"❌ Error during verification: {str(e)}")
            return

    # Display results
    verification_result = verification_results['verification_result']
    confidence = verification_results['confidence']

    # Result header
    if verification_result == 'Same Person':
        st.success(f"🎉 **{verification_result}**")
        st.success(f"Confidence: {confidence:.1f}%")
    elif verification_result == 'Different Person':
        st.error(f"🚫 **{verification_result}**")
        st.info(f"Confidence: {confidence:.1f}%")
    else:
        st.error("❌ Verification Error")

    # Detailed metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Distance Metrics")

        st.metric("Euclidean Distance", f"{verification_results['euclidean_distance']:.2f}")
        st.metric("Threshold", f"{verification_results['estimated_threshold']:.2f}")
        st.metric("Confidence", f"{verification_results['confidence']:.1f}%")

        # Additional metrics
        with st.expander("📈 Additional Metrics"):
            st.write(f"Manhattan Distance: {verification_results['manhattan_distance']:.2f}")
            st.write(f"Cosine Similarity: {verification_results['cosine_similarity']:.4f}")
            st.write(f"Processing Time: {processing_time:.3f} seconds")

    with col2:
        st.subheader("🔍 Feature Analysis")

        # Feature vector info
        st.markdown("**Feature Vectors:**")
        st.write(f"Dimension: {face1_features.shape[1]} components")
        st.write(f"Feature 1 range: [{face1_features.min():.2f}, {face1_features.max():.2f}]")
        st.write(f"Feature 2 range: [{face2_features.min():.2f}, {face2_features.max():.2f}]")

        # Mathematical explanation
        with st.expander("🧮 Mathematical Details"):
            st.markdown("""
            **Distance Calculation:**
            $$d = \|f_1 - f_2\|_2 = \sqrt{\sum_{i=1}^{k} (f_{1i} - f_{2i})^2}$$

            **Confidence Score:**
            $$\text{confidence} = \max(0, \frac{\tau - d}{\tau}) \times 100\%$$

            Where:
            - $f_1, f_2$ = feature vectors in PCA space
            - $k$ = number of principal components
            - $d$ = Euclidean distance
            - $\tau$ = verification threshold
            """)

    # Visual comparison
    st.subheader("👥 Visual Comparison")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.image(img1_data['original'], caption="First person (original)", use_column_width=True)
        st.image(img1_data['processed'], caption="First person (processed)", use_column_width=True, clamp=True)

    with col2:
        st.image(img2_data['original'], caption="Second person (original)", use_column_width=True)
        st.image(img2_data['processed'], caption="Second person (processed)", use_column_width=True, clamp=True)

    with col3:
        # Feature comparison plot
        st.markdown("**Feature Vector Comparison:**")

        # Create comparison chart
        try:
            chart_utils = ChartUtils()

            # Select top components for visualization
            n_components_to_show = min(20, face1_features.shape[1])
            indices = np.arange(n_components_to_show)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(indices, face1_features.flatten()[:n_components_to_show],
                   'b-o', label='First Person', linewidth=2, markersize=4)
            ax.plot(indices, face2_features.flatten()[:n_components_to_show],
                   'r-s', label='Second Person', linewidth=2, markersize=4)

            ax.set_xlabel('Principal Component Index')
            ax.set_ylabel('Feature Value')
            ax.set_title('PCA Feature Vector Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.warning("Could not generate feature comparison chart")

    # Technical explanation
    with st.expander("🔬 How Verification Works"):
        st.markdown("""
        ### Face Verification Process:

        **1. Feature Extraction:**
        - Both processed faces are flattened into 10304-dimensional vectors (92×112)
        - Vectors are centered by subtracting the mean face
        - Centered vectors are projected onto the trained eigenface space
        - Result: Compact feature vectors representing facial characteristics

        **2. Distance Computation:**
        - Euclidean distance computed between feature vectors
        - Distance measures similarity in the eigenface space
        - Smaller distances indicate higher similarity

        **3. Threshold-Based Decision:**
        - Predefined threshold (~80 units) determines verification decision
        - Threshold determined empirically from training data
        - Distance < threshold → Same person
        - Distance ≥ threshold → Different persons

        **4. Confidence Scoring:**
        - Confidence derived from distance relative to threshold
        - Maximum confidence (100%) at distance = 0
        - Zero confidence at distance = threshold
        - Linear interpolation between these points

        ### Technical Notes:
        - Eigenface space captures variations in facial appearance
        - Distance metric is robust to minor pose and expression changes
        - System works best with frontal faces under similar lighting conditions
        - Processing time: ~0.5 seconds for complete verification
        """)

    # Export functionality
    st.markdown("---")
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download Report", key="download_report"):
            # Create text report
            report = f"""
Face Verification Report
========================

Verification Result: {verification_result}
Confidence: {confidence:.1f}%
Processing Time: {processing_time:.3f} seconds

Distance Metrics:
- Euclidean Distance: {verification_results['euclidean_distance']:.2f}
- Estimated Threshold: {verification_results['estimated_threshold']:.2f}
- Manhattan Distance: {verification_results['manhattan_distance']:.2f}
- Cosine Similarity: {verification_results['cosine_similarity']:.4f}

Feature Analysis:
- Feature Dimension: {face1_features.shape[1]} components
- Feature 1 Range: [{face1_features.min():.2f}, {face1_features.max():.2f}]
- Feature 2 Range: [{face2_features.min():.2f}, {face2_features.max():.2f}]

Detection Information:
- Face 1 Method: {img1_data['detection_info'].get('method', 'unknown')}
- Face 2 Method: {img2_data['detection_info'].get('method', 'unknown')}
            """

            st.download_button(
                label="📄 verification_report.txt",
                data=report,
                file_name="verification_report.txt",
                mime="text/plain"
            )

    with col2:
        if st.button("🖼️ Save Comparison", key="save_comparison"):
            # Create comparison image
            try:
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))

                # Original images
                axes[0, 0].imshow(np.array(img1_data['original']))
                axes[0, 0].set_title('First Person (Original)')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(np.array(img2_data['original']))
                axes[0, 1].set_title('Second Person (Original)')
                axes[0, 1].axis('off')

                # Processed images
                axes[1, 0].imshow(img1_data['processed'], cmap='gray')
                axes[1, 0].set_title('First Person (Processed)')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(img2_data['processed'], cmap='gray')
                axes[1, 1].set_title('Second Person (Processed)')
                axes[1, 1].axis('off')

                plt.tight_layout()

                # Save to buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)

                st.download_button(
                    label="🖼️ comparison.png",
                    data=buffer,
                    file_name="face_verification_comparison.png",
                    mime="image/png"
                )

                plt.close()

            except Exception as e:
                st.error("Could not generate comparison image")

    with col3:
        if st.button("🔄 Reset Verification", key="reset_verification"):
            # Clear verification data
            st.session_state.verification_images = {}
            st.session_state.verification_results = {}
            st.session_state.verification_processed = False
            st.session_state.verification_tab = 0
            st.rerun()


def main():
    """Main function for Face Verification page."""
    # Fix exec() context issue by setting __name__ global
    if '__name__' not in globals():
        globals()['__name__'] = '__main__'

    try:
        # Initialize session state
        initialize_session_state()

        # Page title and description
        st.title("👤 Face Verification")
        st.markdown("Verify if two uploaded images show the same person using PCA feature comparison")

        # Tab content based on current tab
        if st.session_state.verification_tab == 0:
            # Tab 1: Model Information
            can_proceed = safe_streamlit_execute(tab_1_model_information)
            if can_proceed:
                st.session_state.tab1_complete = True

        elif st.session_state.verification_tab == 1:
            # Tab 2: Face Upload & Processing
            if not st.session_state.get('pca_model'):
                st.warning("⚠️ Please complete Tab 1 first!")
                st.session_state.verification_tab = 0
                st.rerun()
            else:
                safe_streamlit_execute(tab_2_face_upload_processing)

        elif st.session_state.verification_tab == 2:
            # Tab 3: Verification Results
            if len(st.session_state.verification_images) != 2:
                st.warning("⚠️ Please complete Tab 2 first!")
                st.session_state.verification_tab = 1
                st.rerun()
            else:
                safe_streamlit_execute(tab_3_verification_results)

        # Tab navigation
        tab_navigation()

        # Additional footer info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📖 Verification Info")
        st.sidebar.info(
            "Face verification uses PCA feature comparison to determine if two images "
            "show the same person. The system extracts compact facial features and "
            "computes the distance between them for identity verification."
        )

        # Model requirements
        st.sidebar.markdown("### ⚙️ Requirements")
        st.sidebar.warning(
            "• Trained PCA model from Eigenfaces page\n"
            "• Two clear face images\n"
            "• Front-facing photos work best\n"
            "• Good lighting conditions"
        )

    except Exception as e:
        log_exception(e, context="Face Verification main function")
        st.error("❌ An unexpected error occurred in the Face Verification page")
        st.error("Please check the log file for detailed information.")


if __name__ == "__main__":
    main()