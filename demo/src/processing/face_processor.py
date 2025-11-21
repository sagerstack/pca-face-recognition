"""
Face Image Processor for PCA Face Recognition

This module handles face detection, preprocessing, and image normalization
for the PCA face recognition system. It uses OpenCV for face detection and
provides various image preprocessing utilities to ensure consistent face
image quality before PCA processing.

Key Features:
- Face detection using Haar cascades
- Image preprocessing and normalization
- Face alignment and cropping
- Histogram equalization and contrast enhancement
- Resizing and standardization for PCA input

Author: PCA Face Recognition Team
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Union
import os
import sys

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.mathematical_utils import NormalizationUtils


class FaceProcessor:
    """
    Face image processing and preprocessing utilities.

    This class provides comprehensive face detection and preprocessing
    capabilities for preparing face images for PCA-based face recognition.
    It handles image loading, face detection, cropping, normalization,
    and standardization.
    """

    def __init__(self,
                 target_size: Tuple[int, int] = (92, 112),
                 detect_faces: bool = True,
                 apply_histogram_equalization: bool = True):
        """
        Initialize the Face Processor.

        Args:
            target_size: Target size for face images (width, height)
            detect_faces: Whether to automatically detect faces in images
            apply_histogram_equalization: Whether to apply histogram equalization
        """
        self.target_size = target_size
        self.detect_faces = detect_faces
        self.apply_histogram_equalization = apply_histogram_equalization

        # Load face detection cascade
        self.face_cascade = None
        if detect_faces:
            self._load_face_cascade()

    def _load_face_cascade(self) -> None:
        """
        Load OpenCV's Haar cascade for face detection.
        """
        try:
            # Try to load the Haar cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                print("Warning: Could not load face cascade classifier")
                self.detect_faces = False
        except Exception as e:
            print(f"Warning: Error loading face cascade: {e}")
            self.detect_faces = False

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file path.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array

        Raises:
            FileNotFoundError: If image file does not exist
            ValueError: If image cannot be loaded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image in grayscale (for face processing)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return image

    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save an image to file.

        Args:
            image: Image to save
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

    def detect_face(self, image: np.ndarray,
                   scale_factor: float = 1.1,
                   min_neighbors: int = 5,
                   min_size: Tuple[int, int] = (30, 30)) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in an image.

        Args:
            image: Input image (grayscale)
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size

        Returns:
            Tuple of (x, y, width, height) for the detected face, or None if no face detected
        """
        if not self.detect_faces or self.face_cascade is None:
            return None

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        # Return the largest face (by area)
        if len(faces) > 0:
            # Find the face with the largest area
            areas = [w * h for (x, y, w, h) in faces]
            largest_face_idx = np.argmax(areas)
            return tuple(faces[largest_face_idx])

        return None

    def crop_face(self, image: np.ndarray,
                  face_rect: Optional[Tuple[int, int, int, int]] = None,
                  padding_ratio: float = 0.2) -> np.ndarray:
        """
        Crop face from image.

        Args:
            image: Input image
            face_rect: Face rectangle (x, y, width, height), None if face detection should be used
            padding_ratio: Ratio of padding around the face

        Returns:
            Cropped face image
        """
        if face_rect is None:
            # Try to detect face automatically
            face_rect = self.detect_face(image)

        if face_rect is None:
            # No face detected, use entire image
            return image

        x, y, w, h = face_rect

        # Add padding around the face
        padding_x = int(w * padding_ratio)
        padding_y = int(h * padding_ratio)

        # Calculate cropped region with padding
        crop_x = max(0, x - padding_x)
        crop_y = max(0, y - padding_y)
        crop_w = min(image.shape[1] - crop_x, w + 2 * padding_x)
        crop_h = min(image.shape[0] - crop_y, h + 2 * padding_y)

        # Crop the face
        face_crop = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        return face_crop

    def resize_image(self, image: np.ndarray,
                    target_size: Optional[Tuple[int, int]] = None,
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image
            target_size: Target size (width, height), None to use default
            interpolation: OpenCV interpolation method

        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size

        return cv2.resize(image, target_size, interpolation=interpolation)

    def equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast.

        Args:
            image: Input image

        Returns:
            Histogram-equalized image
        """
        return cv2.equalizeHist(image)

    def normalize_image(self, image: np.ndarray,
                       method: str = 'min_max',
                       target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Normalize image pixel values.

        Args:
            image: Input image
            method: Normalization method ('min_max', 'z_score', 'l2')
            target_range: Target range for min_max normalization

        Returns:
            Normalized image
        """
        image_float = image.astype(np.float64)

        if method == 'min_max':
            # Convert to 1D array for min_max normalization
            image_flat = image_float.flatten()
            normalized_flat = NormalizationUtils.min_max_normalize(
                image_flat, feature_range=target_range
            )
            return normalized_flat.reshape(image_float.shape).astype(np.float32)

        elif method == 'z_score':
            # Convert to 1D array for z-score normalization
            image_flat = image_float.flatten()
            normalized_flat = NormalizationUtils.z_score_normalize(image_flat)
            return normalized_flat.reshape(image_float.shape).astype(np.float32)

        elif method == 'l2':
            return NormalizationUtils.l2_normalize(image_float).astype(np.float32)

        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def apply_gaussian_blur(self, image: np.ndarray,
                           kernel_size: Tuple[int, int] = (3, 3),
                           sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise.

        Args:
            image: Input image
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian kernel

        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def apply_sobel_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel edge detection filter.

        Args:
            image: Input image

        Returns:
            Edge-enhanced image
        """
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize to 0-255 range
        sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return sobel_normalized.astype(np.uint8)

    def process_face_image(self,
                          image: Union[str, np.ndarray],
                          detect_face: Optional[bool] = None,
                          normalize: bool = True,
                          normalize_method: str = 'min_max') -> np.ndarray:
        """
        Complete face image processing pipeline.

        Args:
            image: Input image (file path or numpy array)
            detect_face: Whether to detect and crop face, None to use default
            normalize: Whether to normalize the image
            normalize_method: Normalization method

        Returns:
            Processed face image
        """
        # Load image if path provided
        if isinstance(image, str):
            processed_image = self.load_image(image)
        else:
            processed_image = image.copy()

        # Face detection and cropping
        if detect_face is None:
            detect_face = self.detect_faces

        if detect_face:
            processed_image = self.crop_face(processed_image)

        # Apply histogram equalization
        if self.apply_histogram_equalization:
            processed_image = self.equalize_histogram(processed_image)

        # Resize to target size
        processed_image = self.resize_image(processed_image)

        # Normalize if requested
        if normalize:
            processed_image = self.normalize_image(processed_image, method=normalize_method)

        return processed_image

    def flatten_image(self, image: np.ndarray) -> np.ndarray:
        """
        Flatten image to 1D vector for PCA processing.

        Args:
            image: Input image

        Returns:
            Flattened image vector
        """
        return image.flatten()

    def process_for_pca(self,
                       image: Union[str, np.ndarray],
                       flatten: bool = True,
                       detect_face: Optional[bool] = None) -> np.ndarray:
        """
        Process image specifically for PCA face recognition.

        Args:
            image: Input image (file path or numpy array)
            flatten: Whether to flatten the result to 1D vector
            detect_face: Whether to detect face, None to use default

        Returns:
            Processed image ready for PCA
        """
        processed = self.process_face_image(image, detect_face=detect_face)

        if flatten:
            processed = self.flatten_image(processed)

        return processed

    def batch_process_images(self,
                            image_paths: List[str],
                            show_progress: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Process multiple images in batch.

        Args:
            image_paths: List of image file paths
            show_progress: Whether to show progress

        Returns:
            Tuple of (processed_images, failed_paths)
        """
        processed_images = []
        failed_paths = []

        for i, image_path in enumerate(image_paths):
            try:
                if show_progress and (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")

                processed = self.process_for_pca(image_path)
                processed_images.append(processed)

            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                failed_paths.append(image_path)

        return np.array(processed_images), failed_paths

    def get_processing_info(self) -> dict:
        """
        Get information about current processing settings.

        Returns:
            Dictionary with processing configuration
        """
        return {
            'target_size': self.target_size,
            'detect_faces': self.detect_faces,
            'equalize_histogram': self.apply_histogram_equalization,
            'face_cascade_loaded': self.face_cascade is not None if self.detect_faces else False
        }


def main():
    """
    Demonstrate the Face Processor functionality.
    """
    print("FACE PROCESSOR DEMONSTRATION")
    print("=" * 50)

    # Create face processor
    processor = FaceProcessor(
        target_size=(92, 112),
        detect_faces=True,
        apply_histogram_equalization=True
    )

    print(f"\n1. Face Processor Configuration:")
    info = processor.get_processing_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Create a synthetic face image for demonstration
    print(f"\n2. Creating synthetic face image...")
    np.random.seed(42)

    # Create a simple face-like pattern
    face_size = (200, 200)
    synthetic_face = np.random.randint(50, 200, face_size, dtype=np.uint8)

    # Add some face-like features
    # Eyes
    synthetic_face[60:80, 60:90] = 255  # Left eye
    synthetic_face[60:80, 110:140] = 255  # Right eye
    # Nose
    synthetic_face[80:120, 90:110] = 180
    # Mouth
    synthetic_face[130:150, 70:130] = 200

    print(f"   Original image shape: {synthetic_face.shape}")
    print(f"   Original image dtype: {synthetic_face.dtype}")
    print(f"   Pixel value range: {synthetic_face.min()} - {synthetic_face.max()}")

    # Test face processing pipeline
    print(f"\n3. Testing face processing pipeline...")

    # Process without face detection (since it's synthetic)
    processed_face = processor.process_face_image(
        synthetic_face,
        detect_face=False,
        normalize=True,
        normalize_method='min_max'
    )

    print(f"   Processed image shape: {processed_face.shape}")
    print(f"   Processed image dtype: {processed_face.dtype}")
    print(f"   Normalized pixel range: {processed_face.min():.4f} - {processed_face.max():.4f}")

    # Test PCA preprocessing
    print(f"\n4. Testing PCA preprocessing...")
    pca_ready = processor.process_for_pca(synthetic_face, flatten=True)

    print(f"   PCA-ready vector shape: {pca_ready.shape}")
    print(f"   PCA-ready vector length: {len(pca_ready)}")

    # Test different normalization methods
    print(f"\n5. Testing different normalization methods...")

    methods = ['min_max', 'z_score', 'l2']
    for method in methods:
        normalized = processor.normalize_image(synthetic_face, method=method)
        print(f"   {method:10s}: min={normalized.min():8.4f}, max={normalized.max():8.4f}, "
              f"mean={normalized.mean():8.4f}, std={normalized.std():8.4f}")

    # Test image augmentation techniques
    print(f"\n6. Testing image processing techniques...")

    # Gaussian blur
    blurred = processor.apply_gaussian_blur(synthetic_face)
    print(f"   Gaussian blur applied: {blurred.shape}")

    # Sobel edge detection
    edges = processor.apply_sobel_filter(synthetic_face)
    print(f"   Sobel edges detected: {edges.shape}")

    # Resize test
    resized = processor.resize_image(synthetic_face, target_size=(64, 64))
    print(f"   Resized to 64x64: {resized.shape}")

    # Test histogram equalization
    equalized = processor.equalize_histogram(synthetic_face)
    print(f"   Histogram equalized: {equalized.shape}")
    print(f"   Equalized pixel range: {equalized.min()} - {equalized.max()}")

    # Memory usage estimate for different batch sizes
    print(f"\n7. Memory usage estimates:")
    target_width, target_height = processor.target_size
    pixels_per_face = target_width * target_height
    bytes_per_float32 = 4

    batch_sizes = [100, 500, 1000, 5000]
    for batch_size in batch_sizes:
        memory_mb = (batch_size * pixels_per_face * bytes_per_float32) / (1024 * 1024)
        print(f"   {batch_size:4d} faces: ~{memory_mb:6.2f} MB")

    print(f"\n✅ Face processor demonstration complete!")
    print(f"\n   Ready to process face images for PCA recognition!")
    print(f"   Target size: {processor.target_size} (width x height)")
    print(f"   Face detection: {'Enabled' if processor.detect_faces else 'Disabled'}")
    print(f"   Histogram equalization: {'Enabled' if processor.equalize_histogram else 'Disabled'}")


if __name__ == "__main__":
    main()