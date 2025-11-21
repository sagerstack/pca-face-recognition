"""
Dataset Loader for AT&T Face Dataset

This module provides comprehensive dataset loading and management capabilities
for the AT&T face dataset with robust error handling and logging.

Author: PCA Face Recognition Team
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
try:
    from ..utils.logger import safe_streamlit_execute, log_exception
except ImportError:
    from utils.logger import safe_streamlit_execute, log_exception


class DatasetLoader:
    """
    AT&T dataset loading and management with comprehensive error handling.

    This class handles loading, validation, and splitting of the AT&T face dataset
    with robust error handling and logging capabilities.
    """

    def __init__(self, data_path: str):
        """
        Initialize the dataset loader.

        Args:
            data_path: Path to the AT&T dataset directory

        Raises:
            ValueError: If data_path is invalid
        """
        try:
            self.data_path = Path(data_path)
            self.logger = logging.getLogger('pca_face_recognition')

            # Validate dataset path
            if not self.data_path.exists():
                raise ValueError(f"Dataset path does not exist: {data_path}")

            if not self.data_path.is_dir():
                raise ValueError(f"Dataset path is not a directory: {data_path}")

            self.logger.info(f"DatasetLoader initialized with path: {data_path}")

        except Exception as e:
            log_exception(e, context="DatasetLoader.__init__")
            raise

    def load_att_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load AT&T dataset with labels and comprehensive error handling.

        Returns:
            Tuple of (images, labels) where:
            - images: numpy array of shape (n_samples, height, width)
            - labels: numpy array of shape (n_samples,)

        Raises:
            ValueError: If dataset structure is invalid
            IOError: If image files cannot be read
        """
        try:
            self.logger.info("Starting AT&T dataset loading...")

            images = []
            labels = []

            # AT&T dataset structure: s1/, s2/, ..., s40/ each with 10 images
            subject_dirs = sorted([d for d in self.data_path.iterdir()
                                 if d.is_dir() and d.name.startswith('s')])

            if len(subject_dirs) != 40:
                self.logger.warning(f"Expected 40 subject directories, found {len(subject_dirs)}")

            for subject_dir in subject_dirs:
                try:
                    # Extract subject number from directory name (s1, s2, etc.)
                    subject_num = int(subject_dir.name[1:])

                    # Get all PGM images in subject directory
                    image_files = sorted([f for f in subject_dir.glob('*.pgm')])

                    if len(image_files) != 10:
                        self.logger.warning(f"Subject {subject_num}: Expected 10 images, found {len(image_files)}")

                    for img_file in image_files:
                        try:
                            # Read image using numpy
                            img_data = self._read_pgm_image(img_file)

                            # Validate image shape (should be 112x92 for AT&T dataset)
                            if img_data.shape != (112, 92):
                                self.logger.warning(f"Unexpected image shape {img_data.shape} in {img_file}")
                                # Resize if needed
                                img_data = self._standardize_image_size(img_data)

                            images.append(img_data)
                            labels.append(subject_num)

                        except Exception as e:
                            self.logger.error(f"Failed to read image {img_file}: {str(e)}")
                            continue

                except Exception as e:
                    self.logger.error(f"Failed to process subject directory {subject_dir}: {str(e)}")
                    continue

            if not images:
                raise ValueError("No valid images found in dataset")

            # Convert to numpy arrays
            X = np.array(images, dtype=np.float32)
            y = np.array(labels, dtype=np.int32)

            self.logger.info(f"Successfully loaded {len(images)} images from {len(subject_dirs)} subjects")
            self.logger.info(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

            return X, y

        except Exception as e:
            log_exception(e, context="DatasetLoader.load_att_dataset")
            raise

    def train_test_split(self, X: np.ndarray, y: np.ndarray,
                        train_size: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset by subject with configurable training size.

        Args:
            X: Image data array of shape (n_samples, height, width)
            y: Label array of shape (n_samples,)
            train_size: Number of training images per subject (1-9)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)

        Raises:
            ValueError: If parameters are invalid
        """
        try:
            self.logger.info(f"Splitting dataset with train_size={train_size} per subject")

            if not 1 <= train_size <= 9:
                raise ValueError("train_size must be between 1 and 9")

            if len(X) != len(y):
                raise ValueError("X and y must have the same length")

            # Get unique subjects
            unique_subjects = np.unique(y)
            X_train_list = []
            X_test_list = []
            y_train_list = []
            y_test_list = []

            for subject in unique_subjects:
                try:
                    # Get indices for this subject
                    subject_indices = np.where(y == subject)[0]

                    if len(subject_indices) < train_size:
                        self.logger.warning(f"Subject {subject} has only {len(subject_indices)} images, less than train_size={train_size}")
                        continue

                    # Shuffle indices for random selection
                    np.random.shuffle(subject_indices)

                    # Split into train and test
                    train_indices = subject_indices[:train_size]
                    test_indices = subject_indices[train_size:]

                    # Add to respective lists
                    X_train_list.append(X[train_indices])
                    X_test_list.append(X[test_indices])
                    y_train_list.append(y[train_indices])
                    y_test_list.append(y[test_indices])

                except Exception as e:
                    self.logger.error(f"Failed to split subject {subject}: {str(e)}")
                    continue

            if not X_train_list:
                raise ValueError("No valid training data created")

            # Concatenate all subjects
            X_train = np.concatenate(X_train_list, axis=0)
            X_test = np.concatenate(X_test_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)

            # Shuffle the final datasets
            train_shuffle = np.random.permutation(len(X_train))
            test_shuffle = np.random.permutation(len(X_test))

            X_train = X_train[train_shuffle]
            y_train = y_train[train_shuffle]
            X_test = X_test[test_shuffle]
            y_test = y_test[test_shuffle]

            self.logger.info(f"Dataset split complete:")
            self.logger.info(f"  Training: {X_train.shape[0]} samples from {len(np.unique(y_train))} subjects")
            self.logger.info(f"  Testing: {X_test.shape[0]} samples from {len(np.unique(y_test))} subjects")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            log_exception(e, context="DatasetLoader.train_test_split")
            raise

    def validate_dataset_integrity(self) -> Dict[str, Any]:
        """
        Validate dataset structure and provide integrity report.

        Returns:
            Dictionary containing validation results
        """
        try:
            self.logger.info("Starting dataset integrity validation...")

            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }

            # Check if dataset path exists
            if not self.data_path.exists():
                validation_results['valid'] = False
                validation_results['errors'].append(f"Dataset path does not exist: {self.data_path}")
                return validation_results

            # Check subject directories
            subject_dirs = [d for d in self.data_path.iterdir()
                          if d.is_dir() and d.name.startswith('s')]

            if len(subject_dirs) == 0:
                validation_results['valid'] = False
                validation_results['errors'].append("No subject directories found")
                return validation_results

            validation_results['stats']['total_subjects'] = len(subject_dirs)

            total_images = 0
            expected_subjects = set(range(1, 41))
            found_subjects = set()

            for subject_dir in subject_dirs:
                try:
                    subject_num = int(subject_dir.name[1:])
                    found_subjects.add(subject_num)

                    image_files = [f for f in subject_dir.glob('*.pgm')]
                    total_images += len(image_files)

                    if len(image_files) != 10:
                        validation_results['warnings'].append(
                            f"Subject {subject_num}: Expected 10 images, found {len(image_files)}"
                        )

                except ValueError:
                    validation_results['warnings'].append(f"Invalid subject directory name: {subject_dir.name}")
                except Exception as e:
                    validation_results['errors'].append(f"Error processing {subject_dir}: {str(e)}")

            validation_results['stats']['total_images'] = total_images
            validation_results['stats']['expected_subjects'] = 40
            validation_results['stats']['found_subjects'] = len(found_subjects)
            validation_results['stats']['missing_subjects'] = list(expected_subjects - found_subjects)

            # Check if any critical errors occurred
            if validation_results['errors']:
                validation_results['valid'] = False

            # Log validation results
            if validation_results['valid']:
                self.logger.info("Dataset validation passed")
                self.logger.info(f"Found {total_images} images across {len(found_subjects)} subjects")
            else:
                self.logger.error("Dataset validation failed")
                for error in validation_results['errors']:
                    self.logger.error(f"  {error}")

            return validation_results

        except Exception as e:
            log_exception(e, context="DatasetLoader.validate_dataset_integrity")
            return {
                'valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'stats': {}
            }

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.

        Returns:
            Dictionary containing dataset statistics
        """
        try:
            validation = self.validate_dataset_integrity()

            info = {
                'dataset_path': str(self.data_path),
                'dataset_name': 'AT&T Face Dataset',
                'expected_structure': {
                    'subjects': 40,
                    'images_per_subject': 10,
                    'total_images': 400,
                    'image_size': '92x112 pixels',
                    'format': 'PGM grayscale'
                },
                'validation': validation
            }

            if validation['valid']:
                # Try to get actual dataset info by loading a small sample
                try:
                    X_sample, y_sample = self.load_att_dataset()
                    info['actual_structure'] = {
                        'subjects': len(np.unique(y_sample)),
                        'total_images': len(X_sample),
                        'image_shape': X_sample[0].shape if len(X_sample) > 0 else None,
                        'data_type': str(X_sample.dtype)
                    }
                except Exception as e:
                    info['load_error'] = str(e)

            return info

        except Exception as e:
            log_exception(e, context="DatasetLoader.get_dataset_info")
            return {
                'dataset_path': str(self.data_path),
                'error': f"Failed to get dataset info: {str(e)}"
            }

    def _read_pgm_image(self, filepath: Path) -> np.ndarray:
        """
        Read PGM image file with error handling.

        Args:
            filepath: Path to PGM file

        Returns:
            Image as numpy array

        Raises:
            IOError: If file cannot be read
        """
        try:
            with open(filepath, 'rb') as f:
                # Read PGM header
                header = f.readline().decode('ascii').strip()

                if header != 'P2' and header != 'P5':
                    raise ValueError(f"Unsupported PGM format: {header}")

                # Skip comments
                while True:
                    line = f.readline().decode('ascii').strip()
                    if not line.startswith('#'):
                        break

                # Read dimensions
                width, height = map(int, line.split())

                # Read max value
                max_val = int(f.readline().decode('ascii').strip())

                if header == 'P2':  # ASCII format
                    # Read pixel values
                    pixels = []
                    while len(pixels) < width * height:
                        line = f.readline().decode('ascii').strip()
                        pixels.extend(map(int, line.split()))

                    image = np.array(pixels[:width * height], dtype=np.uint8)
                    image = image.reshape((height, width))

                else:  # P5 binary format
                    # Read binary pixel data
                    if max_val <= 255:
                        image = np.frombuffer(f.read(width * height), dtype=np.uint8)
                    else:
                        image = np.frombuffer(f.read(width * height * 2), dtype=np.uint16)

                    image = image.reshape((height, width))

                return image.astype(np.float32)

        except Exception as e:
            raise IOError(f"Failed to read PGM file {filepath}: {str(e)}")

    def _standardize_image_size(self, image: np.ndarray, target_size: Tuple[int, int] = (112, 92)) -> np.ndarray:
        """
        Standardize image size with basic resizing.

        Args:
            image: Input image array
            target_size: Target size (height, width)

        Returns:
            Resized image array
        """
        try:
            from scipy.ndimage import zoom

            # Calculate zoom factors
            zoom_factors = (target_size[0] / image.shape[0], target_size[1] / image.shape[1])

            # Resize image
            resized_image = zoom(image, zoom_factors, order=1)

            return resized_image.astype(np.float32)

        except ImportError:
            # If scipy is not available, use simple cropping/padding
            self.logger.warning("scipy.ndimage not available, using basic resizing")
            return self._basic_resize(image, target_size)

    def _basic_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Basic image resizing using cropping and padding.

        Args:
            image: Input image array
            target_size: Target size (height, width)

        Returns:
            Resized image array
        """
        height, width = image.shape
        target_height, target_width = target_size

        # Create target image
        resized = np.zeros(target_size, dtype=image.dtype)

        # Calculate cropping/padding dimensions
        h_start = max(0, (height - target_height) // 2)
        h_end = min(height, h_start + target_height)
        w_start = max(0, (width - target_width) // 2)
        w_end = min(width, w_start + target_width)

        # Calculate target dimensions
        t_h_start = max(0, (target_height - (h_end - h_start)) // 2)
        t_h_end = min(target_height, t_h_start + (h_end - h_start))
        t_w_start = max(0, (target_width - (w_end - w_start)) // 2)
        t_w_end = min(target_width, t_w_start + (w_end - w_start))

        # Copy cropped/padded region
        resized[t_h_start:t_h_end, t_w_start:t_w_end] = image[h_start:h_end, w_start:w_end]

        return resized


def main():
    """Test the DatasetLoader class."""
    import sys

    # Create a mock dataset path for testing
    test_path = "path/to/att/dataset"

    try:
        loader = DatasetLoader(test_path)
        info = loader.get_dataset_info()
        print("Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test validation
        validation = loader.validate_dataset_integrity()
        print(f"\nValidation Results:")
        print(f"  Valid: {validation['valid']}")
        if validation['errors']:
            print(f"  Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")
        if validation['stats']:
            print(f"  Stats: {validation['stats']}")

    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()