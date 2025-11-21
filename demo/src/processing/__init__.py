"""
Data Processing and Face Handling Module

This module contains classes for data loading, face detection, image processing,
and data preparation for the PCA face recognition system.

Classes:
- FaceProcessor: Face detection, alignment, and preprocessing
- DatasetLoader: AT&T face dataset loading and management
- ImageUtils: Image processing utilities and helper functions
"""

from .face_processor import FaceProcessor
# from .dataset_loader import DatasetLoader  # Will be implemented next

__all__ = ["FaceProcessor"]