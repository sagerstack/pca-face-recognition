"""
Core PCA and Mathematical Computation Module

This module contains the core PCA implementation and mathematical utilities
for face recognition, implementing algorithms from first principles without
relying on scikit-learn's PCA functionality.

Classes:
- PCA: Principal Component Analysis from first principles
- EigenfacesRecognizer: Face recognition using eigenfaces
- MathematicalUtils: Mathematical helper functions and utilities
"""

from .pca import PCA
from .eigenfaces_recognizer import EigenfacesRecognizer

__all__ = ["PCA", "EigenfacesRecognizer"]