"""
Eigenfaces Face Recognition System

This module implements the Eigenfaces face recognition algorithm using PCA for
dimensionality reduction and various distance metrics for face matching.
The system works with face images that have been preprocessed and flattened
into feature vectors.

Mathematical Foundation:
1. Training: Compute PCA on training face images to extract eigenfaces
2. Projection: Project all faces onto the eigenface subspace
3. Recognition: Compare projected faces using distance metrics
4. Classification: Use nearest neighbor or threshold-based classification

Eigenfaces Recognition Process:
- Compute PCA on training faces → eigenfaces (principal components)
- Project training faces onto eigenface subspace → weight vectors
- Project query face onto same subspace → query weight vector
- Find closest training face using distance metric → recognition result

Author: PCA Face Recognition Team
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Union
import pickle
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from pca import PCA
from mathematical_utils import DistanceMetrics, FaceRecognitionUtils


class EigenfacesRecognizer:
    """
    Eigenfaces-based face recognition system using PCA.

    This class implements the classical eigenfaces algorithm for face recognition,
    which uses Principal Component Analysis to extract the most significant
    facial features (eigenfaces) and then performs recognition by comparing
    face projections in the reduced-dimensional space.

    The system follows these steps:
    1. Train: Learn eigenfaces from training face images using PCA
    2. Project: Transform all faces into the eigenface subspace
    3. Recognize: Match query faces against known faces using distance metrics
    """

    def __init__(self,
                 n_components: Optional[int] = None,
                 distance_metric: str = 'euclidean',
                 recognition_threshold: Optional[float] = None):
        """
        Initialize the Eigenfaces Recognizer.

        Args:
            n_components: Number of eigenfaces to keep (PCA components)
            distance_metric: Distance metric for face matching ('euclidean', 'cosine', 'manhattan')
            recognition_threshold: Threshold for face recognition (None for auto)
        """
        self.n_components = n_components
        self.distance_metric = distance_metric
        self.recognition_threshold = recognition_threshold

        # Core components
        self.pca = PCA(n_components=n_components)

        # Training data
        self.training_faces = None          # Original training faces
        self.training_labels = None         # Subject IDs for training faces
        self.projected_training_faces = None  # Faces projected to eigenface space
        self.subject_face_map = {}          # Maps subject ID to list of face indices
        self.unique_subjects = []           # List of unique subject IDs

        # Recognition statistics
        self.mean_training_distance = None
        self.std_training_distance = None

        # Model state
        self.is_trained = False

    def _validate_training_data(self, faces: np.ndarray, labels: np.ndarray) -> None:
        """
        Validate training data input.

        Args:
            faces: Training face images
            labels: Subject ID labels

        Raises:
            ValueError: If input data is invalid
        """
        if faces.shape[0] == 0:
            raise ValueError("No training faces provided")

        if labels.shape[0] != faces.shape[0]:
            raise ValueError(f"Number of faces ({faces.shape[0]}) and labels ({labels.shape[0]}) must match")

        if faces.ndim != 2:
            raise ValueError(f"Faces must be 2D array (n_samples, n_features), got {faces.ndim}D")

        if len(np.unique(labels)) < 2:
            raise ValueError("Training data must contain at least 2 different subjects")

    def _build_subject_face_map(self, labels: np.ndarray) -> None:
        """
        Build mapping from subject IDs to face indices.

        Args:
            labels: Subject ID labels for training faces
        """
        self.subject_face_map = {}
        self.unique_subjects = np.unique(labels)

        for subject_id in self.unique_subjects:
            face_indices = np.where(labels == subject_id)[0]
            self.subject_face_map[subject_id] = face_indices.tolist()

    def _compute_training_statistics(self) -> None:
        """
        Compute statistics for automatic threshold determination.

        Calculates mean and standard deviation of distances between faces
        of the same subject to set appropriate recognition thresholds.
        """
        if self.projected_training_faces is None:
            return

        distances = []

        # Compute distances between faces of the same subject
        for subject_id, face_indices in self.subject_face_map.items():
            if len(face_indices) < 2:
                continue  # Need at least 2 faces per subject

            subject_faces = self.projected_training_faces[face_indices]

            # Compute all pairwise distances within this subject
            for i in range(len(subject_faces)):
                for j in range(i + 1, len(subject_faces)):
                    if self.distance_metric == 'euclidean':
                        dist = DistanceMetrics.euclidean_distance(
                            subject_faces[i], subject_faces[j]
                        )
                    elif self.distance_metric == 'cosine':
                        dist = DistanceMetrics.cosine_distance(
                            subject_faces[i], subject_faces[j]
                        )
                    elif self.distance_metric == 'manhattan':
                        dist = DistanceMetrics.manhattan_distance(
                            subject_faces[i], subject_faces[j]
                        )

                    distances.append(dist)

        if distances:
            self.mean_training_distance = np.mean(distances)
            self.std_training_distance = np.std(distances)
        else:
            self.mean_training_distance = 0.0
            self.std_training_distance = 1.0

        # Auto-set threshold if not provided
        if self.recognition_threshold is None:
            # Use mean + 2 standard deviations as default threshold
            self.recognition_threshold = self.mean_training_distance + 2 * self.std_training_distance

    def fit(self, faces: np.ndarray, labels: np.ndarray) -> 'EigenfacesRecognizer':
        """
        Train the eigenfaces recognizer.

        This method trains the PCA model on the provided face images and
        projects them into the eigenface subspace for recognition.

        Args:
            faces: Training face images of shape (n_faces, n_features)
            labels: Subject ID labels for each face

        Returns:
            Self (the trained recognizer)

        Raises:
            ValueError: If input data is invalid

        Example:
            >>> recognizer = EigenfacesRecognizer(n_components=50)
            >>> recognizer.fit(training_faces, training_labels)
            >>> print(f"Trained on {len(recognizer.unique_subjects)} subjects")
        """
        # Validate input
        self._validate_training_data(faces, labels)

        # Store original training data
        self.training_faces = faces.copy()
        self.training_labels = labels.copy()

        # Build subject face mapping
        self._build_subject_face_map(labels)

        # Train PCA on training faces
        print(f"Training PCA on {faces.shape[0]} faces with {faces.shape[1]} features...")
        self.projected_training_faces = self.pca.fit_transform(faces)

        # Compute training statistics for threshold setting
        self._compute_training_statistics()

        # Mark as trained
        self.is_trained = True

        print(f"✅ Training complete!")
        print(f"   - Subjects: {len(self.unique_subjects)}")
        print(f"   - Components: {self.pca.n_components}")
        print(f"   - Variance explained: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        print(f"   - Recognition threshold: {self.recognition_threshold:.4f}")

        return self

    def predict(self, query_faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recognize faces in query images.

        Args:
            query_faces: Query face images of shape (n_queries, n_features)

        Returns:
            Tuple of (predicted_labels, distances)

        Raises:
            ValueError: If model is not trained or input is invalid
        """
        if not self.is_trained:
            raise ValueError("Recognizer must be trained before prediction")

        if query_faces.ndim != 2:
            raise ValueError(f"Query faces must be 2D array, got {query_faces.ndim}D")

        if query_faces.shape[1] != self.training_faces.shape[1]:
            raise ValueError(
                f"Query face features ({query_faces.shape[1]}) must match "
                f"training features ({self.training_faces.shape[1]})"
            )

        # Project query faces into eigenface subspace
        projected_query_faces = self.pca.transform(query_faces)

        # Find best matches for each query face
        predicted_labels = np.zeros(query_faces.shape[0], dtype=self.training_labels.dtype)
        distances = np.zeros(query_faces.shape[0])

        for i, query_face in enumerate(projected_query_faces):
            # Find best match in training data
            best_match_idx, distance = FaceRecognitionUtils.find_best_match(
                query_face, self.projected_training_faces, self.distance_metric
            )

            # Check if distance is below threshold
            if distance <= self.recognition_threshold:
                predicted_labels[i] = self.training_labels[best_match_idx]
            else:
                predicted_labels[i] = -1  # Unknown face

            distances[i] = distance

        return predicted_labels, distances

    def predict_single(self, query_face: np.ndarray) -> Tuple[int, float, float]:
        """
        Recognize a single face image.

        Args:
            query_face: Single query face image of shape (n_features,)

        Returns:
            Tuple of (predicted_label, distance, confidence)

        Raises:
            ValueError: If model is not trained or input is invalid
        """
        if not self.is_trained:
            raise ValueError("Recognizer must be trained before prediction")

        query_face = np.asarray(query_face)

        if query_face.ndim != 1:
            raise ValueError(f"Query face must be 1D array, got {query_face.ndim}D")

        # Ensure 2D shape for PCA
        query_face_2d = query_face.reshape(1, -1)

        # Predict
        predicted_labels, distances = self.predict(query_face_2d)

        predicted_label = predicted_labels[0]
        distance = distances[0]

        # Calculate confidence
        confidence = FaceRecognitionUtils.calculate_confidence_score(
            distance, self.recognition_threshold, self.distance_metric
        )

        return predicted_label, distance, confidence

    def evaluate(self, test_faces: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate recognizer performance on test data.

        Args:
            test_faces: Test face images
            test_labels: Ground truth labels for test faces

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Recognizer must be trained before evaluation")

        # Predict on test data
        predicted_labels, distances = self.predict(test_faces)

        # Calculate metrics
        # Note: Label -1 represents "unknown" faces
        known_mask = (test_labels != -1) & (predicted_labels != -1)

        if np.sum(known_mask) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'total_samples': len(test_labels),
                'known_samples': 0
            }

        y_true_known = test_labels[known_mask]
        y_pred_known = predicted_labels[known_mask]

        # Accuracy
        accuracy = np.mean(y_true_known == y_pred_known)

        # Precision, Recall, F1 (macro-averaged)
        unique_labels = np.unique(np.concatenate([y_true_known, y_pred_known]))

        precisions = []
        recalls = []
        f1_scores = []

        for label in unique_labels:
            # True positives
            tp = np.sum((y_true_known == label) & (y_pred_known == label))
            # False positives
            fp = np.sum((y_true_known != label) & (y_pred_known == label))
            # False negatives
            fn = np.sum((y_true_known == label) & (y_pred_known != label))

            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Macro-averaged metrics
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)

        return {
            'accuracy': float(accuracy),
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1_score': float(macro_f1),
            'total_samples': int(len(test_labels)),
            'known_samples': int(np.sum(known_mask)),
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances))
        }

    def get_eigenfaces(self, n_eigenfaces: Optional[int] = None, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get eigenfaces (principal components) as images.

        Args:
            n_eigenfaces: Number of eigenfaces to return (None for all)
            image_shape: Shape to reshape eigenfaces to (None for flat)

        Returns:
            Eigenfaces array
        """
        if not self.is_trained:
            raise ValueError("Recognizer must be trained first")

        eigenfaces = self.pca.components_

        if n_eigenfaces is not None:
            eigenfaces = eigenfaces[:n_eigenfaces]

        if image_shape is not None:
            reshaped_eigenfaces = []
            for eigenface in eigenfaces:
                reshaped_eigenfaces.append(eigenface.reshape(image_shape))
            eigenfaces = np.array(reshaped_eigenfaces)

        return eigenfaces

    def reconstruct_face(self, face: np.ndarray) -> np.ndarray:
        """
        Reconstruct a face using eigenfaces.

        Args:
            face: Face to reconstruct

        Returns:
            Reconstructed face
        """
        if not self.is_trained:
            raise ValueError("Recognizer must be trained first")

        if face.ndim == 1:
            face = face.reshape(1, -1)

        projected = self.pca.transform(face)
        reconstructed = self.pca.inverse_transform(projected)

        # Return flattened if input was flattened
        return reconstructed[0] if len(reconstructed) == 1 else reconstructed

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'pca': self.pca,
            'training_faces': self.training_faces,
            'training_labels': self.training_labels,
            'projected_training_faces': self.projected_training_faces,
            'subject_face_map': self.subject_face_map,
            'unique_subjects': self.unique_subjects,
            'n_components': self.n_components,
            'distance_metric': self.distance_metric,
            'recognition_threshold': self.recognition_threshold,
            'mean_training_distance': self.mean_training_distance,
            'std_training_distance': self.std_training_distance,
            'is_trained': True
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> 'EigenfacesRecognizer':
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Self (the loaded recognizer)
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Restore model state
        self.pca = model_data['pca']
        self.training_faces = model_data['training_faces']
        self.training_labels = model_data['training_labels']
        self.projected_training_faces = model_data['projected_training_faces']
        self.subject_face_map = model_data['subject_face_map']
        self.unique_subjects = model_data['unique_subjects']
        self.n_components = model_data['n_components']
        self.distance_metric = model_data['distance_metric']
        self.recognition_threshold = model_data['recognition_threshold']
        self.mean_training_distance = model_data['mean_training_distance']
        self.std_training_distance = model_data['std_training_distance']
        self.is_trained = model_data['is_trained']

        print(f"Model loaded from: {filepath}")
        return self

    def get_model_summary(self) -> Dict[str, Union[str, int, float]]:
        """
        Get a summary of the trained model.

        Returns:
            Dictionary containing model information
        """
        if not self.is_trained:
            return {"status": "Not trained"}

        return {
            "status": "Trained",
            "n_subjects": len(self.unique_subjects),
            "n_training_faces": len(self.training_faces),
            "n_components": self.pca.n_components,
            "n_features": self.pca.n_features_,
            "total_variance_explained": float(np.sum(self.pca.explained_variance_ratio_)),
            "distance_metric": self.distance_metric,
            "recognition_threshold": float(self.recognition_threshold),
            "mean_training_distance": float(self.mean_training_distance) if self.mean_training_distance else 0.0,
            "std_training_distance": float(self.std_training_distance) if self.std_training_distance else 0.0
        }


def main():
    """
    Demonstrate the Eigenfaces Recognizer functionality.
    """
    print("EIGENFACES RECOGNIZER DEMONSTRATION")
    print("=" * 50)

    # Generate synthetic face data for demonstration
    print("\n1. Creating synthetic face data...")
    np.random.seed(42)

    n_subjects = 10
    n_faces_per_subject = 5
    image_size = 50  # 50x50 = 2500 features

    # Generate synthetic face data with some structure
    training_faces = []
    training_labels = []

    for subject_id in range(n_subjects):
        # Base pattern for each subject
        base_pattern = np.random.randn(image_size, image_size)

        for _ in range(n_faces_per_subject):
            # Add noise to create different faces of same subject
            face = base_pattern + np.random.randn(image_size, image_size) * 0.1
            face_flat = face.flatten()

            training_faces.append(face_flat)
            training_labels.append(subject_id)

    training_faces = np.array(training_faces)
    training_labels = np.array(training_labels)

    print(f"Generated {training_faces.shape[0]} training faces")
    print(f"Number of subjects: {n_subjects}")
    print(f"Face image size: {image_size}x{image_size}")

    # Create and train recognizer
    print("\n2. Training Eigenfaces Recognizer...")
    recognizer = EigenfacesRecognizer(
        n_components=25,
        distance_metric='euclidean'
    )

    recognizer.fit(training_faces, training_labels)

    # Test on some training faces (as a simple test)
    print("\n3. Testing recognizer...")
    test_faces = training_faces[:5]  # Use first 5 faces as test
    test_labels = training_labels[:5]

    predicted_labels, distances = recognizer.predict(test_faces)

    print("\nTest Results:")
    print("True Label | Predicted | Distance | Confidence")
    print("-" * 50)

    for i in range(len(test_labels)):
        confidence = FaceRecognitionUtils.calculate_confidence_score(
            distances[i], recognizer.recognition_threshold, recognizer.distance_metric
        )
        print(f"     {test_labels[i]:2d}    |     {predicted_labels[i]:2d}    | "
              f"{distances[i]:8.4f} | {confidence:8.4f}")

    # Get model summary
    print("\n4. Model Summary:")
    summary = recognizer.get_model_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # Get eigenfaces
    print("\n5. Eigenfaces Information:")
    eigenfaces = recognizer.get_eigenfaces()
    print(f"   Number of eigenfaces: {len(eigenfaces)}")
    print(f"   Eigenface shape: {eigenfaces[0].shape}")

    # Face reconstruction test
    print("\n6. Face Reconstruction Test:")
    test_face = test_faces[0]
    reconstructed_face = recognizer.reconstruct_face(test_face)
    reconstruction_error = np.mean((test_face - reconstructed_face) ** 2)
    print(f"   Reconstruction MSE: {reconstruction_error:.6f}")

    print("\n✅ Eigenfaces recognizer demonstration complete!")


if __name__ == "__main__":
    main()