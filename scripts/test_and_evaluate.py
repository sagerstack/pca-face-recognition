"""
PCA Face Recognition Testing and Evaluation Suite
Comprehensive benchmarking and performance analysis
Author: AI Math Master's Project Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           roc_curve, auc, confusion_matrix)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA as SklearnPCA
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import our custom implementations
from pca_face_recognition import PCAFromScratch, EigenfacesRecognizer
from advanced_pca_techniques import KernelPCA, IncrementalPCA, OptimizedPCA


class FaceRecognitionBenchmark:
    """
    Comprehensive benchmarking suite for face recognition algorithms
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, image_shape):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.image_shape = image_shape
        self.results = {}
        
    def benchmark_pca_implementations(self):
        """
        Compare different PCA implementations
        """
        print("\n" + "="*60)
        print("BENCHMARKING PCA IMPLEMENTATIONS")
        print("="*60)
        
        implementations = {
            'Custom PCA': PCAFromScratch(n_components=50),
            'Sklearn PCA': SklearnPCA(n_components=50),
            'Optimized PCA': OptimizedPCA(n_components=50, method='randomized'),
            'Incremental PCA': IncrementalPCA(n_components=50, batch_size=100)
        }
        
        benchmark_results = {}
        
        for name, pca_impl in implementations.items():
            print(f"\nTesting {name}...")
            
            # Measure fitting time
            start_time = time.time()
            
            if name == 'Sklearn PCA':
                pca_impl.fit(self.X_train)
                X_train_transformed = pca_impl.transform(self.X_train)
                X_test_transformed = pca_impl.transform(self.X_test)
            else:
                X_train_transformed = pca_impl.fit_transform(self.X_train)
                X_test_transformed = pca_impl.transform(self.X_test)
            
            fit_time = time.time() - start_time
            
            # Train simple classifier on transformed data
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(X_train_transformed, self.y_train)
            
            # Measure prediction time
            start_time = time.time()
            y_pred = clf.predict(X_test_transformed)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )
            
            benchmark_results[name] = {
                'fit_time': fit_time,
                'pred_time': pred_time,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"  Fit time: {fit_time:.3f}s")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  F1 Score: {f1:.3f}")
        
        self.results['pca_implementations'] = benchmark_results
        return benchmark_results
    
    def evaluate_component_selection(self, component_range=None):
        """
        Evaluate performance across different numbers of components
        """
        print("\n" + "="*60)
        print("EVALUATING COMPONENT SELECTION")
        print("="*60)
        
        if component_range is None:
            component_range = [10, 20, 30, 40, 50, 75, 100, 150]
        
        metrics = {
            'n_components': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'variance_explained': [],
            'fit_time': [],
            'reconstruction_error': []
        }
        
        for n_comp in component_range:
            print(f"\nTesting with {n_comp} components...")
            
            # Fit PCA
            start_time = time.time()
            pca = PCAFromScratch(n_components=n_comp)
            pca.fit(self.X_train)
            fit_time = time.time() - start_time
            
            # Transform data
            X_train_pca = pca.transform(self.X_train)
            X_test_pca = pca.transform(self.X_test)
            
            # Train recognizer
            recognizer = EigenfacesRecognizer(n_components=n_comp)
            recognizer.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = recognizer.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )
            
            # Calculate variance explained
            var_explained = np.sum(pca.explained_variance_ratio_)
            
            # Calculate reconstruction error
            recon_error = pca.calculate_reconstruction_error(self.X_test[:100])
            
            # Store metrics
            metrics['n_components'].append(n_comp)
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)
            metrics['variance_explained'].append(var_explained)
            metrics['fit_time'].append(fit_time)
            metrics['reconstruction_error'].append(recon_error)
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Variance Explained: {var_explained:.3f}")
        
        self.results['component_selection'] = metrics
        return pd.DataFrame(metrics)
    
    def evaluate_distance_metrics(self):
        """
        Compare different distance metrics for face matching
        """
        print("\n" + "="*60)
        print("EVALUATING DISTANCE METRICS")
        print("="*60)
        
        distance_metrics = ['euclidean', 'cosine']
        n_components = 50
        
        metric_results = {}
        
        for metric in distance_metrics:
            print(f"\nTesting {metric} distance...")
            
            # Train recognizer
            recognizer = EigenfacesRecognizer(
                n_components=n_components,
                distance_metric=metric
            )
            recognizer.fit(self.X_train, self.y_train)
            
            # Predict with confidence
            y_pred, confidences = recognizer.predict_with_confidence(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )
            
            metric_results[metric] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences)
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Mean Confidence: {np.mean(confidences):.3f}")
        
        self.results['distance_metrics'] = metric_results
        return metric_results
    
    def evaluate_robustness(self):
        """
        Evaluate robustness to various transformations
        """
        print("\n" + "="*60)
        print("EVALUATING ROBUSTNESS")
        print("="*60)
        
        # Train baseline recognizer
        recognizer = EigenfacesRecognizer(n_components=50)
        recognizer.fit(self.X_train, self.y_train)
        
        # Baseline accuracy
        y_pred_baseline = recognizer.predict(self.X_test)
        baseline_accuracy = accuracy_score(self.y_test, y_pred_baseline)
        
        robustness_results = {
            'baseline': baseline_accuracy
        }
        
        # Test with noise
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        for noise_level in noise_levels:
            X_test_noisy = self.X_test + np.random.randn(*self.X_test.shape) * noise_level
            y_pred_noisy = recognizer.predict(X_test_noisy)
            accuracy = accuracy_score(self.y_test, y_pred_noisy)
            robustness_results[f'noise_{noise_level}'] = accuracy
            print(f"  Noise (σ={noise_level}): {accuracy:.3f}")
        
        # Test with brightness changes
        brightness_factors = [0.7, 0.85, 1.15, 1.3]
        for factor in brightness_factors:
            X_test_bright = self.X_test * factor
            X_test_bright = np.clip(X_test_bright, 0, 1)  # Assuming normalized data
            y_pred_bright = recognizer.predict(X_test_bright)
            accuracy = accuracy_score(self.y_test, y_pred_bright)
            robustness_results[f'brightness_{factor}'] = accuracy
            print(f"  Brightness (×{factor}): {accuracy:.3f}")
        
        self.results['robustness'] = robustness_results
        return robustness_results
    
    def perform_cross_validation(self, n_folds=5):
        """
        Perform k-fold cross-validation
        """
        print("\n" + "="*60)
        print(f"PERFORMING {n_folds}-FOLD CROSS-VALIDATION")
        print("="*60)
        
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline
        from sklearn.neighbors import KNeighborsClassifier
        
        # Create pipeline
        pipeline = Pipeline([
            ('pca', SklearnPCA(n_components=50)),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ])
        
        # Combine train and test for cross-validation
        X_all = np.vstack([self.X_train, self.X_test])
        y_all = np.hstack([self.y_train, self.y_test])
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X_all, y_all, cv=n_folds, 
                                   scoring='accuracy')
        
        cv_results = {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'min': np.min(cv_scores),
            'max': np.max(cv_scores)
        }
        
        print(f"  Mean Accuracy: {cv_results['mean']:.3f} ± {cv_results['std']:.3f}")
        print(f"  Min/Max: {cv_results['min']:.3f} / {cv_results['max']:.3f}")
        
        self.results['cross_validation'] = cv_results
        return cv_results
    
    def generate_roc_curves(self):
        """
        Generate ROC curves for multi-class classification
        """
        print("\n" + "="*60)
        print("GENERATING ROC CURVES")
        print("="*60)
        
        # Train recognizer
        recognizer = EigenfacesRecognizer(n_components=50)
        recognizer.fit(self.X_train, self.y_train)
        
        # Get unique classes
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        
        # Binarize labels
        y_test_bin = label_binarize(self.y_test, classes=classes)
        
        # For ROC, we need probability scores, so we'll compute distances
        X_test_pca = recognizer.pca.transform(self.X_test)
        
        # Compute decision scores (negative distances)
        decision_scores = np.zeros((len(self.X_test), n_classes))
        
        for i, test_point in enumerate(X_test_pca):
            for j, class_label in enumerate(classes):
                # Get all training points of this class
                class_mask = recognizer.training_labels == class_label
                class_projections = recognizer.training_projections[class_mask]
                
                # Find minimum distance to this class
                distances = np.linalg.norm(class_projections - test_point, axis=1)
                decision_scores[i, j] = -np.min(distances)  # Negative distance as score
        
        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], decision_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        self.results['roc_curves'] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
        
        return fpr, tpr, roc_auc


def visualize_benchmark_results(benchmark: FaceRecognitionBenchmark):
    """
    Create comprehensive visualizations of benchmark results
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. PCA Implementation Comparison
    if 'pca_implementations' in benchmark.results:
        impl_results = benchmark.results['pca_implementations']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract data
        impl_names = list(impl_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'fit_time', 'pred_time']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = [impl_results[name][metric] for name in impl_names]
            
            bars = ax.bar(range(len(impl_names)), values, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_xlabel('Implementation')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(range(len(impl_names)))
            ax.set_xticklabels(impl_names, rotation=45, ha='right')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom')
        
        plt.suptitle('PCA Implementation Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # 2. Component Selection Analysis
    if 'component_selection' in benchmark.results:
        comp_results = benchmark.results['component_selection']
        df = pd.DataFrame(comp_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy vs Components
        axes[0, 0].plot(df['n_components'], df['accuracy'], 'b-', marker='o')
        axes[0, 0].set_xlabel('Number of Components')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Number of Components')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Variance Explained vs Components
        axes[0, 1].plot(df['n_components'], df['variance_explained'], 'g-', marker='s')
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Variance Explained')
        axes[0, 1].set_title('Variance Explained vs Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reconstruction Error vs Components
        axes[1, 0].plot(df['n_components'], df['reconstruction_error'], 'r-', marker='^')
        axes[1, 0].set_xlabel('Number of Components')
        axes[1, 0].set_ylabel('Reconstruction Error')
        axes[1, 0].set_title('Reconstruction Error vs Components')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score vs Components
        axes[1, 1].plot(df['n_components'], df['f1_score'], 'm-', marker='d')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 Score vs Components')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Component Selection Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # 3. Robustness Analysis
    if 'robustness' in benchmark.results:
        rob_results = benchmark.results['robustness']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Noise robustness
        noise_keys = [k for k in rob_results.keys() if 'noise' in k]
        noise_levels = [float(k.split('_')[1]) for k in noise_keys]
        noise_accuracies = [rob_results[k] for k in noise_keys]
        
        axes[0].plot([0] + noise_levels, 
                    [rob_results['baseline']] + noise_accuracies,
                    'b-', marker='o', markersize=8)
        axes[0].axhline(y=rob_results['baseline'], color='g', linestyle='--', 
                       label='Baseline')
        axes[0].set_xlabel('Noise Level (σ)')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Robustness to Noise')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Brightness robustness
        bright_keys = [k for k in rob_results.keys() if 'brightness' in k]
        bright_factors = [float(k.split('_')[1]) for k in bright_keys]
        bright_accuracies = [rob_results[k] for k in bright_keys]
        
        axes[1].plot([1.0] + bright_factors,
                    [rob_results['baseline']] + bright_accuracies,
                    'r-', marker='s', markersize=8)
        axes[1].axhline(y=rob_results['baseline'], color='g', linestyle='--',
                       label='Baseline')
        axes[1].set_xlabel('Brightness Factor')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Robustness to Brightness Changes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Robustness Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # 4. ROC Curves
    if 'roc_curves' in benchmark.results:
        roc_data = benchmark.results['roc_curves']
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(min(5, len(roc_data['fpr']))):  # Show first 5 classes
            plt.plot(roc_data['fpr'][i], roc_data['tpr'][i],
                    label=f'Class {i} (AUC = {roc_data["auc"][i]:.2f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Face Recognition')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()


def generate_performance_report(benchmark: FaceRecognitionBenchmark):
    """
    Generate a comprehensive performance report
    """
    print("\n" + "="*60)
    print("PERFORMANCE REPORT")
    print("="*60)
    
    report = []
    
    # 1. Best PCA Implementation
    if 'pca_implementations' in benchmark.results:
        impl_results = benchmark.results['pca_implementations']
        best_impl = max(impl_results.items(), key=lambda x: x[1]['accuracy'])
        report.append(f"Best PCA Implementation: {best_impl[0]}")
        report.append(f"  - Accuracy: {best_impl[1]['accuracy']:.3f}")
        report.append(f"  - F1 Score: {best_impl[1]['f1_score']:.3f}")
        report.append(f"  - Fit Time: {best_impl[1]['fit_time']:.3f}s")
        report.append("")
    
    # 2. Optimal Number of Components
    if 'component_selection' in benchmark.results:
        comp_df = pd.DataFrame(benchmark.results['component_selection'])
        
        # Find optimal based on accuracy-variance trade-off
        comp_df['score'] = comp_df['accuracy'] * comp_df['variance_explained']
        optimal_idx = comp_df['score'].idxmax()
        optimal_components = comp_df.loc[optimal_idx, 'n_components']
        
        report.append(f"Optimal Number of Components: {optimal_components}")
        report.append(f"  - Accuracy: {comp_df.loc[optimal_idx, 'accuracy']:.3f}")
        report.append(f"  - Variance Explained: {comp_df.loc[optimal_idx, 'variance_explained']:.3f}")
        report.append("")
    
    # 3. Best Distance Metric
    if 'distance_metrics' in benchmark.results:
        metric_results = benchmark.results['distance_metrics']
        best_metric = max(metric_results.items(), key=lambda x: x[1]['accuracy'])
        report.append(f"Best Distance Metric: {best_metric[0]}")
        report.append(f"  - Accuracy: {best_metric[1]['accuracy']:.3f}")
        report.append(f"  - Mean Confidence: {best_metric[1]['mean_confidence']:.3f}")
        report.append("")
    
    # 4. Cross-Validation Results
    if 'cross_validation' in benchmark.results:
        cv_results = benchmark.results['cross_validation']
        report.append(f"Cross-Validation Results ({len(cv_results['scores'])}-fold):")
        report.append(f"  - Mean Accuracy: {cv_results['mean']:.3f} ± {cv_results['std']:.3f}")
        report.append(f"  - Range: [{cv_results['min']:.3f}, {cv_results['max']:.3f}]")
        report.append("")
    
    # 5. Robustness Summary
    if 'robustness' in benchmark.results:
        rob_results = benchmark.results['robustness']
        report.append("Robustness Summary:")
        report.append(f"  - Baseline Accuracy: {rob_results['baseline']:.3f}")
        
        # Average noise robustness
        noise_keys = [k for k in rob_results.keys() if 'noise' in k]
        if noise_keys:
            avg_noise = np.mean([rob_results[k] for k in noise_keys])
            report.append(f"  - Average Noise Robustness: {avg_noise:.3f}")
        
        # Average brightness robustness
        bright_keys = [k for k in rob_results.keys() if 'brightness' in k]
        if bright_keys:
            avg_bright = np.mean([rob_results[k] for k in bright_keys])
            report.append(f"  - Average Brightness Robustness: {avg_bright:.3f}")
    
    # Print report
    for line in report:
        print(line)
    
    return report


def main():
    """
    Main testing and evaluation pipeline
    """
    print("="*60)
    print("PCA FACE RECOGNITION - TESTING & EVALUATION SUITE")
    print("="*60)
    
    # Load dataset
    print("\nLoading face dataset...")
    lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    image_shape = lfw_people.images[0].shape
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Number of people: {len(target_names)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Initialize benchmark
    benchmark = FaceRecognitionBenchmark(
        X_train, X_test, y_train, y_test, image_shape
    )
    
    # Run all benchmarks
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE BENCHMARKS")
    print("="*60)
    
    # 1. Compare PCA implementations
    benchmark.benchmark_pca_implementations()
    
    # 2. Evaluate component selection
    comp_df = benchmark.evaluate_component_selection()
    
    # 3. Evaluate distance metrics
    benchmark.evaluate_distance_metrics()
    
    # 4. Evaluate robustness
    benchmark.evaluate_robustness()
    
    # 5. Cross-validation
    benchmark.perform_cross_validation()
    
    # 6. Generate ROC curves
    benchmark.generate_roc_curves()
    
    # Visualize results
    visualize_benchmark_results(benchmark)
    
    # Generate performance report
    report = generate_performance_report(benchmark)
    
    # Save results to CSV
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save component selection results
    if 'component_selection' in benchmark.results:
        comp_df = pd.DataFrame(benchmark.results['component_selection'])
        comp_df.to_csv('/mnt/user-data/outputs/component_selection_results.csv', index=False)
        print("Component selection results saved to: component_selection_results.csv")
    
    # Save implementation comparison
    if 'pca_implementations' in benchmark.results:
        impl_df = pd.DataFrame(benchmark.results['pca_implementations']).T
        impl_df.to_csv('/mnt/user-data/outputs/implementation_comparison.csv')
        print("Implementation comparison saved to: implementation_comparison.csv")
    
    print("\n" + "="*60)
    print("TESTING & EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
