# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's level academic project implementing Principal Component Analysis (PCA) for face recognition from mathematical first principles. The project emphasizes rigorous mathematical derivation over black-box usage of scikit-learn.

## Development Commands

### Running Code
- **Primary**: Use `poetry run python <script>.py` for all Python execution
- **Alternative**: Use `python <script>.py` if poetry is unavailable

### Key Scripts
```bash
# Run comprehensive benchmarks (requires PCAFromScratch implementation)
poetry run python scripts/test_and_evaluate.py

# Test advanced PCA techniques
poetry run python scripts/advanced_pca_techniques.py
```

### Git Workflow
- **Main Branch Protected**: Cannot push directly to main, must use feature branches
- **Smart Git Operations**: Use `/git merge main` to merge latest changes before creating PRs
- **Always Push**: Commit and push changes to feature branches before switching contexts

## Architecture

### Core Structure
```
pca-face-recognition/
├── scripts/                     # Core implementation files
│   ├── advanced_pca_techniques.py    # 4 PCA variants (Kernel, Incremental, Optimized, Robust)
│   └── test_and_evaluate.py          # Comprehensive benchmarking suite
├── maths/                       # Mathematical foundations
│   ├── README.md                     # Complete mathematical formulations
│   └── pca_derivation.html           # Interactive mathematical proofs
├── demo/                        # Placeholder for demonstration materials
└── project-files/              # Academic project documentation
```

### Missing Core Implementation
The project currently lacks the main `scripts/pca_face_recognition.py` file which should contain:
- `PCAFromScratch` class
- `EigenfacesRecognizer` class
- Core PCA implementation from first principles

### Current Implementation Status

#### ✅ Implemented
- **Advanced PCA Techniques**: 4 complete implementations with mathematical foundations
  - KernelPCA (RBF, polynomial, sigmoid kernels)
  - IncrementalPCA (memory-efficient batch processing)
  - OptimizedPCA (randomized SVD)
  - RobustPCA (outlier handling)
- **Mathematical Documentation**: Complete derivations and proofs
- **Testing Framework**: Comprehensive benchmarking utilities

#### ❌ Missing Core Components
- **PCAFromScratch**: Base PCA implementation from first principles
- **EigenfacesRecognizer**: Main face recognition system
- **Dependency Management**: No `pyproject.toml` file
- **Working Demo**: Demonstration materials in `/demo/`

## Mathematical Approach

### First-Principles Implementation
- All PCA algorithms built from mathematical fundamentals
- No reliance on scikit-learn's PCA for core functionality
- Emphasis on eigenvalue decomposition, SVD, and covariance matrices
- Comprehensive testing against sklearn for validation only

### Key Mathematical Concepts
- **Eigenvalues & Eigenvectors**: Directions of maximum variance
- **Diagonalization**: Covariance matrix simplification
- **Kernel Trick**: Non-linear dimensionality reduction
- **Randomized SVD**: Large-scale optimization

## Testing and Evaluation

### Benchmarking Framework
- Uses Labeled Faces in the Wild (LFW) dataset
- Comprehensive performance metrics (accuracy, precision, recall, F1)
- ROC curves and multi-class evaluation
- Robustness testing (noise, illumination variations)
- Cross-validation and statistical analysis

### Visualization Output
- matplotlib-based result presentations
- Eigenfaces visualization
- Performance comparison charts
- Component selection analysis

## Academic Project Requirements

### Mathematical Rigor
- Complete step-by-step derivations required
- Educational value prioritized over performance optimization
- Team collaboration support with clear module separation
- Comprehensive evaluation metrics

### Educational Focus
- Clear explanations for learning
- Modular design for teaching concepts
- Multiple implementation approaches
- Extensive mathematical documentation

## Development Guidelines

### Code Style
- Type hints required for all function signatures
- Comprehensive docstrings with mathematical formulations
- Modular class design with clear interfaces
- Academic citation format for algorithms

### Testing Strategy
- Compare custom implementations against sklearn
- Validate mathematical correctness
- Performance benchmarking across datasets
- Robustness testing with edge cases

## Technology Stack

### Core Dependencies
- **NumPy**: Numerical computations and linear algebra
- **SciPy**: Advanced mathematical functions and optimizations
- **Matplotlib**: Visualization and result presentation
- **Seaborn**: Statistical data visualization
- **scikit-learn**: Dataset loading and comparison only

### Development Tools
- **Poetry**: Preferred dependency manager
- **Git**: Feature branch workflow with protected main
- **Claude Code**: AI-assisted development with custom configurations

## Common Tasks

### When implementing PCA variants:
1. Start from mathematical formulation in `maths/README.md`
2. Implement step-by-step using NumPy/SciPy
3. Add comprehensive docstrings with mathematical foundations
4. Test against sklearn implementation for validation
5. Add performance benchmarks to evaluation framework

### When adding new features:
1. Update mathematical documentation first
2. Implement from first principles
3. Add appropriate test cases
4. Update benchmarking framework
5. Ensure educational value is maintained
- always check streamlit official documentation for referencing HOW to implement a capability or feature in streamlit- https://docs.streamlit.io/develop/api-reference
- for streamlit components, use additional references- @demo/docs/streamlit_antd_components.md , @demo/docs/streamlit_shadcn_ui.md
- test the app, check for errors after completing a task. fix errors if any
- when you launch a new streamlit app process, sleep for 10s, check the associated logfile for any errors and fix them