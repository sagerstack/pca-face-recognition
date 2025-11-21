# US-001: PCA Facial Recognition Streamlit Demo

## User Story

**As a** Master's student evaluating PCA for face recognition,
**I want to** interactively train a PCA model with adjustable parameters and test it on a known dataset,
**So that** I can demonstrate my understanding of the mathematical concepts through a working facial recognition system for academic evaluation.

## Acceptance Criteria

### AC-001: Core Implementation (One Class Per File)
- **PCA** class (renamed from PCAFromScratch) in `demo/src/core/pca.py` implementing PCA from first principles
- **EigenfacesRecognizer** class in `demo/src/core/eigenfaces_recognizer.py` for face recognition using eigenfaces method
- **FaceProcessor** class in `demo/src/processing/face_processor.py` for face detection, preprocessing, and standardization
- **VisualizationUtils** class in `demo/src/visualization/visualizations.py` for plotting and chart generation
- **MathematicalUtils** class in `demo/src/core/mathematical_utils.py` for mathematical helper functions
- **DatasetLoader** class in `demo/src/processing/dataset_loader.py` for AT&T dataset handling
- Mathematical approach based on covariance matrix: `C = (1/n) Σ(x_i - μ)(x_i - μ)^T`
- Eigendecomposition using `scipy.linalg.eigh` following existing notebook methodology
- Poetry dependency management with `pyproject.toml` in demo root directory

### AC-002: Dataset Integration
- **AT&T Dataset Structure**:
  - 40 subjects (labeled s1, s2, ..., s40)
  - 10 images per subject = 400 total images
  - Image dimensions: 92 × 112 pixels (grayscale)
  - **Mathematical representation**: Images: I_k ∈ ℝ^(92×112) where k = 1..400
  - **Total dataset**: D = {I_1, I_2, ..., I_400}
- Train-test split functionality with configurable training size per subject
- Proper labeling and data organization following existing notebook structure

### AC-003: Multi-Page Streamlit Application (3-Page Architecture)
- **Page 1 - src/pages/1_Eigenfaces.py**: Eigenfaces workflow with 5-tab navigation
- **Page 2 - src/pages/2_Face_Recognition.py**: Face recognition workflow with tabbed navigation
- **Page 3 - src/pages/3_Face_Verification.py**: Face verification workflow with 3-tab navigation
- **Tabbed workflow navigation** with Next button progression on each page
- **Real-time parameter adjustment** through interactive controls
- **Model Sharing**: Trained model automatically available to recognition and verification pages

### AC-004: Interactive Parameter Controls
- **PCA component count slider** (1-10304 components, default 50)
- **Training size selector** per subject (1-9 images, default 6)
- **Distance metric selector** (Euclidean, Cosine similarity)
- **Real-time updates** when parameters change

### AC-005: Eigenfaces Page Workflow (4 Sequential Tabs)
**Navigation**: Sequential tabs with Previous/Next buttons, users can go back to update settings
**State Management**: All parameter settings preserved across tabs
**Purpose**: Build PCA eigenfaces model and demonstrate mathematical concepts through visual interpretations of real face subjects

**Tab 1: Dataset Loading & Understanding**
**Mathematical Concept**: **Data Representation & Labeling**
- **Theory**: High-dimensional data representation, train/test split strategy
- **Formula**: `X ∈ ℝ^(n×10304)`, `y ∈ ℝ^n` where 10304 = 92×112 pixels flattened
- **Visual Interpretation**:
  - **Dataset Structure Visualization**: Display AT&T dataset organization (40 subjects s1-s40, 10 images each)
  - **Subject Gallery**: Show face samples from different subjects demonstrating natural variability (lighting, expression, glasses)
  - **Train/Test Split Visualization**: Side-by-side display of training vs. test images with subject labels
  - **Data Matrix Representation**: Visual explanation of faces as rows and pixels as columns
  - **Training Size Selection**: Slider to choose training images per subject (1-9, default 6)
- **Progress Check**: Validate dataset integrity and show summary statistics (total faces, subjects, dimensions)
- **Next Button**: Enable after dataset loading and understanding is complete

**Tab 2: PCA Configuration & Mathematical Foundations**
**Mathematical Concept**: **Mean Centering & Covariance Matrix**
- **Theory**: Removing mean bias, capturing feature correlations and variance structure
- **Formula**: `X_centered = X - μ` where `μ = (1/n) Σx_i`, `C = cov(X_centered.T)`
- **Visual Interpretation**:
  - **Mean Face Display**: The computed average face `μ` across all 40 subjects - visual mathematical mean
  - **Mean Centering Effects**: Side-by-side comparison showing original faces vs. mean-centered faces (ghostly appearance)
  - **Covariance Matrix Heatmap**: Visual representation of feature correlations (subset of 10304×10304 matrix)
  - **Matrix Dimension Preview**: Interactive display of computational complexity and memory requirements
  - **Component Count Selection**: Slider for number of PCA components (1-10304, default 50)
  - **Parameter Summary**: Display mathematical implications of component choices
- **Mathematical Explanation**: Hover tooltips showing formulas and their visual impact on face processing
- **Next Button**: Enable after mathematical configuration is applied

**Tab 3: Eigenface Generation & Principal Components**
**Mathematical Concept**: **Eigenvalue Problem & Principal Components**
- **Theory**: Finding directions of maximum variance through eigenvalue decomposition
- **Formula**: `Cw = λw` where eigenfaces are eigenvectors, eigenvalues are variance magnitudes
- **Visual Interpretation**:
  - **Mean Face Gallery**: Central average face computed as `μ = (1/n) Σx_i` - mathematical foundation of PCA
  - **Eigenfaces Visualization**: Gallery of top eigenfaces with:
    - Component numbers and variance explained percentages (`λ_i / Σλ`)
    - Visual explanations of what patterns each eigenface captures (lighting, pose, facial features)
    - Subject-specific contributions showing which faces influence each eigenface most
  - **Variance Explained Plot**: Interactive cumulative variance plot with face reconstruction examples at different component levels
  - **Eigenvalue Distribution**: Bar chart showing eigenvalue magnitudes with face quality correlation
  - **Component Selection Impact**: Real-time slider showing how many components capture desired variance thresholds (95%, 99%)
- **Interactive Learning**: Click eigenfaces to see mathematical formula `Cw = λw` in action with specific face examples
- **Next Button**: Enable after eigenface generation and understanding

**Tab 4: Facial Reconstruction & Mathematical Interpretation**
**Mathematical Concept**: **Projection & Reconstruction**
- **Theory**: Dimensionality reduction and approximation using principal components
- **Formula**: `x_reconstructed = μ + Σ(x · w_i) * w_i` for i=1 to k components
- **Visual Interpretation**:
  - **Reconstruction Gallery**: Side-by-side comparison for different component counts (5, 10, 50, 100, 1000) showing mathematical fidelity
  - **Progressive Animation**: Real-time visualization of face reconstruction improving as components increase, demonstrating `Σ(x · w_i) * w_i` accumulation
  - **Subject-Specific Analysis**: Show how different subjects (s1-s40) reconstruct at various component levels with error patterns
  - **Component Contribution Visualization**: Interactive display showing which eigenfaces contribute most to specific face reconstructions
  - **Error Visualization**: Display reconstruction errors as difference images with mathematical `||x - x_reconstructed||` calculations
  - **Mathematical Quality Metrics**: Interactive sliders showing MSE, Euclidean distance, and their relationship to component count
  - **Component Trade-off Analysis**: Visual demonstration of bias-variance trade-off through face reconstruction examples
- **Mathematical Formula Display**: Interactive formula `x_reconstructed = μ + Σ(x · w_i) * w_i` with real-time parameter updates
- **Complete Button**: Finish eigenfaces workflow with mathematical understanding and prepare model for recognition/verification pages

### AC-006: Face Recognition Page Workflow (Sequential Tabs)
**Navigation**: Sequential tabs with Previous/Next buttons, users can go back to update settings
**State Management**: All parameters and results preserved across tabs
**Auto-Model Loading**: Automatically uses trained model from Model Training page
**Purpose**: Test trained model on held-out test images and measure recognition accuracy

**Tab 1: Model Information**
- **Trained Model Display**: Show details of model automatically loaded from Model Training page
- **Training Parameters**: Display component count, distance metric, training size used
- **Model Statistics**: Processing time, memory usage, eigenface information
- **Training Completion Confirmation**: Confirm model is ready for recognition testing
- **Next Button**: Enable after model validation

**Tab 2: Test Image Selection**
- **Test Dataset Preview**: Display available test images from held-out set
- **Image Selection**: Choose specific test image(s) from dataset using gallery view
- **Ground Truth Display**: Show actual identity label for selected test image
- **Test Configuration**: Configure batch processing or single image testing
- **Selection Summary**: Display selected test images and their ground truth labels
- **Next Button**: Enable after test image selection

**Tab 3: Recognition Testing**
- **Recognition Progress**: Real-time progress during face recognition testing
- **Individual Results**: Show recognition results for each test image with:
  - **Predicted Identity**: Which subject the model thinks this is
  - **Confidence Score**: Distance-based confidence metric
  - **Ground Truth Comparison**: Actual vs. predicted identity
  - **Success/Failure Indicator**: Visual indicator for correct/incorrect predictions
- **Batch Processing**: Process multiple test images with progress tracking
- **Next Button**: Enable after recognition testing completion

**Tab 4: Recognition Analysis & Results**
- **Overall Accuracy**: Final recognition accuracy percentage across test set
- **Confusion Matrix**: Visual heatmap of predicted vs. actual labels
- **Error Analysis**: Show misclassified examples with detailed analysis
- **Performance Metrics**: Per-subject accuracy, processing time, confidence distributions
- **Recognition Comparison**: Compare results with different component counts or distance metrics
- **Export Results**: Option to save detailed recognition results and visualizations
- **Complete Button**: Finish face recognition workflow

### AC-007: Face Verification Page Workflow (3 Sequential Tabs)
**Navigation**: Sequential tabs with Previous/Next buttons, users can go back to update settings
**State Management**: All parameters and results preserved across tabs
**Auto-Model Loading**: Automatically uses trained model from Model Training page
**Purpose**: Verify if two uploaded images show the same person using PCA feature comparison

**Tab 1: Model Information**
- **Trained Model Display**: Show details of model automatically loaded from Model Training page
- **Training Parameters**: Display component count, distance metric, training size used
- **Model Statistics**: Processing time, memory usage, eigenface information
- **Verification Explanation**: Brief overview of face verification process using PCA features
- **Next Button**: Enable after model validation

**Tab 2: Face Upload & Processing**
- **Upload Image 1**: Reference image via file uploader or camera input
- **Upload Image 2**: Test image via file uploader or camera input
- **Face Detection**: Use OpenCV Haar cascades to detect and extract faces
- **Image Preprocessing**:
  - **Resize**: Standardize to 92x112 pixels (same as AT&T dataset)
  - **Grayscale Conversion**: Convert to grayscale for PCA compatibility
  - **Histogram Equalization**: Normalize lighting conditions using OpenCV
  - **Face Alignment**: Eye-level alignment if detectable (optional enhancement)
- **Quality Check**: Validate face detection success and image quality
- **Preprocessing Preview**: Show processed faces side-by-side
- **Next Button**: Enable after successful processing

**Tab 3: Face Verification Results**
- **Feature Extraction**: Apply trained PCA model to extract facial features from both images
- **Distance Calculation**: Compute Euclidean distance between feature vectors
- **Verification Decision**:
  - **Same Person** if distance < threshold (estimated ~50-100 units based on eigenface space)
  - **Different Person** if distance > threshold
- **Confidence Score**: Normalized confidence based on distance from threshold
- **Visual Comparison**: Side-by-side display of processed face images with feature projections
- **Detailed Metrics**: Show distance value, threshold used, processing time
- **Technical Explanation**: Brief explanation of how PCA feature comparison works
- **Export Results**: Option to save verification results and processed images
- **Complete Button**: Finish face verification workflow

### AC-008: Tabbed Workflow State Management
- **Sequential Progress**: Users must complete tabs in order, Previous button allows back navigation
- **Parameter Preservation**: All settings maintained when navigating between tabs
- **Progress Indicators**: Visual progress bar showing current tab position
- **State Validation**: Ensure data integrity when switching between tabs
- **Memory Efficiency**: Cache computed results to avoid re-computation
- **Error Recovery**: Graceful handling of invalid states with clear guidance

### AC-009: Workflow Visual Diagram

**Page 1: Eigenfaces Flow (4 Tabs):**
```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Tab 1: Dataset    │───▶│  Tab 2: PCA         │───▶│   Tab 3: Eigenface  │───▶│ Tab 4: Facial      │
│   Loading &         │    │  Configuration &   │    │   Generation &     │    │   Reconstruction   │
│   Understanding      │    │  Mathematical      │    │   Principal        │    │   & Mathematical   │
│                     │    │  Foundations       │    │   Components       │    │   Interpretation    │
│                     │    │                     │    │                     │    │                     │
│ • Subject gallery    │    │ • Mean face μ        │    │ • Mean face gallery│    │ • Reconstruction    │
│ • Train/test split   │    │ • Mean centering     │    │ • Eigenfaces viz    │    │   gallery           │
│ • Data matrix X     │    │ • Covariance C      │    │ • Variance plot     │    │ • Progressive       │
│ • Mathematical      │    │ • Matrix dimensions │    │ • Eigenvalues λ     │    │   animation         │
│   formulas          │    │ • Component count   │    │ • Formula Cw=λw    │    │ • Error metrics       │
│ • Training size     │    │ • Real-time math     │    │ • Interactive       │    │ • Subject-specific   │
│ • Next ➡️           │    │ • explanations      │    │   learning          │    │   analysis          │
└─────────┬───────────┘    └─────────┬───────────┘    └─────────┬───────────┘    └─────────┬───────────┘
          │                          │                          │                          │
          ▼                          ▼                          ▼                          ▼
    ⬅️ Previous                ⬅️ Previous                ⬅️ Previous                ⬅️ Previous
```

**Page 2: Face Recognition Flow (4 Tabs):**
```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Tab 1: Model      │───▶│   Tab 2: Test       │───▶│   Tab 3: Recognition│───▶│ Tab 4: Analysis     │
│   Information       │    │   Image Selection   │    │     Testing         │    │   & Results         │
│                     │    │                     │    │                     │    │                     │
│ • Model info        │    │ • Test dataset      │    │ • Progress bar      │    │ • Accuracy metrics  │
│ • Training params   │    │ • Image gallery     │    │ • Individual results│    │ • Confusion matrix  │
│ • Performance stats │    │ • Ground truth      │    │ • Success/failure   │    │ • Error analysis    │
│ • Next ➡️           │    │ • Selection & Next ➡️ │    │ • Next ➡️           │    │ • Complete ✓        │
└─────────┬───────────┘    └─────────┬───────────┘    └─────────┬───────────┘    └─────────┬───────────┘
          │                          │                          │                          │
          ▼                          ▼                          ▼                          ▼
    ⬅️ Previous                ⬅️ Previous                ⬅️ Previous                ⬅️ Previous
```

**Page 3: Face Verification Flow (3 Tabs):**
```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Tab 1: Model      │───▶│   Tab 2: Face       │───▶│ Tab 3: Verification  │
│   Information       │    │   Upload            │    │   Results           │
│                     │    │                     │    │                     │
│ • Model info        │    │ • Upload Image 1    │    │ • Distance metrics  │
│ • Training params   │    │ • Upload Image 2    │    │ • Same/different    │
│ • Performance stats │    │ • Face detection    │    │ • Confidence score  │
│ • Verification exp  │    │ • Preprocessing     │    │ • Feature comparison│
│ • Next ➡️           │    │ • Verify & Next ➡️   │    │ • Complete ✓        │
└─────────┬───────────┘    └─────────┬───────────┘    └─────────┬───────────┘
          │                          │                          │
          ▼                          ▼                          ▼
    ⬅️ Previous                ⬅️ Previous                ⬅️ Previous
```

**Cross-Page Workflow:**
1. **Eigenfaces**: Build PCA eigenfaces model from AT&T dataset with comprehensive preprocessing and reconstruction
2. **Face Recognition**: Test model on held-out test images for accuracy evaluation
3. **Face Verification**: Use trained model to verify if two uploaded images show same person

**State Management:**
- **Global Session State**: Maintains all parameters across tab navigation and page transitions
- **Model Sharing**: Trained model automatically available to Face Recognition and Face Verification pages
- **Caching Strategy**: Computed results cached to prevent re-computation
- **Validation Layer**: Ensures data integrity when switching tabs and pages
- **Progress Tracking**: Visual indicators for workflow completion status across all pages

### AC-010: Visualization Capabilities
- **Eigenfaces gallery** showing top N eigenfaces with component numbers
- **Image reconstruction comparison** (original vs. reconstructed with different component counts)
- **Accuracy vs. components plot** (real-time updates)
- **Mean squared error analysis** (logarithmic and linear scales)
- **Confusion matrix** for recognition performance

### AC-011: Performance Metrics
- **Recognition accuracy** for different component counts
- **Mean squared error** for reconstruction quality
- **Cumulative variance explained** by principal components
- **Processing time** metrics for training and inference
- **Confidence scoring** for recognition predictions

### AC-012: Mathematical Transparency
- **Step-by-step PCA process** visualization with mathematical formulas
- **Eigenvalue and eigenvector explanations** with visual demonstrations
- **Covariance matrix computation** details
- **Reconstruction error analysis** with mathematical foundations

### AC-013: Educational Features
- **Mathematical formula display** using LaTeX rendering in Streamlit
- **Component selection guidance** based on variance explained thresholds
- **Real-time explanations** of what each parameter does
- **Comparison with existing notebook results** for validation

## Technical Requirements

### TR-001: Dependencies
```toml
[tool.poetry.dependencies]
python = "^3.8"
streamlit = "^1.28"
numpy = "^1.24"
scipy = "^1.10"
matplotlib = "^3.7"
seaborn = "^0.12"
plotly = "^5.15"
pillow = "^10.0"
scikit-learn = "^1.3"  # For dataset loading only
opencv-python = "^4.8"  # For face detection and image processing
```

### TR-002: File Structure (One Class Per File)
```
demo/
├── pyproject.toml             # Poetry dependency management
├── streamlit_app.py           # Multi-page entry point
├── src/                       # All source code
│   ├── __init__.py           # Source package initialization
│   ├── pages/                # Streamlit pages
│   │   ├── __init__.py       # Pages package initialization
│   │   ├── 1_Eigenfaces.py       # Page 1: Eigenfaces workflow
│   │   ├── 2_Face_Recognition.py # Page 2: Face recognition workflow
│   │   └── 3_Face_Verification.py # Page 3: Face verification workflow
│   ├── core/                 # Core PCA and mathematical computations
│   │   ├── __init__.py       # Core package initialization
│   │   ├── pca.py            # PCA class (from first principles)
│   │   ├── eigenfaces_recognizer.py  # EigenfacesRecognizer class
│   │   └── mathematical_utils.py     # Mathematical helper functions
│   ├── processing/           # Data processing and face handling
│   │   ├── __init__.py       # Processing package initialization
│   │   ├── face_processor.py # FaceProcessor class for face detection and preprocessing
│   │   ├── dataset_loader.py # AT&T-specific loader
│   │   └── image_utils.py    # Image processing utilities
│   ├── visualization/        # Visualization and plotting
│   │   ├── __init__.py       # Visualization package initialization
│   │   ├── visualizations.py # VisualizationUtils class for plotting
│   │   ├── chart_utils.py    # Chart and plotting utilities
│   │   └── eigenfaces_viz.py # Eigenfaces-specific visualizations
│   └── utils/                # Utility functions
│       ├── __init__.py       # Utils package initialization
│       ├── streamlit_utils.py# Streamlit-specific utilities and components
│       └── file_utils.py     # File handling utilities
├── data/
│   ├── __init__.py           # Data package initialization
│   └── ATnT/                 # AT&T face dataset
├── assets/
│   └── sample_faces/         # Demo face images
└── docs/
    ├── us-001-pca-facial-recognition.md  # This user story
    └── implementation-plan.md             # Detailed implementation plan
```

### TR-003: Mathematical Accuracy
- Follow existing notebook's mathematical approach exactly
- Use same covariance matrix computation method
- Implement same eigenface selection and sorting logic
- Reproduce accuracy and MSE analysis methods

## Success Metrics

### SM-001: Functional Success
- **Accuracy**: 85-92% with 50 components (matches existing notebook)
- **Processing Speed**: <0.1s per face for recognition
- **Memory Usage**: 80% reduction with PCA dimensionality
- **Robustness**: Handles different training sizes (1-9 images per subject)

### SM-002: Educational Success
- **Interactive Learning**: Users can adjust parameters and see immediate results
- **Mathematical Understanding**: Clear visualization of PCA concepts
- **Comparison Capability**: Side-by-side comparison of different component counts
- **Reproducibility**: Results match existing notebook implementation

## Non-Functional Requirements

### NFR-001: Performance
- **Real-time Interaction**: Parameter changes update visualizations within 2 seconds
- **Dataset Loading**: AT&T dataset loads within 5 seconds
- **Model Training**: PCA training completes within 10 seconds
- **Memory Efficiency**: Handle full AT&T dataset (400 images) efficiently

### NFR-002: Usability
- **Intuitive Navigation**: Clear tab progression with Next buttons
- **Responsive Design**: Works on different screen sizes
- **Error Handling**: Graceful handling of invalid inputs
- **Loading Indicators**: Visual feedback during computations

### NFR-003: Maintainability
- **Modular Design**: Clear separation of concerns across components
- **Documentation**: Comprehensive docstrings for all classes and methods
- **Type Hints**: Full type annotation for better code clarity
- **Error Logging**: Proper error handling and logging mechanisms

## Definition of Done

A feature is considered complete when:
- [ ] All acceptance criteria are met
- [ ] Code follows project coding standards
- [ ] Comprehensive testing validates functionality
- [ ] Documentation is complete and up-to-date
- [ ] Performance meets specified metrics
- [ ] Mathematical accuracy verified against existing notebook
- [ ] User experience is smooth and intuitive
- [ ] Error handling covers edge cases
- [ ] Integration testing passes with AT&T dataset
- [ ] Academic evaluation requirements are satisfied

## Risk Mitigation

### RM-001: Technical Risks
- **Mathematical Accuracy**: Continuously validate against existing notebook
- **Performance Issues**: Implement caching for expensive computations
- **Memory Constraints**: Use efficient NumPy operations and lazy loading
- **Dataset Issues**: Include fallback dataset loading mechanisms

### RM-002: Usability Risks
- **Complex Interface**: Conduct user testing with fellow students
- **Parameter Overload**: Provide sensible defaults and explanations
- **Navigation Confusion**: Clear visual indicators and help text
- **Performance Delays**: Implement progress indicators and async operations

## Dependencies and Assumptions

### AD-001: External Dependencies
- AT&T dataset availability in demo/data/ATnT/ directory
- Streamlit server environment for local development
- Python environment with required packages installed
- Existing notebook mathematical validation reference

### AD-002: Assumptions
- Users have basic understanding of PCA concepts
- Development machine can handle image processing workloads
- Internet connection available for package installation
- AT&T dataset follows expected folder structure

---

**Created**: 2025-01-21
**Author**: PCA Face Recognition Development Team
**Version**: 1.0.0
**Status**: Ready for Implementation