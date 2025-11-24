# PCA Face Recognition Demo – UI Specification

This document describes the **UI requirements** for a Streamlit app that demonstrates how **PCA is used for face recognition**.  
The app is organized into **tabs**, each corresponding to a conceptual and visual step in the pipeline.

---

## 0. Global UI Layout

### 0.1 Page Settings
- **Title (top of page):**  
  `PCA for Face Recognition (Eigenfaces Demo)`
- **Layout:** Wide (`layout="wide"`)

---

### 0.2 Sidebar Controls (Visible on all tabs)

The sidebar is used to control the **main demo parameters** that stay consistent across tabs.

#### Components

1. **Header**
   - Text: `Step selection`

2. **Person Selection**
   - **Control type:** `selectbox`
   - **Label:** `Choose a person`
   - **Options:** List of unique person IDs/names from the dataset (e.g., `Person 0`, `Person 1`, …).
   - **Behavior:** Selecting a person filters indices for the next dropdown.

3. **Image Selection**
   - **Control type:** `selectbox`
   - **Label:** `Choose an image index for this person`
   - **Options:** List of global indices of images belonging to the selected person.
   - **Display format:** `Global index {i}`
   - **Behavior:** Changing this value updates the **“tracked” face** shown in all tabs.

4. **Separator**
   - Simple horizontal rule (`---`) to visually separate selection from PCA controls.

5. **Number of Components (k)**
   - **Control type:** `slider`
   - **Label:** `Number of PCA components (k)`
   - **Range:** `min = 5`, `max = 100`, `step = 5`
   - **Default value:** `30`
   - **Behavior:** Used in:
     - Eigenfaces tab (how many components to show),
     - Projection & Reconstruction tab,
     - Recognition tab (feature dimensionality).

---

## 1. Tab: “1. Dataset & Intro”

### 1.1 Purpose
Introduce the dataset and show the **specific face** that will be followed across all steps.

### 1.2 Header & Description
- **Tab title:** `1. Dataset & Intro`
- **Short explanation (1–2 sentences):**
  - “PCA helps us represent faces in a lower-dimensional space while keeping most of the important variation. We will follow one selected face through each step of the process to see how PCA transforms it.”

### 1.3 UI Elements

1. **Section: Sample faces grid**
   - **Label:** `Sample faces from the dataset`
   - **Layout:** A grid of thumbnails using columns.
     - Up to **25 faces** shown (e.g., 5 columns × 5 rows).
   - **Each item:**
     - Small image (width ≈ 80px).
     - Caption: person label (e.g., `Person 0`, `Person 1`).

2. **Section: Selected face**
   - **Label:** `Selected face to follow through all steps`
   - **Content:**
     - Image of the **currently selected face** (as chosen in the sidebar).
     - Width ≈ 200px.
     - Caption: `Person: {selected_person} (index {selected_idx})`.

### 1.4 Behavior
- Changing the selected person or image in the sidebar immediately updates:
  - The selected face image here,
  - And the same face on all other tabs.

---

## 2. Tab: “2. Image → Vector”

### 2.1 Purpose
Show how a 2D face image becomes a 1D vector and part of the data matrix.

### 2.2 Header & Description
- **Tab title:** `2. Image → Vector`
- **Short explanation:**
  - “Each face image is a 2D grid of pixels that we flatten into a long vector. Stacking these vectors gives us a data matrix where each row is one face.”

### 2.3 Layout
- Two-column layout:
  - **Left column:** Original image.
  - **Right column:** Visualization of flattened vector.

### 2.4 UI Elements

1. **Left Column – Original Image**
   - **Title:** `Original image`
   - **Content:**
     - Show the selected face as an image.
     - Width ≈ 250px.

2. **Right Column – Pixel Vector Plot**
   - **Title:** `Flattened pixel vector (first 300 values)`
   - **Visualization:**
     - Line plot of the first 300 pixel intensities.
       - x-axis: “Pixel index”
       - y-axis: “Intensity”
       - Title: “First 300 pixel values”

3. **Text Summary**
   - Below columns, show two short lines:
     - `Image size: H x W = (H*W) pixels → vector in (H*W)-dimensional space.`
     - `Total images: N, each row of X is one flattened face.`

### 2.5 Behavior
- Plot updates when:
  - Selected image changes in the sidebar.

---

## 3. Tab: “3. Mean & Centering”

### 3.1 Purpose
Show the **mean face** and how subtracting it centers a specific face.

### 3.2 Header & Description
- **Tab title:** `3. Mean Face & Centering`
- **Short explanation:**
  - “We first compute the average (mean) face and subtract it from each image. This centers the data so PCA focuses on differences between faces rather than overall brightness.”

### 3.3 Layout
- Three columns:
  - Selected face,
  - Mean face,
  - Centered face (difference).

### 3.4 UI Elements

1. **Column 1 – Selected Face**
   - **Title:** `Selected face`
   - **Content:** Selected face image.
   - Width ≈ 200px.

2. **Column 2 – Mean Face**
   - **Title:** `Mean face`
   - **Content:** Mean face image (computed across all images).
   - Width ≈ 200px.

3. **Column 3 – Centered Face**
   - **Title:** `Centered face (face − mean)`
   - **Content:**
     - Centered image obtained by subtracting the mean face from the selected face.
     - Should be **normalized** to a [0,1] range for display (so contrasts are visible).
   - Width ≈ 200px.

### 3.5 Behavior
- All three images change when:
  - Selected face in sidebar updates.
- Mean face stays the same for the dataset (does not depend on `k`).

---

## 4. Tab: “4. Eigenfaces”

### 4.1 Purpose
Visualize eigenfaces (principal components) and how much variance they capture.

### 4.2 Header & Description
- **Tab title:** `4. Eigenfaces (Principal Components)`
- **Short explanation:**
  - “PCA finds directions in pixel space where the faces vary the most; these directions, when reshaped as images, are called eigenfaces. Combining a few eigenfaces lets us reconstruct and recognize faces efficiently.”

### 4.3 UI Elements

1. **Section: Explained Variance Plot**
   - **Label:** `Explained variance (how much variation each component captures)`
   - **Visualization:**
     - Line plot of **cumulative explained variance ratio** for the first 100 components.
       - x-axis: “Number of components”
       - y-axis: “Cumulative explained variance”
       - Title: “Cumulative explained variance of first 100 components”

2. **Section: Eigenfaces Grid**
   - **Label:** `Top N eigenfaces`
     - Actual text: `Top {min(16, k)} eigenfaces`
   - **Visualization:**
     - Show up to **16 eigenfaces** in a grid (4 columns).
     - Each eigenface:
       - Reshape eigenvector to (H, W),
       - Normalize to [0,1] for display,
       - Thumbnail width ≈ 120px,
       - Caption: `PC i` (e.g., `PC 1`, `PC 2`, …).

### 4.4 Behavior
- Number of eigenfaces shown limited by:
  - `N = min(16, k)` where `k` is from the sidebar slider.
- Cumulative variance plot is fixed (based on full PCA), independent of selected person.

---

## 5. Tab: “5. Projection & Reconstruction”

### 5.1 Purpose
Show how the selected face is **projected** into PCA space and **reconstructed** using `k` components.

### 5.2 Header & Description
- **Tab title:** `5. Projection & Reconstruction`
- **Short explanation:**
  - “We project the centered face onto the first k eigenfaces to get a short vector of PCA coefficients. Using only those k coefficients, we can reconstruct an approximation of the original face.”

### 5.3 Layout
- Two-column layout:
  - Left: Original face.
  - Right: Reconstructed face.
- Below: bar chart of PCA coefficients.

### 5.4 UI Elements

1. **Left Column – Original Face**
   - **Title:** `Original face`
   - **Content:** Selected face image.
   - Width ≈ 250px.

2. **Right Column – Reconstructed Face**
   - **Title:** `Reconstructed with k = {k} components`
   - **Content:**
     - Reconstructed face image (after projection onto the first `k` eigenfaces and reconstruction).
     - Image should be clipped/normalized to [0,1] for display.
   - **Text below:** `Reconstruction MSE: {mse}` (formatted to ~6 decimal places).

3. **Section: PCA Coefficients Plot**
   - **Label:** `PCA coefficients (first 20 values)`
   - **Visualization:**
     - Bar chart of the first 20 PCA coefficients for the selected face.
       - x-axis: “Component index”
       - y-axis: “Coefficient value”
       - Title: “First 20 PCA coefficients for this face”

### 5.5 Behavior
- Changing:
  - Selected face → recomputes its coefficients and reconstruction.
  - `k` slider → updates reconstruction and MSE, but not the coefficient vector (which uses the same first `k` components).

---

## 6. Tab: “6. Recognition”

### 6.1 Purpose
Demonstrate **face recognition** by comparing PCA representations of faces to per-person templates.

### 6.2 Header & Description
- **Tab title:** `6. Recognition in PCA Space`
- **Short explanation:**
  - “For recognition, we represent each person by the mean of their PCA vectors and compare a new face to these templates using distance in the PCA space.”

### 6.3 Layout
- Two-column layout:
  - Left: Query (selected) face and its true label.
  - Right: Predicted label and distance chart.
- Info note at bottom.

### 6.4 UI Elements

1. **Left Column – Query Face**
   - **Title:** `Query face (to recognize)`
   - **Content:**
     - Image of the selected face (same as previous tabs).
     - Caption: `True person: {selected_person}`.

2. **Right Column – Predicted Person & Distances**
   - **Title:** `Predicted person`
   - **Content:**
     - Text: `Prediction: {best_person}`
   - **Subsection: Distance Bar Chart**
     - **Label:** `Distances to each person template (lower is better)`
     - **Visualization:**
       - Horizontal bar chart:
         - y-axis: person IDs.
         - x-axis: distance value.
         - Bars sorted by person list order.
         - Title: `Query face vs person templates`.
       - y-axis inverted so the first entry appears at the top.

3. **Info Box**
   - Type: Informational note.
   - Text:
     - “In a full experiment, you would evaluate recognition accuracy on a held-out test set and compare how it changes as you vary k.”

### 6.5 Behavior
- Internally:
  - For each `k`, the app recomputes:
    - Person templates in PCA space (mean PCA vector per person).
    - Distances from selected face to each template.
- UI:
  - Prediction and distances update when:
    - Selected face changes,
    - `k` slider changes.

---

## Notes & Extensions (Optional – Non-blocking)

These are **optional extras** you can add later. They are not required for the base app, but you might mention them in your report.

1. **Extra Tab: Accuracy vs k**
   - Plot recognition accuracy over a test split as `k` varies.

2. **Extra Visualization: Covariance Heatmap**
   - In an advanced tab or under Eigenfaces, show a small patch (e.g., 50×50) of the covariance matrix as a heatmap, to reinforce the “classical PCA” approach.

3. **Theme**
   - Use a consistent grayscale/neutral color scheme to keep focus on the images and plots.

---

This UI spec should be enough to implement the entire Streamlit app and to describe it clearly in your project report or presentation.
