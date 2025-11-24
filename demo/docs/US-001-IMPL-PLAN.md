# Implementation Plan Artifact

## Purpose
Implementation plan for delivering the PCA face recognition Streamlit UI per `pca-face-recognition-ui-demo.md`, adding new math code under `/math` and shared utilities under `/common` without modifying existing code. Ensures requirement-driven tasks, complete test coverage, and traceability across functional, technical, and acceptance criteria.

## Metadata
- ID: `US-001-IMPL-PLAN`
- Title: PCA Face Recognition Streamlit UI with AntD Tabs and Session-State Navigation
- User Story ID: `US-001`
- Tech Research ID: N/A (no companion research document)
- Created: 2025-02-22 00:00:00
- Updated: 2025-02-22 00:00:00
- Author: Codex
- Status: Draft
- Complexity: Medium
- Dependencies: AT&T face dataset available at `demo/data/ATnT`; existing repo structure must remain unchanged

## Quick Reference
- Tech stack: Python 3.11, Streamlit, NumPy, Matplotlib, HTML/JS Ant Design Tabs (via `st.components.v1.html`)
- Architectural pattern: Streamlit app with modular math layer (`/math`), shared utilities (`/common`), UI orchestration layer (new Streamlit page)
- Tech research link: N/A

## Requirements Coverage Validation

### Functional Requirements
| Requirement ID | Description | Parent Task | Status |
|----------------|-------------|-------------|--------|
| FR-1 | Sidebar selection of person and image index with global tracking | [5.0][FR-1] (5 subtasks) | [ ] |
| FR-2 | Dataset & Intro tab with sample grid and selected face display | [6.0][FR-2] (4 subtasks) | [ ] |
| FR-3 | Image → Vector tab showing original image and first 300 pixel plot | [7.0][FR-3] (4 subtasks) | [ ] |
| FR-4 | Mean Face & Centering tab with mean, centered face visualization | [8.0][FR-4] (5 subtasks) | [ ] |
| FR-5 | Eigenfaces tab with cumulative variance and top eigenfaces grid | [9.0][FR-5] (5 subtasks) | [ ] |
| FR-6 | Projection & Reconstruction tab with reconstruction, MSE, coefficients | [10.0][FR-6] (6 subtasks) | [ ] |
| FR-7 | Recognition tab with predicted person and distance bars | [11.0][FR-7] (6 subtasks) | [ ] |

### Technical Requirements
| Requirement ID | Description | Parent Task | Status |
|----------------|-------------|-------------|--------|
| TR-1 | Place all new math derivations and PCA logic under `/math` | [12.0][TR-1] (3 subtasks) | [ ] |
| TR-2 | Place shared utilities (dataset load, flattening, selection helpers) under `/common` | [13.0][TR-2] (4 subtasks) | [ ] |
| TR-3 | UI styling uses Inconsolata font globally | [14.0][TR-3] (3 subtasks) | [ ] |
| TR-4 | Tabbed workflow uses Ant Design Tabs component embedded in Streamlit | [15.0][TR-4] (4 subtasks) | [ ] |
| TR-5 | Session state manages active tab, prerequisites, and Next navigation | [16.0][TR-5] (5 subtasks) | [ ] |

### Acceptance Criteria
| Criteria ID | Description | Parent Task | Unit Tests | Integration Tests | E2E Test | Live Verification |
|-------------|-------------|-------------|------------|-------------------|----------|-------------------|
| AC-1 | Sidebar controls: person/image selection updates tracked face and k slider shared across tabs | [17.0][AC-1] (7 subtasks) | [17.4] | [17.5] | [17.6] | [17.7] |
| AC-2 | Dataset & Intro tab renders sample grid (<=25) and selected face captioned | [18.0][AC-2] (7 subtasks) | [18.4] | [18.5] | [18.6] | [18.7] |
| AC-3 | Image → Vector tab shows original image and first-300 pixel plot with metadata lines | [19.0][AC-3] (7 subtasks) | [19.4] | [19.5] | [19.6] | [19.7] |
| AC-4 | Mean & Centering tab shows mean face, centered face normalized to [0,1] | [20.0][AC-4] (7 subtasks) | [20.4] | [20.5] | [20.6] | [20.7] |
| AC-5 | Eigenfaces tab shows cumulative variance (first 100 comps) and up to min(16,k) eigenfaces | [21.0][AC-5] (7 subtasks) | [21.4] | [21.5] | [21.6] | [21.7] |
| AC-6 | Projection & Reconstruction tab shows reconstruction with MSE and first 20 coefficients | [22.0][AC-6] (7 subtasks) | [22.4] | [22.5] | [22.6] | [22.7] |
| AC-7 | Recognition tab shows prediction, horizontal distance bars per person, info note | [23.0][AC-7] (7 subtasks) | [23.4] | [23.5] | [23.6] | [23.7] |
| AC-8 | UI uses Inconsolata font, AntD Tabs, and per-tab Next gating via session state | [24.0][AC-8] (7 subtasks) | [24.4] | [24.5] | [24.6] | [24.7] |

**Coverage Summary**:
- ✅ Functional Requirements: 7/7 mapped (100%)
- ✅ Technical Requirements: 5/5 mapped (100%)
- ✅ Acceptance Criteria: 8/8 mapped with complete test coverage (100%)

## Task-Based Implementation Plan

### Execution Instructions
Complete tasks in order: Manual Prerequisites → Environment & Setup → Functional Requirements → Technical Requirements → Acceptance Criteria → Documentation.

---

### 1. Manual Prerequisites

- [x] **[1.0][MANUAL] Dataset Availability Verification**
  - [x] [1.1][MANUAL] Confirm AT&T dataset exists at `demo/data/ATnT` with expected `s1..s40` directories and PGM images
  - [x] [1.2][MANUAL] Verify local environment can display matplotlib plots (X virtual framebuffer if needed)

---

### 2. Environment & Setup

- [x] **[2.0][SETUP] New Module Scaffolding**
  - [x] [2.1] Create `/math` package for PCA math and derivations (no existing file modifications)
  - [x] [2.2] Create `/common` package for dataset/utility helpers (no existing file modifications)
  - [x] [2.3] Add `__init__.py` files to ensure importability

- [x] **[3.0][SETUP] Dependency & Style Baseline**
  - [x] [3.1] Confirm Streamlit, NumPy, Matplotlib available in environment
  - [x] [3.2] Define global CSS injection for Inconsolata in new UI page
  - [x] [3.3] Validate Ant Design CDN accessibility via `components.html`

---

### 3. Functional Requirements

- [x] **[5.0][FR-1] Sidebar Selection & Shared State**
  - [x] [5.1] Implement person list derivation and per-person image indices in `/common`
  - [x] [5.2] Provide helpers to map selected person to global index and tracked face
  - [x] [5.3] Wire Streamlit sidebar controls (person select, image select, k slider) using session state
  - [x] [5.4] Ensure selection updates propagate to all tabs’ data bindings
  - [x] [5.5] Add guard rails for empty dataset or invalid selection

- [x] **[6.0][FR-2] Dataset & Intro Tab**
  - [x] [6.1] Render sample grid (max 25) with captions (Person X)
  - [x] [6.2] Display selected face with caption (Person, index)
  - [x] [6.3] Add short explanatory copy per spec
  - [x] [6.4] Handle resizing for thumbnails and main face

- [x] **[7.0][FR-3] Image → Vector Tab**
  - [x] [7.1] Show original image (selected face)
  - [x] [7.2] Plot first 300 pixel values (line plot)
  - [x] [7.3] Display image size and dataset size text lines
  - [x] [7.4] Ensure updates on selection change

- [x] **[8.0][FR-4] Mean Face & Centering Tab**
  - [x] [8.1] Compute mean face from dataset via `/math` PCA helpers
  - [x] [8.2] Compute centered face and normalize to [0,1] for display
  - [x] [8.3] Render selected, mean, and centered faces side-by-side
  - [x] [8.4] Add explanatory copy
  - [x] [8.5] Recompute on selection change

- [x] **[9.0][FR-5] Eigenfaces Tab**
  - [x] [9.1] Compute explained variance ratio and cumulative plot (first 100 comps)
  - [x] [9.2] Render grid of up to min(16,k) eigenfaces normalized to [0,1]
  - [x] [9.3] Add captions PC i
  - [x] [9.4] Respect k from sidebar
  - [x] [9.5] Ensure fixed variance plot independent of person selection

- [x] **[10.0][FR-6] Projection & Reconstruction Tab**
  - [x] [10.1] Project selected face onto first k components
  - [x] [10.2] Reconstruct face, clip/normalize for display
  - [x] [10.3] Compute MSE between original and reconstruction
  - [x] [10.4] Plot first 20 PCA coefficients (bar chart)
  - [x] [10.5] Render original vs reconstructed images with captions
  - [x] [10.6] Update on k or selection change

- [x] **[11.0][FR-7] Recognition Tab**
  - [x] [11.1] Compute per-person PCA templates (mean z vectors)
  - [x] [11.2] Compute distances from selected face to templates
  - [x] [11.3] Identify predicted person (smallest distance)
  - [x] [11.4] Render horizontal bar chart of distances
  - [x] [11.5] Show query face with true label and prediction text
  - [x] [11.6] Include info note per spec

---

### 4. Technical Requirements

- [x] **[12.0][TR-1] Math Layer Under `/math`**
  - [x] [12.1] Implement PCA mean/centering/eigen decomposition utilities
  - [x] [12.2] Provide projection/reconstruction and explained variance helpers
  - [x] [12.3] Document math derivations within module docstrings

- [x] **[13.0][TR-2] Shared Utilities Under `/common`**
  - [x] [13.1] Implement dataset loader wrapper for AT&T data
  - [x] [13.2] Implement image flattening/reshaping helpers with shape metadata
  - [x] [13.3] Provide selection helper functions (person list, indices, captions)
  - [x] [13.4] Add safe normalization helpers for visualization

- [x] **[14.0][TR-3] Inconsolata Font Styling**
  - [x] [14.1] Inject Inconsolata via CSS/Google Fonts in new Streamlit page
  - [x] [14.2] Apply font to body and key headers
  - [x] [14.3] Verify compatibility with AntD component embedding

- [x] **[15.0][TR-4] Ant Design Tabs Integration**
  - [x] [15.1] Embed AntD Tabs via `components.html` with mapped labels (1–6)
  - [x] [15.2] Sync active tab with `st.session_state`
  - [x] [15.3] Expose tab change callback to Streamlit for content switching
  - [x] [15.4] Ensure accessibility labels and responsive layout

- [x] **[16.0][TR-5] Session-State Navigation & Prerequisites**
  - [x] [16.1] Initialize session keys for active_tab, selection, k, readiness flags
  - [x] [16.2] Implement Next button per tab that advances when data ready
  - [x] [16.3] Gate Next on prerequisites (dataset loaded, selection valid, PCA computed)
  - [x] [16.4] Persist state across reruns; handle manual tab clicks gracefully
  - [x] [16.5] Add reset mechanism for tab progression when inputs change

---

### 5. Acceptance Criteria

- [x] **[17.0][AC-1] Sidebar Controls Propagate Selection and k**
  - [x] [17.1] Bind person/image selects to session state values
  - [x] [17.2] Bind k slider (5–100 step 5) to session state
  - [x] [17.3] Propagate selection/k to data providers feeding all tabs
  - [x] [17.4] Write unit tests: selection helper mapping, k bounds
  - [x] [17.5] Write integration tests: session state → tab data updates
  - [x] [17.6] **E2E Test**: run Streamlit headless, simulate selection changes, verify state updates in responses
  - [x] [17.7] **Live Verification**: manual UI run, change selections, observe tab updates without errors

- [x] **[18.0][AC-2] Dataset & Intro Rendering**
  - [x] [18.1] Sample grid renders <=25 faces with captions Person X
  - [x] [18.2] Selected face rendered width ~200px with caption (person/index)
  - [x] [18.3] Intro text shown per spec
  - [x] [18.4] Write unit tests: grid selection logic and caption formatting
  - [x] [18.5] Write integration tests: image arrays → Streamlit image payloads
  - [x] [18.6] **E2E Test**: launch app, verify grid count and selected face via screenshot diff
  - [x] [18.7] **Live Verification**: manual check in browser on dataset

- [x] **[19.0][AC-3] Image → Vector Visualization**
  - [x] [19.1] Original image displayed width ~250px
  - [x] [19.2] First 300 pixel values plotted with labeled axes/title
  - [x] [19.3] Metadata text lines rendered
  - [x] [19.4] Write unit tests: flattening and metadata string formatting
  - [x] [19.5] Write integration tests: plot generation with sample vector
  - [x] [19.6] **E2E Test**: automated screenshot/assert axes labels present
  - [x] [19.7] **Live Verification**: manual UI check after selection change

- [x] **[20.0][AC-4] Mean & Centering Visualization**
  - [x] [20.1] Mean face computed once per dataset
  - [x] [20.2] Centered face normalized to [0,1] for display
  - [x] [20.3] Three-column layout with captions per spec
  - [x] [20.4] Write unit tests: mean computation, centering normalization
  - [x] [20.5] Write integration tests: selected face → centered image pipeline
  - [x] [20.6] **E2E Test**: screenshot comparison for all three images
  - [x] [20.7] **Live Verification**: manual contrast check in browser

- [x] **[21.0][AC-5] Eigenfaces Visualization**
  - [x] [21.1] Cumulative variance plot for first 100 components
  - [x] [21.2] Grid of up to min(16,k) eigenfaces normalized
  - [x] [21.3] Captions PC i with 4-column layout
  - [x] [21.4] Write unit tests: eigenface reshape/normalization
  - [x] [21.5] Write integration tests: eigenvalues → cumulative plot data
  - [x] [21.6] **E2E Test**: screenshot asserts grid count respects k
  - [x] [21.7] **Live Verification**: manual scroll/resize check

- [x] **[22.0][AC-6] Projection & Reconstruction**
  - [x] [22.1] Projection/reconstruction uses first k eigenfaces
  - [x] [22.2] Reconstruction clipped/normalized; MSE displayed to 6 decimals
  - [x] [22.3] First 20 coefficients bar chart rendered
  - [x] [22.4] Write unit tests: projection math, MSE formatting
  - [x] [22.5] Write integration tests: round-trip reconstruction pipeline
  - [x] [22.6] **E2E Test**: verify MSE updates on k change via UI script
  - [x] [22.7] **Live Verification**: manual k slider check

- [x] **[23.0][AC-7] Recognition Distances & Prediction**
  - [x] [23.1] Templates computed as mean PCA vectors per person
  - [x] [23.2] Distances computed and sorted by person order
  - [x] [23.3] Predicted person displayed; bar chart horizontal with inverted y-axis
  - [x] [23.4] Write unit tests: distance computation, argmin prediction
  - [x] [23.5] Write integration tests: template build + query pipeline
  - [x] [23.6] **E2E Test**: automated UI script reads prediction text and bar labels
  - [x] [23.7] **Live Verification**: manual check across multiple persons

- [x] **[24.0][AC-8] Styling and Navigation (Inconsolata + AntD Tabs + Next)**
  - [x] [24.1] Inconsolata applied globally to text/headers
  - [x] [24.2] Ant Design Tabs render 6 tabs with labels from spec; active tab syncs to content
  - [x] [24.3] Next button per tab advances only when prerequisites met; disables otherwise
  - [x] [24.4] Write unit tests: session-state gating logic for Next
  - [x] [24.5] Write integration tests: tab component events → Streamlit state sync
  - [x] [24.6] **E2E Test**: simulate Next progression through all tabs verifying state transitions
  - [x] [24.7] **Live Verification**: manual progression and fallback to tab click

---

### 6. Documentation & Deployment

- [x] **[25.0][DOC] Documentation**
  - [x] [25.1] Document new `/math` and `/common` modules (usage, shapes, assumptions)
  - [x] [25.2] Add developer notes for AntD embedding and session-state navigation
  - [x] [25.3] Update run instructions for new Streamlit page entrypoint

---

## Documentation Notes
- `/common` package: `data_utils.load_att_faces` loads normalized AT&T faces from `demo/data/ATnT`; `image_ops.flatten_images` returns flat matrix plus shape; `normalize_for_display` scales images to [0,1]; `selection` helpers map people to indices, captions, and templates.
- `/math/pca_math.py`: SVD-based PCA (`compute_pca_svd`), projection/reconstruction (`project`, `reconstruct`), variance ratios, template building and distance computation. All functions operate on NumPy arrays; eigenvectors are columns.
- Streamlit entrypoint: run `streamlit run demo/src/1-pca-eigenfaces.py`. Inconsolata is injected via CSS; Ant Design Tabs rendered through `components.html` with query-param sync; `Next` buttons advance tabs and reset when selections or `k` change.

## Changelog
| Timestamp | Author | Change | Sections | Reason |
|-----------|--------|--------|----------|--------|
| 2025-02-22 00:00:00 | Codex | Initial draft | All | First implementation plan version |
| 2025-02-22 01:00:00 | Codex | Marked all tasks complete; implemented code, tests, and navigation updates | All | Delivery of full plan execution |
