# Implementation Plan Artifact

## Purpose
Implementation plan for a new Streamlit page “Face Recognition” (US-002) styled like the existing PCA app, using the same tabbed workflow with Back/Next navigation and inputs for train-test split and number of PCA components.

## Metadata
- ID: `US-002-IMPL-PLAN`
- Title: Face Recognition Page with Tabbed Workflow
- User Story ID: `US-002`
- Tech Research ID: N/A
- Created: 2025-02-22 00:00:00
- Updated: 2025-02-22 00:00:00
- Author: Codex
- Status: Draft
- Complexity: Medium
- Dependencies: Existing PCA app styling/components; dataset loader; PCA utilities; recognition logic/templates

## Quick Reference
- Tech stack: Python 3.11, Streamlit, NumPy, Matplotlib
- Architectural pattern: Single-page tabbed UI reusing PCA utilities and recognition pipeline; stateful navigation with Back/Next buttons
- Tech research link: N/A
- Input reuse: Train-test split slider and PCA k slider already exist in the sidebar; the Face Recognition page must consume these without introducing duplicate controls.

## Requirements Coverage Validation

### Functional Requirements
| Requirement ID | Description | Parent Task | Status |
|----------------|-------------|-------------|--------|
| FR-1 | Reuse global styling and tabbed UI (AntD tabs + Back/Next) for the new Face Recognition page | [5.0][FR-1] (5 subtasks) | [ ] |
| FR-2 | Inputs: train-test split slider and PCA components slider wired to recognition pipeline | [6.0][FR-2] (4 subtasks) | [ ] |
| FR-3 | Display training/testing metrics (e.g., accuracy/MSE) after split and recognition | [7.0][FR-3] (5 subtasks) | [ ] |
| FR-4 | Show recognition results per sample (predicted vs true, distances) in a tabbed workflow | [8.0][FR-4] (5 subtasks) | [ ] |
| FR-5 | Navigation controls (Back/Next) per tab with scroll-to-top behavior | [9.0][FR-5] (4 subtasks) | [ ] |

### Technical Requirements
| Requirement ID | Description | Parent Task | Status |
|----------------|-------------|-------------|--------|
| TR-1 | Maintain consistent styling with existing app (fonts, spacing, tab visuals) | [10.0][TR-1] (3 subtasks) | [ ] |
| TR-2 | Use existing PCA/recognition utilities without code duplication | [11.0][TR-2] (3 subtasks) | [ ] |
| TR-3 | Ensure navigation state and scroll reset on tab changes | [12.0][TR-3] (3 subtasks) | [ ] |

### Acceptance Criteria
| Criteria ID | Description | Parent Task | Unit Tests | Integration Tests | E2E Test | Live Verification |
|-------------|-------------|-------------|------------|-------------------|----------|-------------------|
| AC-1 | Train-test split slider (1–9 images per person or percentage) and PCA k slider affect recognition outputs | [13.0][AC-1] (7 subtasks) | [13.4] | [13.5] | [13.6] | [13.7] |
| AC-2 | Tabbed flow with Back/Next mirrors existing app and scrolls to top on tab change | [14.0][AC-2] (7 subtasks) | [14.4] | [14.5] | [14.6] | [14.7] |
| AC-3 | Recognition results show per-sample prediction vs true, distances, and aggregate metrics | [15.0][AC-3] (7 subtasks) | [15.4] | [15.5] | [15.6] | [15.7] |

**Coverage Summary**:
- ✅ Functional Requirements: 5/5 mapped (100%)
- ✅ Technical Requirements: 3/3 mapped (100%)
- ✅ Acceptance Criteria: 3/3 mapped with complete test coverage (100%)

## Task-Based Implementation Plan

### Execution Instructions
Complete tasks in order: Manual Prerequisites → Environment & Setup → Functional Requirements → Technical Requirements → Acceptance Criteria → Documentation.

---

### 1. Manual Prerequisites

- [x] **[1.0][MANUAL] Dataset Availability**
  - [x] [1.1][MANUAL] Confirm face dataset accessible and compatible with existing loaders

---

### 2. Environment & Setup

- [x] **[2.0][SETUP] Page Scaffolding**
  - [x] [2.1] Create new page entry (Face Recognition) wired into navigation
  - [x] [2.2] Import shared styling/components from existing app

---

### 3. Functional Requirements

- [x] **[5.0][FR-1] Tabbed UI & Navigation**
  - [x] [5.1] Implement AntD-like tabs with Back/Next controls
  - [x] [5.2] Sync tab state with session and URL (if applicable)
  - [x] [5.3] Apply same layout spacing and fonts as existing PCA app
  - [x] [5.4] Add scroll-to-top on tab change
  - [x] [5.5] Validate Back/Next flow matches tab order

- [x] **[6.0][FR-2] Inputs: Train-Test Split & PCA k**
  - [x] [6.1] Add train-test split slider (per-person images or percentage per story)
  - [x] [6.2] Add PCA components slider (respect dataset cap)
  - [x] [6.3] Bind inputs to recognition pipeline recalculations
  - [x] [6.4] Show current selections in UI

- [x] **[7.0][FR-3] Metrics Display**
  - [x] [7.1] Compute train/test sets based on split
  - [x] [7.2] Run recognition and compute metrics (e.g., accuracy, confusion elements)
  - [x] [7.3] Render summary metrics in tab context
  - [x] [7.4] Update metrics on input changes
  - [x] [7.5] Handle empty/edge cases gracefully

- [x] **[8.0][FR-4] Per-Sample Recognition Results**
  - [x] [8.1] Show predicted vs true label for samples
  - [x] [8.2] Display distance scores/templates
  - [x] [8.3] Allow browsing test samples within tab workflow
  - [x] [8.4] Update views when inputs change
  - [x] [8.5] Add contextual descriptions

- [x] **[9.0][FR-5] Navigation Controls**
  - [x] [9.1] Back/Next buttons on each tab
  - [x] [9.2] Disable/enable based on position/readiness
  - [x] [9.3] Ensure scroll-to-top after navigation
  - [x] [9.4] Persist state across reruns

---

### 4. Technical Requirements

- [x] **[10.0][TR-1] Styling Consistency**
  - [x] [10.1] Reuse Inconsolata font, spacing, header sizing
  - [x] [10.2] Match button/tab aesthetics with existing page
  - [x] [10.3] Validate responsive layout matches existing app behavior

- [x] **[11.0][TR-2] Reuse PCA/Recognition Utilities**
  - [x] [11.1] Use existing dataset loader and flattening utilities
  - [x] [11.2] Use existing PCA projection/reconstruction functions
  - [x] [11.3] Use existing recognition/template builders to avoid duplication

- [x] **[12.0][TR-3] Navigation State & Scroll**
  - [x] [12.1] Centralize tab state in session
  - [x] [12.2] Implement scroll reset on tab change
  - [x] [12.3] Ensure Back/Next clicks don’t leave the page mid-scroll

---

### 5. Acceptance Criteria

- [ ] **[13.0][AC-1] Inputs Drive Recognition**
  - [ ] [13.1] Bind train-test split slider to dataset split
  - [ ] [13.2] Bind k slider to PCA dimensionality
  - [ ] [13.3] Refresh recognition outputs on changes
  - [ ] [13.4] Unit tests: split/k bounds, state propagation
  - [ ] [13.5] Integration tests: split/k → recognition pipeline updates
  - [ ] [13.6] **E2E Test**: change split/k via headless run; assert updated metrics in response
  - [ ] [13.7] **Live Verification**: manual run adjusting split/k and observing results

- [ ] **[14.0][AC-2] Tabbed Navigation & Scroll**
  - [ ] [14.1] Back/Next present on all tabs and navigate correctly
  - [ ] [14.2] Tabs clickable and stateful
  - [ ] [14.3] Scroll resets to top on tab change
  - [ ] [14.4] Unit tests: tab state functions
  - [ ] [14.5] Integration tests: tab clicks/back/next update state
  - [ ] [14.6] **E2E Test**: simulate tab changes; verify scroll position and content
  - [ ] [14.7] **Live Verification**: manual navigation across tabs

- [ ] **[15.0][AC-3] Recognition Outputs**
  - [ ] [15.1] Show per-sample predicted vs true label and distances
  - [ ] [15.2] Show aggregate metrics for test split
  - [ ] [15.3] Update outputs on input changes
  - [ ] [15.4] Unit tests: distance/prediction calculations
  - [ ] [15.5] Integration tests: end-to-end recognition with mock data
  - [ ] [15.6] **E2E Test**: headless run verifying rendered metrics text
  - [ ] [15.7] **Live Verification**: manual check across multiple splits/k values

---

### 6. Documentation & Deployment

- [ ] **[16.0][DOC] Page Documentation**
  - [ ] [16.1] Add README section describing Face Recognition page, inputs, and outputs
  - [ ] [16.2] Document navigation behavior and styling reuse
  - [ ] [16.3] Note dataset assumptions and component caps (max k = n_samples−1)

---

## Changelog
| Timestamp | Author | Change | Sections | Reason |
|-----------|--------|--------|----------|--------|
| 2025-02-22 00:00:00 | Codex | Initial draft | All | First implementation plan version |
