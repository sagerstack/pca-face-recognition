# PCA Component Limit Derivation (AT&T Faces Demo)

Let **X ∈ ℝⁿˣᵈ** be the centered training matrix (each row is a flattened face).

- Image dimensionality: **D = 112 × 92 = 10 304**.
- Empirical covariance: **C = (1 / (N_train−1)) · XᵀX ∈ ℝᵈˣᵈ**.
- Rank bound: **rank(C) ≤ min(D, N_train−1)**. After centering, one degree of freedom is lost, so only **N_train−1** singular values can be nonzero.

Therefore, the maximum number of usable principal components (eigenfaces) is:

**k_max = min(D, N_train−1)**.

Any components beyond this add zero variance.
