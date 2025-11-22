import numpy as np

from common.image_ops import flatten_images, normalize_for_display
from common.selection import default_selection, indices_for_person, template_mean_by_person, unique_people
from math import isclose

# Avoid name clash with built-in math by importing via relative path
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_pca_math_path = ROOT / "math" / "pca_math.py"
_spec = importlib.util.spec_from_file_location("pca_math_module_tests", _pca_math_path)
_pca_math = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_pca_math)  # type: ignore[arg-type]

compute_pca_svd = _pca_math.compute_pca_svd
project = _pca_math.project
reconstruct = _pca_math.reconstruct
explained_variance_ratio = _pca_math.explained_variance_ratio
build_templates = _pca_math.build_templates
distances_to_templates = _pca_math.distances_to_templates


def test_flatten_and_normalize():
    imgs = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    flat, shape = flatten_images(imgs)
    assert shape == (3, 2)
    assert flat.shape == (2, 6)
    norm = normalize_for_display(imgs[0])
    assert norm.min() == 0.0
    assert norm.max() == 1.0


def test_selection_helpers():
    labels = np.array([0, 0, 1, 1, 2])
    people = unique_people(labels)
    assert list(people) == [0, 1, 2]
    p, idx = default_selection(labels)
    assert p == 0 and idx == 0
    np.testing.assert_array_equal(indices_for_person(labels, 1), np.array([2, 3]))
    z = np.random.randn(len(labels), 4)
    templates = template_mean_by_person(z, labels)
    assert set(templates.keys()) == {0, 1, 2}


def test_pca_projection_round_trip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 6))
    mu, eigvals, eigvecs = compute_pca_svd(X)
    k = 3
    z = project(X[0], mu, eigvecs, k)
    x_hat = reconstruct(z, mu, eigvecs)
    assert x_hat.shape == (6,)
    ratio = explained_variance_ratio(eigvals)
    assert isclose(ratio.sum(), 1.0, rel_tol=1e-5)


def test_template_and_recognition_pipeline():
    rng = np.random.default_rng(1)
    # Two persons, separable clusters
    base0 = rng.normal(loc=0.0, scale=0.1, size=(10, 5))
    base1 = rng.normal(loc=3.0, scale=0.1, size=(10, 5))
    X = np.vstack([base0, base1])
    labels = np.array([0] * 10 + [1] * 10)
    mu, eigvals, eigvecs = compute_pca_svd(X)
    k = 3
    z_all = (X - mu) @ eigvecs[:, :k]
    templates = build_templates(z_all, labels)
    query = project(base1[0], mu, eigvecs, k)
    dists = distances_to_templates(query, templates)
    best = min(dists, key=dists.get)
    assert best == 1


if __name__ == "__main__":
    test_flatten_and_normalize()
    test_selection_helpers()
    test_pca_projection_round_trip()
    test_template_and_recognition_pipeline()
    print("All tests passed.")
