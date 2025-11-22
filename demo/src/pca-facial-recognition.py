import base64
import importlib.util
import logging
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit import components

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data_utils import load_att_faces
from common.image_ops import flatten_images, normalize_for_display
from common.selection import (
    default_selection,
    indices_for_person,
    person_display_name,
    selection_caption,
    unique_people,
)

# Load math/pca_math without clashing with built-in math module
_pca_math_path = ROOT / "math" / "pca_math.py"
_spec = importlib.util.spec_from_file_location("pca_math_module", _pca_math_path)
_pca_math = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_pca_math)  # type: ignore[arg-type]

build_templates = _pca_math.build_templates
compute_pca_svd = _pca_math.compute_pca_svd
distances_to_templates = _pca_math.distances_to_templates
explained_variance_ratio = _pca_math.explained_variance_ratio
project = _pca_math.project
reconstruct = _pca_math.reconstruct

# Paths
DATA_DIR = ROOT / "data" / "ATnT"

# Tab config
TAB_KEYS = ["1", "2", "3", "4", "5", "6"]
TAB_LABELS = {
    "1": "1. Dataset & Intro",
    "2": "2. Image → Vector",
    "3": "3. Mean & Centering",
    "4": "4. Eigenfaces",
    "5": "5. Projection & Reconstruction",
    "6": "6. Recognition",
}


# -----------------------------
# Logging
# -----------------------------
def init_logger() -> logging.Logger:
    """Initialize logger writing to logs/streamlit-app-YYYYMMDD-HHMMSS.log and capturing uncaught errors."""
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"streamlit-app-{timestamp}.log"

    logger = logging.getLogger("pca_face_demo")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s %(pathname)s:%(lineno)d %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reset handlers each run to ensure clean logging
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(sh)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    for h in list(warnings_logger.handlers):
        warnings_logger.removeHandler(h)
    warnings_logger.addHandler(fh)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    # Root logger also writes to file to catch Streamlit internals
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    fh_root = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh_root.setFormatter(formatter)
    root_logger.addHandler(fh_root)

    logger.info(f"Logging initialized at {log_path}")
    st.session_state["log_file"] = str(log_path)
    return logger


# -----------------------------
# Cached data + PCA computation
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    return load_att_faces(str(DATA_DIR))


@st.cache_resource(show_spinner=True)
def compute_pca_cache(X_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return compute_pca_svd(X_flat)


# -----------------------------
# Session state helpers
# -----------------------------
def init_session_state(labels: np.ndarray):
    qp = st.query_params.get("tab") if hasattr(st, "query_params") else None
    tab_param = qp[0] if isinstance(qp, list) else qp
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tab_param if tab_param in TAB_KEYS else "1"
    elif tab_param in TAB_KEYS and tab_param != st.session_state.active_tab:
        st.session_state.active_tab = tab_param
    if "selected_person" not in st.session_state or "selected_idx" not in st.session_state:
        p, idx = default_selection(labels)
        st.session_state.selected_person = p
        st.session_state.selected_idx = idx
    if "k" not in st.session_state:
        st.session_state.k = 30
    if "prev_person" not in st.session_state:
        st.session_state.prev_person = st.session_state.selected_person
    if "prev_idx" not in st.session_state:
        st.session_state.prev_idx = st.session_state.selected_idx
    if "prev_k" not in st.session_state:
        st.session_state.prev_k = st.session_state.k


def set_active_tab(tab_key: str):
    st.session_state.active_tab = tab_key
    if hasattr(st, "query_params"):
        st.query_params["tab"] = tab_key


# -----------------------------
# UI helpers
# -----------------------------
def inject_inconsolata():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inconsolata', monospace !important;
        }
        .block-container {padding-top: 2rem !important;}
        main {padding-top: 1rem !important;}
        h1, h2, h3, h4 {margin-top: 0 !important; margin-bottom: 0.4rem !important;}
        section[data-testid="stHorizontalBlock"] {margin-top: 0 !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_antd_tabs(active_key: str):
    items = [{"key": k, "label": v} for k, v in TAB_LABELS.items()]
    items_js = str(items).replace("'", '"')
    html = f"""
    <div id="antd-tabs-root"></div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/antd/4.24.15/antd.min.css" />
    <script crossorigin src="https://unpkg.com/react@17/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/antd/4.24.15/antd.min.js"></script>
    <script>
      const e = React.createElement;
      const items = {items_js};
      const mount = document.getElementById("antd-tabs-root");
      const onChange = (key) => {{
        const params = new URLSearchParams(window.location.search);
        params.set("tab", key);
        window.location.search = "?" + params.toString();
      }};
      ReactDOM.render(
        e(antd.Tabs, {{
          activeKey: "{active_key}",
          onChange,
          items
        }}),
        mount
      );
    </script>
    """
    components.v1.html(html, height=80, scrolling=False)
    return None


def next_button(ready: bool = True, disabled_reason: str = ""):
    current = st.session_state.active_tab
    idx = TAB_KEYS.index(current)
    back_col, spacer, next_col = st.columns([1, 12, 1])
    with back_col:
        if idx > 0 and st.button("Back", key=f"back-{current}"):
            set_active_tab(TAB_KEYS[idx - 1])
            st.rerun()
    with next_col:
        if idx < len(TAB_KEYS) - 1:
            disabled = not ready
            label = "Next" if ready else f"Next (locked: {disabled_reason})"
            if st.button(label, key=f"next-{current}", disabled=disabled):
                set_active_tab(TAB_KEYS[idx + 1])
                st.rerun()


# -----------------------------
# Plotting helpers
# -----------------------------
def line_plot_first_300(x_flat: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot(x_flat[:300])
    ax.set_xlabel("Pixel index")
    ax.set_ylabel("Intensity")
    ax.set_title("First 300 pixel values")
    st.pyplot(fig)


def bar_plot_coeffs(z: np.ndarray):
    fig, ax = plt.subplots()
    m = min(20, len(z))
    ax.bar(np.arange(m), z[:m])
    ax.set_xlabel("Component index")
    ax.set_ylabel("Coefficient value")
    ax.set_title("First 20 PCA coefficients for this face")
    st.pyplot(fig)


def variance_plot(explained: np.ndarray, threshold: float = 0.95):
    cum = np.cumsum(explained) * 100.0
    ks = np.arange(1, len(cum) + 1)
    optimal_idx = int(np.argmax(cum >= threshold * 100)) if np.any(cum >= threshold * 100) else len(cum) - 1
    selected_k = min(st.session_state.get("k", len(cum)), len(cum))
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(ks, cum, label="Cumulative variance")
    ax.axhline(threshold * 100, color="gray", linestyle="--", linewidth=1, label=f"{int(threshold*100)}% target")
    ax.scatter(ks[optimal_idx], cum[optimal_idx], color="red", zorder=5, label=f"Optimal k={ks[optimal_idx]}")
    ax.scatter(selected_k, cum[selected_k - 1], color="blue", zorder=5, label=f"Selected k={selected_k}")
    ax.annotate(
        f"Optimal k={ks[optimal_idx]}\n{cum[optimal_idx]:.1f}%",
        (ks[optimal_idx], cum[optimal_idx]),
        textcoords="offset points",
        xytext=(10, -20),
        ha="left",
        va="top",
        color="red",
    )
    ax.annotate(
        f"Selected k={selected_k}\n{cum[selected_k-1]:.1f}%",
        (selected_k, cum[selected_k - 1]),
        textcoords="offset points",
        xytext=(10, -20),
        ha="left",
        va="top",
        color="blue",
    )
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Cumulative explained variance")
    ax.set_ylim(0, 105)
    ax.legend()
    st.pyplot(fig)


def distance_bar_chart(distances: Dict[int, float]):
    person_ids = list(distances.keys())
    vals = np.array([distances[p] for p in person_ids])
    fig, ax = plt.subplots()
    ax.barh([person_display_name(pid) for pid in person_ids], vals)
    ax.invert_yaxis()
    ax.set_xlabel("Distance")
    ax.set_title("Query face vs person templates")
    st.pyplot(fig)


# -----------------------------
# Tab content
# -----------------------------
def tab_dataset_intro(images, labels, x_image):
    st.subheader("1. Dataset & Intro")
    st.write(
        "PCA helps us represent faces in a lower-dimensional space while keeping most of the important variation. "
        "We will follow one selected face through each step of the process to see how PCA transforms it."
    )
    train_count = st.session_state.get("train_count", 6)

    def bordered_image_html(image: np.ndarray, caption: str, color: str, width: int = 90) -> str:
        img8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(img8)
        buf = BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"""
        <div style="display:inline-block; border:2px solid {color}; padding:4px; margin:4px; text-align:center;">
            <img src="data:image/png;base64,{b64}" width="{width}" />
            <div style="font-size:12px; margin-top:4px;">{caption}</div>
        </div>
        """

    col_left, col_right = st.columns([0.4, 0.6])

    with col_left:
        st.markdown("**Sample faces from the dataset (top 5 people, 50 images)**")
        blocks = []
        top_people = sorted(unique_people(labels))[:5]
        for pid in top_people:
            pid_indices = indices_for_person(labels, pid)[:10]
            for i, idx in enumerate(pid_indices):
                color = "royalblue" if i < train_count else "orange"
                blocks.append(
                    bordered_image_html(
                        images[idx],
                        f"{person_display_name(int(pid))} #{i+1}",
                        color=color,
                        width=70,
                    )
                )
        components.v1.html(
            f"""<div style="display:flex; flex-wrap:wrap;">{''.join(blocks)}</div>""",
            height=520,
            scrolling=True,
        )

    with col_right:
        st.markdown("**Selected person (all images)**")
        st.image(x_image, width=200, caption=selection_caption(st.session_state.selected_person, st.session_state.selected_idx))
        person_indices = indices_for_person(labels, st.session_state.selected_person)
        selected_imgs = images[person_indices]
        blocks_sel = []
        for i in range(len(selected_imgs)):
            color = "royalblue" if i < train_count else "orange"
            blocks_sel.append(
                bordered_image_html(
                    selected_imgs[i],
                    f"{person_display_name(st.session_state.selected_person)} #{i+1}",
                    color=color,
                    width=90,
                )
            )
        components.v1.html(
            f"""<div style="display:flex; flex-wrap:wrap;">{''.join(blocks_sel)}</div>""",
            height=520,
            scrolling=True,
        )
    next_button()


def tab_image_vector(x_image, x_flat, shape, total_images: int):
    h, w = shape
    st.subheader("2. Image → Vector")
    st.write(
        f"Each face image is a 2D grid of pixels (H={h}, W={w}) that we flatten into a {h*w}-length vector. "
        "Stacking these vectors gives us a data matrix where each row X_i is the flattened face."
    )
    st.markdown(f"Image size: `{h} x {w} = {h*w}` pixels → vector in `{h*w}`-dimensional space.")
    st.markdown(f"Total images: `{total_images}`, each row of `X` is one flattened face.")
    st.latex(rf"X \in \mathbb{{R}}^{{{total_images} \times ({h}\cdot {w})}},\quad X_i = \mathrm{{vec}}(I_i)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original image**")
        st.image(x_image, width=250)
    with col2:
        st.markdown("**Flattened pixel vector (first 300 values)**")
        line_plot_first_300(x_flat)
    next_button()


def tab_mean_center(mu, x_flat, shape):
    h, w = shape
    st.subheader("3. Mean Face & Centering")
    st.write(
        "We first compute the average (mean) face and subtract it from each image. "
        "This centers the data so PCA focuses on differences between faces rather than overall brightness."
    )
    st.latex(r"\mu = \frac{1}{N}\sum_{i=1}^N X_i,\quad \tilde{{X}}_i = X_i - \mu")

    mean_face = mu.reshape(h, w)
    centered_flat = x_flat - mu
    centered_img_norm = normalize_for_display(centered_flat.reshape(h, w))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Selected face**")
        st.image(x_flat.reshape(h, w), width=200)
    with col2:
        st.markdown("**Mean face**")
        st.image(mean_face, width=200)
    with col3:
        st.markdown("**Centered face (face − mean)**")
        st.image(centered_img_norm, width=200)
    next_button()


def tab_eigenfaces(eigvals, eigvecs, shape):
    h, w = shape
    st.subheader("4. Eigenfaces (Principal Components)")
    st.write(
        "We compute eigenfaces by finding principal directions of variation in the centered faces. "
        "Each eigenface highlights a pattern of variation across the dataset; higher-variance eigenfaces capture more "
        "global structure, lower ones capture finer detail. Variance per eigenface comes from its eigenvalue."
    )
    st.markdown(
        "Compute the covariance on centered faces, then solve the eigen-decomposition; reshaping each eigenvector gives an eigenface "
        "that captures a dominant pattern of facial variation for compact recognition."
    )
    st.latex(
        rf"C = \frac{{1}}{{n-1}}\,\tilde{{X}}^\top \tilde{{X}} \in \mathbb{{R}}^{{({h}\cdot {w})\times({h}\cdot {w})}}"
    )
    st.markdown(
        f"- With {len(eigvals)+1} samples, rank(C) ≤ min({h*w}, {len(eigvals)+1}-1) = {min(h*w, len(eigvals)+1-1)}, so there are at most {len(eigvals)} non-zero eigenvalues/eigenfaces."
    )
    st.latex(r"C v_i = \lambda_i v_i \quad\text{(reshape } v_i \text{ to view each eigenface)}")
    st.markdown("---")
    # Show PC1 and PC2 math + visuals
    st.markdown("<div style='font-size:1.1rem; font-weight:700;'>PC1 and PC2 (math and visuals)</div>", unsafe_allow_html=True)
    if eigvecs.shape[1] >= 2:
        pc1 = eigvecs[:, 0].reshape(h, w)
        pc2 = eigvecs[:, 1].reshape(h, w)
        evr = explained_variance_ratio(eigvals)
        cols_pc = st.columns(2)
        with cols_pc[0]:
            st.markdown(f"**PC1:** eigenvector $v_1 \\in \\mathbb{{R}}^{{{h*w}}}$ (reshaped below), eigenvalue $\\lambda_1 = {eigvals[0]:.3f}$")
            st.image(normalize_for_display(pc1), width=140, caption="PC1 (Eigenface 1)")
            st.caption("PC1 highlights the dominant variation across all faces (e.g., overall illumination/structure).")
        with cols_pc[1]:
            st.markdown(f"**PC2:** eigenvector $v_2 \\in \\mathbb{{R}}^{{{h*w}}}$ (reshaped below), eigenvalue $\\lambda_2 = {eigvals[1]:.3f}$")
            st.image(normalize_for_display(pc2), width=140, caption="PC2 (Eigenface 2)")
            st.caption("PC2 captures the next most significant variation (e.g., contrast or local facial feature changes).")
    st.markdown("---")
    st.markdown("<div style='font-size:1.1rem; font-weight:700;'>Calculating Variance</div>", unsafe_allow_html=True)
    st.latex(r"\text{variance}(v_i) = \lambda_i,\quad \text{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}")
    st.write("Each eigenvalue λ_i measures how much facial variation that eigenface captures. Higher λ_i = more dominant pattern across the dataset.")
    st.markdown("---")

    num_show = min(16, st.session_state.k)
    st.markdown(f"<div style='font-size:1.1rem; font-weight:700;'>Top {num_show} eigenfaces</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    evr = explained_variance_ratio(eigvals)
    patterns = [
        "Overall illumination/structure",
        "Horizontal contrast (eyes vs cheeks)",
        "Vertical contrast (forehead vs chin)",
        "Left-right asymmetry",
        "Eye region details",
        "Mouth/cheek emphasis",
        "Nose bridge variation",
        "Jawline/cheekbone contrast",
        "Forehead/temple shading",
        "Upper-lower face balance",
        "Eyebrow/eye socket differences",
        "Nasal width/contour",
        "Lip/upper-lip shading",
        "Cheek fullness",
        "Chin/jaw prominence",
        "Hairline/upper face shading",
    ]
    for i in range(num_show):
        eigface = eigvecs[:, i].reshape(h, w)
        ef_norm = normalize_for_display(eigface)
        with cols[i % 4]:
            st.image(ef_norm, width=120, caption=f"PC {i+1} (λ={eigvals[i]:.3f})")
            desc = patterns[i] if i < len(patterns) else f"Pattern #{i+1}"
            st.caption(desc)
            st.caption(f"Variance share: {evr[i]*100:.1f}%")
    st.markdown("---")
    st.markdown("<div style='font-size:1.1rem; font-weight:700;'>PC vs cumulative variance</div>", unsafe_allow_html=True)
    variance_plot(evr)
    st.caption(
        "Eigenfaces are ordered by variance (from the covariance eigenvalues). Cumulative variance shows how many PCs are needed; "
        "the selected image is a weighted mix of these eigenfaces (see coefficients in Tab 5)."
    )
    next_button()


def tab_projection_recon(x_flat, mu, eigvecs, shape):
    h, w = shape
    st.subheader("5. Projection & Reconstruction")
    st.write(
        "We project the centered face onto the first k eigenfaces to get coefficients (θ_i). "
        "Using those θ_i with the top eigenfaces reconstructs the image:  ŷ ≈ μ + Σ θ_i · eigenface_i."
    )
    k = st.session_state.k
    z = project(x_flat, mu, eigvecs, k)
    x_hat = reconstruct(z, mu, eigvecs).reshape(h, w)
    x_hat_disp = np.clip(x_hat, 0.0, 1.0)
    mse = np.mean((x_flat - x_hat.flatten()) ** 2)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original face**")
        st.image(x_flat.reshape(h, w), width=250)
    with col2:
        st.markdown(f"**Reconstructed with k = {k} components**")
        st.image(x_hat_disp, width=250)
        st.markdown(f"Reconstruction MSE: `{mse:.6f}`")
    st.markdown(f"**PCA coefficients (first min(20, k={k}) values)**")
    st.caption(
        "Visual combination: reconstructed image ≈ μ "
        + " + ".join([f"θ{i+1}·PC{i+1}" for i in range(min(6, k))])
        + " (showing first few eigenfaces as building blocks)."
    )
    show_blocks = min(6, k, eigvecs.shape[1])
    if show_blocks > 0:
        cols = st.columns(show_blocks)
        for i in range(show_blocks):
            ef = normalize_for_display(eigvecs[:, i].reshape(h, w))
            with cols[i]:
                st.image(ef, width=100, caption=f"PC{i+1}, θ={z[i]:.2f}")
    st.markdown("---")
    col_coeffs, col_mse = st.columns(2)
    with col_coeffs:
        st.markdown(f"**PCA coefficients (1..k={k})**")
        fig, ax = plt.subplots()
        ks_full = np.arange(1, len(z) + 1)
        ax.bar(ks_full, z)
        ax.set_xlim(1, len(z))
        ax.set_xlabel("Component index")
        ax.set_ylabel("Coefficient value")
        ax.set_title("PCA coefficients for this face")
        st.pyplot(fig)
    with col_mse:
        st.markdown("**MSE vs components**")
        max_k = min(len(eigvecs), st.session_state.k)
        ks = np.arange(1, max_k + 1)
        mse_vals = []
        for kk in ks:
            z_tmp = project(x_flat, mu, eigvecs, kk)
            x_hat_tmp = reconstruct(z_tmp, mu, eigvecs).flatten()
            mse_vals.append(np.mean((x_flat - x_hat_tmp) ** 2))
        fig2, ax2 = plt.subplots()
        ax2.plot(ks, mse_vals, marker="o")
        ax2.set_xlim(1, ks[-1])
        ax2.set_xlabel("Number of components")
        ax2.set_ylabel("MSE")
        ax2.set_title("Reconstruction MSE vs k")
        st.pyplot(fig2)
    next_button()


def tab_recognition(X_flat, labels, mu, eigvecs, x_flat, shape):
    h, w = shape
    st.subheader("6. Recognition in PCA Space")
    st.write(
        "For recognition, we represent each person by the mean of their PCA vectors and compare a new face to these "
        "templates using distance in the PCA space."
    )
    k = st.session_state.k
    z_all = (X_flat - mu) @ eigvecs[:, :k]
    templates = build_templates(z_all, labels)
    z_query = project(x_flat, mu, eigvecs, k)
    dists = distances_to_templates(z_query, templates)
    best_pid = min(dists, key=dists.get)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Query face (to recognize)**")
        st.image(
            x_flat.reshape(h, w),
            width=250,
            caption=f"True person: {person_display_name(st.session_state.selected_person)}",
        )
    with col2:
        st.markdown("**Predicted person**")
        st.write(f"**Prediction:** {person_display_name(best_pid)}")
        st.markdown("**Distances to each person template (lower is better)**")
        distance_bar_chart(dists)
    st.info(
        "In a full experiment, you would evaluate recognition accuracy on a held-out test set and compare how it changes as you vary k."
    )


# -----------------------------
# Main app
# -----------------------------
def main():
    logger = init_logger()
    try:
        st.set_page_config(page_title="PCA for Face Recognition (Eigenfaces Demo)", layout="wide")
        inject_inconsolata()
        st.title("PCA for Face Recognition (Eigenfaces Demo)")

        images, labels = load_data()
        X_flat, shape = flatten_images(images)
        mu, eigvals, eigvecs = compute_pca_cache(X_flat)

        init_session_state(labels)

        # Sidebar controls
        st.sidebar.header("Step selection")
        people = unique_people(labels)
        selected_person = st.sidebar.selectbox(
            "Choose a person",
            options=people,
            format_func=person_display_name,
            index=list(people).index(st.session_state.selected_person),
        )
        person_indices = indices_for_person(labels, selected_person)
        selected_idx = int(person_indices[0])
        st.sidebar.markdown("---")
        train_count = st.sidebar.slider(
            "Training images per person (1-9)", min_value=1, max_value=9, value=st.session_state.get("train_count", 6), step=1
        )
        k = st.sidebar.slider(
            "Number of PCA components (k)", min_value=1, max_value=1000, value=min(st.session_state.k, 1000), step=1
        )

        if "log_file" in st.session_state:
            st.sidebar.caption(f"Log file: `{st.session_state['log_file']}`")

        # Detect changes to reset progression
        changed = (
            int(selected_person) != st.session_state.selected_person
            or int(selected_idx) != st.session_state.selected_idx
            or int(k) != st.session_state.k
            or int(train_count) != st.session_state.get("train_count", train_count)
        )

        # Persist selections
        st.session_state.selected_person = int(selected_person)
        st.session_state.selected_idx = int(selected_idx)
        st.session_state.k = int(k)
        st.session_state.train_count = int(train_count)

        if changed:
            set_active_tab("1")

        st.session_state.prev_person = st.session_state.selected_person
        st.session_state.prev_idx = st.session_state.selected_idx
        st.session_state.prev_k = st.session_state.k

        # Selected face
        x_image = images[selected_idx]
        x_flat = X_flat[selected_idx]
        st.session_state.current_x_flat = x_flat

        # Sidebar tab navigation (ensures tabs are always clickable)
        tab_choice = st.sidebar.radio(
            "Navigate tabs",
            options=TAB_KEYS,
            format_func=lambda k: TAB_LABELS[k],
            index=TAB_KEYS.index(st.session_state.active_tab),
        )
        if tab_choice != st.session_state.active_tab:
            set_active_tab(tab_choice)

        # AntD tabs (visual) still rendered
        render_antd_tabs(st.session_state.active_tab)

        def safe_tab(fn, *args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception:
                logger.exception("Error rendering tab")
                st.error("Error rendering this tab. See logs for details.")

        # Tab content
        if st.session_state.active_tab == "1":
            safe_tab(tab_dataset_intro, images, labels, x_image)
        elif st.session_state.active_tab == "2":
            safe_tab(tab_image_vector, x_image, x_flat, shape, total_images=len(images))
        elif st.session_state.active_tab == "3":
            safe_tab(tab_mean_center, mu, x_flat, shape)
        elif st.session_state.active_tab == "4":
            safe_tab(tab_eigenfaces, eigvals, eigvecs, shape)
        elif st.session_state.active_tab == "5":
            safe_tab(tab_projection_recon, x_flat, mu, eigvecs, shape)
        elif st.session_state.active_tab == "6":
            safe_tab(tab_recognition, X_flat, labels, mu, eigvecs, x_flat, shape)
    except Exception:
        logger.exception("Unhandled error in Streamlit app")
        st.error("❌ An unexpected error occurred. Please check the log file under logs/ for details.")


if __name__ == "__main__":
    main()
