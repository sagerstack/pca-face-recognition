import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit import components

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data_utils import load_att_faces
from common.image_ops import flatten_images, normalize_for_display
from common.selection import person_display_name, template_mean_by_person

BASE_DIR = Path(__file__).resolve().parents[2]  # points to demo/

# -----------------------------
# Logging
# -----------------------------
def init_logger() -> logging.Logger:
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    if "log_file_fr" not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.session_state["log_file_fr"] = str(logs_dir / f"streamlit-app-{timestamp}.log")
    log_path = Path(st.session_state["log_file_fr"])

    logger = logging.getLogger("face_recognition_page")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s %(pathname)s:%(lineno)d %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    fh_root = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh_root.setFormatter(formatter)
    root_logger.addHandler(fh_root)

    st.session_state["log_file"] = str(log_path)
    logger.info(f"Logging initialized at {log_path}")
    return logger


# Initialize logger early to capture import-time errors
LOGGER = init_logger()

# Load pca_math from project math folder (avoid builtin math)
try:
    _pca_math_path = BASE_DIR / "src" / "math" / "pca_math.py"
    _spec = importlib.util.spec_from_file_location("pca_math_module", _pca_math_path)
    _pca_math = importlib.util.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(_pca_math)  # type: ignore[arg-type]
    compute_pca_svd = _pca_math.compute_pca_svd
    project = _pca_math.project
    reconstruct = _pca_math.reconstruct
except Exception:
    LOGGER.exception("Failed to load pca_math module")
    raise


# -----------------------------
# UI helpers
# -----------------------------
def inject_inconsolata():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inconsolata', monospace !important; }
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


def scroll_to_top():
    components.v1.html(
        "<script>window.scrollTo({top: 0, behavior: 'auto'});</script>",
        height=0,
        width=0,
    )


def next_button(ready: bool = True, disabled_reason: str = ""):
    current = st.session_state.active_tab
    idx = TAB_KEYS.index(current)
    back_col, spacer, next_col = st.columns([1, 12, 1])
    with back_col:
        if idx > 0 and st.button("Back", key=f"back-{current}"):
            st.session_state.active_tab = TAB_KEYS[idx - 1]
            st.rerun()
    with next_col:
        if idx < len(TAB_KEYS) - 1:
            disabled = not ready
            label = "Next" 
            if st.button(label, key=f"next-{current}", disabled=disabled):
                st.session_state.active_tab = TAB_KEYS[idx + 1]
                st.rerun()


# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    return load_att_faces(str(BASE_DIR / "data" / "ATnT"))


@st.cache_resource(show_spinner=True)
def compute_pca_cache(X_flat: np.ndarray):
    return compute_pca_svd(X_flat)


def split_train_test(labels: np.ndarray, train_per_person: int):
    rng = np.random.default_rng(0)
    train_indices = []
    test_indices = []
    for pid in np.unique(labels):
        idxs = np.where(labels == pid)[0]
        rng.shuffle(idxs)
        train = idxs[:train_per_person]
        test = idxs[train_per_person:]
        train_indices.extend(train.tolist())
        test_indices.extend(test.tolist())
    return np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)


def build_templates(z_train: np.ndarray, labels_train: np.ndarray) -> Dict[int, np.ndarray]:
    return template_mean_by_person(z_train, labels_train)


def evaluate_recognition(z_test: np.ndarray, labels_test: np.ndarray, templates: Dict[int, np.ndarray]) -> Tuple[float, np.ndarray]:
    preds = []
    dists_all = []
    for i in range(len(z_test)):
        dists = {pid: float(np.linalg.norm(z_test[i] - tvec)) for pid, tvec in templates.items()}
        pred = min(dists, key=dists.get)
        preds.append(pred)
        dists_all.append(dists)
    preds = np.array(preds)
    acc = float(np.mean(preds == labels_test)) if len(labels_test) > 0 else 0.0
    return acc, dists_all


# -----------------------------
# Tabs content
# -----------------------------
TAB_KEYS = ["1", "2", "3", "4"]
TAB_LABELS = {
    "1": "1. Split & Inputs",
    "2": "2. Project & Reconstruct",
    "3": "3. Compute Distances",
    "4": "4. Recognition",
}


def tab_split_inputs(labels, hw, mu, eigvals, eigvecs, train_count, train_idx, x_flat_shape):
    scroll_to_top()
    st.subheader("1. Split & Inputs")
    st.info("Use the sidebar sliders for train-test split and number of components. Upload an image to proceed.")
    k_ready = min(st.session_state.k, len(eigvals), len(train_idx) - 1 if len(train_idx) > 0 else 0, x_flat_shape)

    col_info, col_upload = st.columns([0.6, 0.4])
    with col_info:
        st.markdown(
            f"""
            ✅ AT&T dataset loaded for 40 subjects with 280 images used for training (per-person split = 7)  

            ✅ PCA fitted on 112×92 grayscale faces (flattened μ length = 10,304; eigenvectors shape = (10,304, 280))  

            ✅ Preprocessed eigenfaces ready for recognition: **{k_ready}** components (bounded by N_train−1 and D)
            """
        )
    with col_upload:
        st.caption("Note: Max usable components are bounded by training samples (N-1).")
        uploaded = st.file_uploader("Upload a face image (will be resized)", type=["png", "jpg", "jpeg", "pgm"])
        if uploaded is not None:
            try:
                from PIL import Image

                h, w = hw  # hw passed in as (H, W)
                img = Image.open(uploaded).convert("L").resize((w, h))
                img_arr = np.array(img, dtype=np.float32)
                img_norm = np.clip(img_arr / 255.0, 0.0, 1.0)
                st.session_state["uploaded_image"] = img_norm
                st.session_state["uploaded_image_name"] = uploaded.name
                st.success("Uploaded image saved for recognition.")
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                st.session_state.pop("uploaded_image", None)
                st.session_state.pop("uploaded_image_name", None)

        if "uploaded_image" in st.session_state:
            caption = f"Uploaded: {st.session_state.get('uploaded_image_name', 'normalized')}"
            st.image(st.session_state["uploaded_image"], width=200, caption=caption)

    ready = "uploaded_image" in st.session_state
    next_button(ready=ready, disabled_reason="Upload an image first")


def tab_metrics(X_flat, labels, train_idx, test_idx, mu, eigvecs):
    scroll_to_top()
    st.subheader("2. Project & Reconstruct")
    if "uploaded_image" not in st.session_state:
        st.warning("Upload an image in Tab 1 to view the projection steps.")
        next_button(ready=False, disabled_reason="Upload an image first")
        return
    k = st.session_state.k
    k_eff = min(k, eigvecs.shape[1])
    k_eff = min(k, eigvecs.shape[1])
    k_eff = min(k, eigvecs.shape[1])
    k_eff = min(k, eigvecs.shape[1])
    k_eff = min(k, eigvecs.shape[1])
    k_eff = min(k, eigvecs.shape[1])
    h, w = st.session_state["shape_hw"]

    img = st.session_state["uploaded_image"]
    flat = img.reshape(-1)
    centered = flat - mu
    z_query = centered @ eigvecs[:, :k_eff]
    recon_flat = reconstruct(z_query, mu, eigvecs[:, :k_eff])
    centered_img = normalize_for_display(centered.reshape(h, w))
    recon_img = normalize_for_display(recon_flat.reshape(h, w))

    st.markdown(
        """
        **Processing steps**

        1. Normalize input (already grayscale, resized to 92x112).  
        2. Center by subtracting the training-set mean face μ.  
        3. Project onto the first k eigenfaces to get coefficients (one per component).  
        4. Reconstruct using the same k components:  x̂ = μ + Σ (αᵢ · eigenfaceᵢ).
        """
    )
    st.markdown("---")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(img, width=220, caption="Uploaded image")
    with c2:
        st.image(centered_img, width=220, caption="Centered (x - μ)")
    st.markdown("---")

    # Visual projection breakdown with eigenfaces
    top_show = min(5, k_eff)
    ef_imgs = [normalize_for_display(eigvecs[:, i].reshape(h, w)) for i in range(top_show)]
    st.markdown(
        f"""
        **Projection of centered(x) onto eigenfaces (PCs) to calculate coefficients (showing first {top_show})**
        """
    )
    recon_top = reconstruct(z_query[:top_show], mu, eigvecs[:, :top_show])
    recon_top_img = normalize_for_display(recon_top.reshape(h, w))

    cols = st.columns(top_show)
    for i in range(top_show):
        with cols[i]:
            st.image(ef_imgs[i], width=120, caption=f"ef{i+1}\nθ{i+1} = {z_query[i]:.3f}")
    st.markdown("---")

    # Full coefficients across selected k
    st.markdown(f"**All coefficients across k={k} components**")
    fig_coef, ax_coef = plt.subplots(figsize=(7, 3))
    ax_coef.bar(np.arange(1, k_eff + 1), z_query[:k_eff])
    ax_coef.set_xlabel("Component index")
    ax_coef.set_ylabel("Coefficient (θᵢ)")
    ax_coef.set_title("Projection coefficients for input image")
    st.pyplot(fig_coef)
    st.markdown("---")

    st.markdown("**Reconstruction Test**  \nx̂ = μ + Σ (θᵢ · eigenfaceᵢ) using all k components")
    mean_face_img = normalize_for_display(mu.reshape(h, w))
    recon_k = reconstruct(z_query, mu, eigvecs[:, :k_eff])
    recon_k_img = normalize_for_display(recon_k.reshape(h, w))

    st.markdown("Mean face and first 5 eigenfaces with coefficients (summed to reconstruct)")
    # mean face + plus signs + 5 eigenfaces + equals + reconstructed image (all in one row)
    total_cols = (top_show + 2) + (top_show + 1)  # images plus separators
    chain_cols = st.columns(total_cols, gap="small")
    symbol_html = "<div style='display:flex; align-items:center; justify-content:center; height:180px; font-size:24px;'>{{sym}}</div>"
    idx_col = 0
    # Mean face
    with chain_cols[idx_col]:
        st.image(mean_face_img, width=110, caption="μ")
    idx_col += 1
    # Plus between mean and first eigenface
    with chain_cols[idx_col]:
        st.markdown(symbol_html.replace("{{sym}}", "+"), unsafe_allow_html=True)
    idx_col += 1
    # Eigenfaces with separators
    for i in range(top_show):
        with chain_cols[idx_col]:
            st.image(ef_imgs[i], width=110, caption=f"ef{i+1}\nθ{i+1} = {z_query[i]:.3f}")
        idx_col += 1
        if i < top_show - 1:
            with chain_cols[idx_col]:
                st.markdown(symbol_html.replace("{{sym}}", "+"), unsafe_allow_html=True)
            idx_col += 1
    # Equals before reconstructed image
    with chain_cols[idx_col]:
        st.markdown(symbol_html.replace("{{sym}}", "="), unsafe_allow_html=True)
    idx_col += 1
    with chain_cols[idx_col]:
        st.image(recon_k_img, width=120, caption=f"x̂ using all k={k}")
    st.markdown("---")

    st.caption(f"Reconstruction uses full θ (length k={k}); preview of first 5 shown above. x̂ = μ + Σ θᵢ·eigenfaceᵢ")

    st.info("Distances to templates are computed in Tab 3 using these coefficients.")
    next_button()


def tab_recognition(images, labels, train_idx, mu, eigvecs):
    scroll_to_top()
    st.subheader("4. Recognition")
    if "uploaded_image" not in st.session_state:
        st.warning("Upload an image and finish earlier tabs first.")
        next_button(ready=False, disabled_reason="Upload image first")
        return

    k = st.session_state.k
    k_eff = min(k, eigvecs.shape[1])
    img = st.session_state["uploaded_image"]
    h, w = st.session_state["shape_hw"]
    flat = img.reshape(-1)
    centered = flat - mu
    z_query = centered @ eigvecs[:, :k_eff]

    z_all = (images.reshape(len(images), -1) - mu) @ eigvecs[:, :k_eff]
    templates = build_templates(z_all[train_idx], labels[train_idx])
    if not templates:
        st.warning("No templates available. Increase training images per person.")
        next_button(ready=False, disabled_reason="Need templates")
        return

    dists = {pid: float(np.linalg.norm(z_query - tvec)) for pid, tvec in templates.items()}

    # Top-5 closest templates with images
    st.markdown("**Top 5 closest templates (PCA space)**")
    st.markdown("Given Input Image:")
    st.image(img, width=150)
    st.markdown("Below are the training images closest to the input image's projection in PCA space")
    top5 = sorted(dists.items(), key=lambda x: x[1])[:5]
    recognized_pid = top5[0][0] if top5 else None
    img_cols = st.columns(len(top5))
    for idx, (pid, dist_val) in enumerate(top5):
        with img_cols[idx]:
            samples = [i for i in train_idx if labels[i] == pid]
            if samples:
                caption_style = "color:green;" if idx == 0 else ""
                st.markdown(
                    f"<div style='text-align:center; {caption_style}'>{person_display_name(pid)}</div>",
                    unsafe_allow_html=True,
                )
                st.image(images[samples[0]], width=150)
            st.markdown(
                f"""
                <div style="text-align:left; { 'color:green;' if idx==0 else '' }">
                    Distance: <strong>{dist_val:.6f}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if idx == 0:
                st.markdown(
                    """
                    <div style="
                        text-align: left;
                        margin-top: 6px;
                        color: green;
                        font-size: 20px;
                    ">
                        ✔ Closest
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    if recognized_pid is not None:
        st.info(f"Recognized input image as {person_display_name(int(recognized_pid))}")
    st.markdown("---")

    # Navigation: Back and Home (to Tab 1) on the same row
    nav_cols = st.columns([1, 8, 1])
    with nav_cols[0]:
        if st.button("Back", key="back-4"):
            st.session_state.active_tab = TAB_KEYS[TAB_KEYS.index("4") - 1]
            st.rerun()
    with nav_cols[2]:
        if st.button("Home", key="home-4"):
            st.session_state.active_tab = "1"
            st.rerun()


def tab_sample_results(images, labels, train_idx, test_idx, mu, eigvecs):
    scroll_to_top()
    st.subheader("3. Compute Distances between Input Image & Training Images Projections")
    if "uploaded_image" not in st.session_state:
        st.warning("Upload an image and finish projection in Tab 2 before comparing distances.")
        next_button(ready=False, disabled_reason="Upload image first")
        return

    k = st.session_state.k
    k_eff = min(k, eigvecs.shape[1])
    z_all = (images.reshape(len(images), -1) - mu) @ eigvecs[:, :k_eff]
    templates = build_templates(z_all[train_idx], labels[train_idx])

    img = st.session_state["uploaded_image"]
    h, w = st.session_state["shape_hw"]
    flat = img.reshape(-1)
    centered = flat - mu
    z_query = centered @ eigvecs[:, :k_eff]
    recon_flat = reconstruct(z_query, mu, eigvecs[:, :k_eff])
    recon_img = normalize_for_display(recon_flat.reshape(h, w))
    dists = {pid: float(np.linalg.norm(z_query - tvec)) for pid, tvec in templates.items()}

    col_info, col_chart = st.columns(2)
    with col_info:
        st.markdown(
            r"""
            **How distances are computed**

            - Let **x** = projection coefficients of the uploaded image (length k).  
            - Let **yᵖ** = template coefficients for person p (mean of that person's training projections, length k).  
            - Distance uses Euclidean norm in PCA space:  **d(x, yᵖ) = ‖x − yᵖ‖₂**.  
            - Templates are generated with the same pipeline as the uploaded image: center by μ, project onto the first k eigenfaces, then average per person.
            """
        )
    with col_chart:
        # Example visualization in first 3 components
        comp_dim = min(3, k)
        def take3(vec):
            if comp_dim == 3:
                return vec[:3]
            pad = np.zeros(3)
            pad[:comp_dim] = vec[:comp_dim]
            return pad

        x3 = take3(z_query)
        persons = list(templates.keys())[:3]  # show only three templates y1,y2,y3
        y3_list = {pid: take3(tvec) for pid, tvec in templates.items() if pid in persons}

        fig = plt.figure(figsize=(3.4, 2.1))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*x3, c="#ff6b6b", s=55, label="Input x")
        colors = ["#4c78a8", "#72b7b2", "#54a24b"]
        for idx, pid in enumerate(persons):
            yv = y3_list[pid]
            ax.scatter(*yv, c=colors[idx % len(colors)], s=45, label=f"Template y{idx+1}")
            ax.plot([x3[0], yv[0]], [x3[1], yv[1]], [x3[2], yv[2]], c=colors[idx % len(colors)], alpha=0.7, linestyle="--")
        ax.set_xlabel("PC1", fontsize=5)
        ax.set_ylabel("PC2", fontsize=5)
        ax.set_zlabel("PC3", fontsize=5)
        ax.tick_params(labelsize=6)
        ax.set_title("Distances in first 3 PCs", fontsize=8)
        ax.legend(loc="upper right", fontsize=6)
        st.pyplot(fig, use_container_width=False)
    st.markdown("---")

    # Removed extra uploaded image display to focus on distances

    next_button()


# -----------------------------
# Main
# -----------------------------
def main():
    logger = LOGGER
    try:
        st.set_page_config(page_title="Face Recognition", layout="wide")
        inject_inconsolata()
        st.title("Face Recognition")

        images, labels = load_data()
        X_flat, (H, W) = flatten_images(images)

        if "active_tab" not in st.session_state or st.session_state.active_tab not in TAB_KEYS:
            st.session_state.active_tab = "1"
        if "k" not in st.session_state:
            st.session_state.k = 30
        if "train_count" not in st.session_state:
            st.session_state.train_count = 6
        st.session_state["shape_hw"] = (H, W)

        # Sidebar controls (reuse existing)
        st.sidebar.header("Face Recognition Controls")
        train_count = st.sidebar.slider(
            "Training images per person (1-9)",
            min_value=1,
            max_value=9,
            value=st.session_state.train_count,
            step=1,
            key="train_per_person_fr",
        )
        train_idx, test_idx = split_train_test(labels, train_count)
        if len(train_idx) < 2:
            st.error("Not enough training samples to compute PCA. Increase training images per person.")
            return

        max_k = max(1, min(len(train_idx) - 1, X_flat.shape[1], 400))
        k = st.sidebar.slider(
            "Number of PCA components (k)",
            min_value=1,
            max_value=max_k,
            value=min(st.session_state.k, max_k),
            step=1,
            key="k_slider_fr",
        )
        st.sidebar.caption(f"Log file: `{st.session_state.get('log_file','')}`")

        st.session_state.train_count = train_count
        st.session_state.k = k

        # Fit PCA on training subset only
        mu, eigvals, eigvecs = compute_pca_svd(X_flat[train_idx])

        render_antd_tabs(st.session_state.active_tab)

        if st.session_state.active_tab == "1":
            tab_split_inputs(labels, (H, W), mu, eigvals, eigvecs, train_count, train_idx, X_flat.shape[1])
        elif st.session_state.active_tab == "2":
            tab_metrics(X_flat, labels, train_idx, test_idx, mu, eigvecs)
        elif st.session_state.active_tab == "3":
            tab_sample_results(images, labels, train_idx, test_idx, mu, eigvecs)
        elif st.session_state.active_tab == "4":
            tab_recognition(images, labels, train_idx, mu, eigvecs)

    except Exception:
        logger.exception("Unhandled error in Face Recognition page")
        st.error("❌ An unexpected error occurred. Please check the log file under logs/ for details.")


if __name__ == "__main__":
    main()
