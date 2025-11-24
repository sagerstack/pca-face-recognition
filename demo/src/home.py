"""
Home page for the PCA face recognition Streamlit app.
Provides an introduction and links to the two feature pages.
"""

import streamlit as st

st.set_page_config(page_title="PCA Face Recognition Home", layout="wide")

# Lightweight shadcn-like card styles
st.markdown(
    """
    <style>
    .card-grid {display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 1.25rem;}
    .card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.25rem; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06); transition: box-shadow 0.2s, transform 0.2s; display: flex; flex-direction: column; height: 100%;}
    .card h3 {margin: 0 0 0.35rem 0; font-size: 1.05rem;}
    .card p {margin: 0 0 0.75rem 0; line-height: 1.5;}
    .card ul {padding-left: 1.1rem; margin: 0 0 0.75rem 0;}
    .card li {margin-bottom: 0.25rem;}
    .card a {text-decoration: none; font-weight: 600;}
    .card-link {text-decoration: none; color: inherit; display: block;}
    .card-link:hover .card {box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-2px);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PCA for Face Recognition")
st.write(
    "Explore PCA eigenfaces and run face recognition experiments. "
    "Use the cards below to open each experience."
)

st.markdown(
    """
    <div class="card-grid">
      <div class="card">
        <h3>🖼️ PCA Eigenfaces</h3>
        <p>Inspect the dataset, mean face, eigenfaces, projections, and a recognition demo.</p>
        <ul>
          <li>Sidebar: choose person/image.</li>
          <li>Sidebar: set PCA components <strong>k</strong> (up to 1000; practical cap = samples−1).</li>
        </ul>
      </div>
      <div class="card">
        <h3>🧠 Face Recognition</h3>
        <p>Train/test recognition with PCA dimensionality control.</p>
        <ul>
          <li>Sidebar: <strong>Training images per person (1-9)</strong> controls split.</li>
          <li>Sidebar: <strong>k</strong> controls PCA dimensions; metrics and per-sample results update accordingly.</li>
        </ul>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1-PCA Eigenfaces.py", label="Try It Out →", icon="🖼️")
with col2:
    st.page_link("pages/2-Face Recognition.py", label="Try It Out →", icon="🧭")

st.markdown("---")
st.write(
    "Tips:\n"
    "- Back/Next buttons on each page step through tabs.\n"
    "- If you adjust sliders, tabs reset to keep the walkthrough consistent.\n"
    "- Logs are written under `demo/logs/` for troubleshooting."
)
