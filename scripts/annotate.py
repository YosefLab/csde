"""
Streamlit annotation UI for per-cell panels.

Usage
-----
streamlit run scripts/annotate.py -- --dir /path/to/annotations/R2_macrophages
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Annotation directory (output of export.py).")
    return p.parse_args()


def load_annotations(annotation_dir: Path) -> dict:
    ann_path = annotation_dir / "annotations.json"
    if ann_path.exists():
        with open(ann_path) as f:
            return json.load(f)
    return {}


def save_annotations(annotations: dict, annotation_dir: Path) -> None:
    with open(annotation_dir / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)


def main():
    args = parse_args()
    annotation_dir = Path(args.dir)

    cell_type_of_interest = "cell of interest"
    config_path = annotation_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        cell_type_of_interest = config.get("cell_type_of_interest", cell_type_of_interest)

    st.set_page_config(layout="wide", page_title="Cell Annotator")
    st.title(f"Cell Annotation — {cell_type_of_interest}")

    metadata_path = annotation_dir / "metadata.csv"
    if not metadata_path.exists():
        st.warning("metadata.csv not found — waiting for export.py to write the first cell.")
        st.stop()

    metadata = pd.read_csv(metadata_path)
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    n_total = len(metadata)

    annotations = load_annotations(annotation_dir)
    n_done = len(annotations)

    st.progress(n_done / n_total, text=f"{n_done} / {n_total} annotated")

    # Initialize navigation index to first unannotated cell
    if "current_idx" not in st.session_state:
        unannotated_mask = ~metadata["cell_id"].isin(annotations)
        first_unannotated = unannotated_mask.idxmax() if unannotated_mask.any() else 0
        st.session_state.current_idx = int(first_unannotated)

    idx = st.session_state.current_idx

    # Jump-to input (form so Enter submits without looping)
    jump_col, nav_col = st.columns([2, 3])
    with jump_col:
        with st.form("jump_form", clear_on_submit=True):
            fc1, fc2 = st.columns([4, 1])
            with fc1:
                jump_id = st.text_input("Jump to cell ID", placeholder="paste cell_id here", label_visibility="collapsed")
            with fc2:
                submitted = st.form_submit_button("Go")
        if submitted and jump_id.strip():
            matches = metadata.index[metadata["cell_id"] == jump_id.strip()].tolist()
            if matches:
                st.session_state.current_idx = matches[0]
                idx = matches[0]
            else:
                st.warning(f"Cell ID `{jump_id.strip()}` not found.")

    with nav_col:
        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("← Prev", use_container_width=True, disabled=(idx == 0)):
                st.session_state.current_idx = idx - 1
                st.rerun()
        with nav2:
            st.markdown(f"<div style='text-align:center; padding-top:6px'>{idx + 1} / {n_total}</div>", unsafe_allow_html=True)
        with nav3:
            if st.button("Next →", use_container_width=True, disabled=(idx == n_total - 1)):
                st.session_state.current_idx = idx + 1
                st.rerun()

    row = metadata.iloc[idx]
    cell_id = row["cell_id"]
    existing = annotations.get(cell_id)
    status = "✓ correct" if existing is True else ("✗ incorrect" if existing is False else "not annotated")

    st.subheader(f"Cell `{cell_id}` — predicted: **{row['cell_type']}** — {status}")
    st.image(str(row["image_path"]), use_container_width=True)

    def annotate(is_correct: bool) -> None:
        annotations[cell_id] = is_correct
        save_annotations(annotations, annotation_dir)
        # Advance to next unannotated after annotating
        remaining = metadata.index[~metadata["cell_id"].isin(annotations)]
        next_idx = next((i for i in remaining if i > idx), None)
        if next_idx is not None:
            st.session_state.current_idx = int(next_idx)
        elif remaining.any():
            st.session_state.current_idx = int(remaining[0])

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("✓ Correct [1]", type="primary", use_container_width=True):
            annotate(True)
            st.rerun()
    with col2:
        if st.button("✗ Incorrect [2]", use_container_width=True):
            annotate(False)
            st.rerun()

    # Keyboard shortcuts: 1/2 annotate, ←/→ navigate
    components.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        doc.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            doc.querySelectorAll('button').forEach(function(btn) {
                const t = btn.textContent.trim();
                if (e.key === '1' && t.startsWith('✓')) btn.click();
                if (e.key === '2' && t.startsWith('✗')) btn.click();
                if (e.key === 'ArrowLeft'  && t.startsWith('←')) btn.click();
                if (e.key === 'ArrowRight' && t.startsWith('Next')) btn.click();
            });
        }, true);
    })();
    </script>
    """, height=0)


if __name__ == "__main__":
    main()
