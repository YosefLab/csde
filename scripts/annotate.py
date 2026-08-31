"""
Streamlit annotation UI for per-cell panels.

Usage
-----
streamlit run scripts/annotate.py -- --dir /path/to/annotations/R2_macrophages

The available actions are set by ``annotation_mode`` in config.json (written by
export.py):

- ``accept_reject``          1 = accept, 2 = reject
- ``accept_correct_reject``  1 = accept, 2 = correct, 3 = reject

Writes ``annotations.json`` as
``{cell_id: {"action": "accept" | "correct" | "reject", "label": str | None}}``.
``label`` is set only for corrections; the manual cell type of an accepted cell is
resolved from the automated label downstream. This schema is validated by
``csde.read_annotations`` — keep the two in step.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ACCEPT, CORRECT, REJECT = "accept", "correct", "reject"

# Action -> (button glyph, button label). The glyph is what the keyboard handler
# matches on, so each must be unique.
ACTION_STYLE = {
    ACCEPT: ("✓", "Accept"),
    CORRECT: ("✎", "Correct"),
    REJECT: ("✗", "Reject"),
}
MODE_ACTIONS = {
    "accept_reject": [ACCEPT, REJECT],
    "accept_correct_reject": [ACCEPT, CORRECT, REJECT],
}


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


def format_status(record: dict | None) -> str:
    if record is None:
        return "not annotated"
    glyph, label = ACTION_STYLE[record["action"]]
    if record["action"] == CORRECT:
        return f"{glyph} corrected → **{record['label']}**"
    return f"{glyph} {label.lower()}ed"


def main():
    args = parse_args()
    annotation_dir = Path(args.dir)

    cell_type_of_interest = "cell of interest"
    annotation_mode = "accept_correct_reject"
    vocabulary = []
    config_path = annotation_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        cell_type_of_interest = config.get("cell_type_of_interest", cell_type_of_interest)
        annotation_mode = config.get("annotation_mode", annotation_mode)
        vocabulary = config.get("cell_type_vocabulary", [])

    actions = MODE_ACTIONS[annotation_mode]

    st.set_page_config(layout="wide", page_title="Cell Annotator")
    st.title(f"Cell Annotation — {cell_type_of_interest}")

    if CORRECT in actions and not vocabulary:
        st.error(
            "`cell_type_vocabulary` is missing from config.json, so corrections "
            "cannot offer any cell type. Re-run scripts/export.py, or set "
            '`"annotation_mode": "accept_reject"` in config.json.'
        )
        st.stop()

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

    st.subheader(
        f"Cell `{cell_id}` — predicted: **{row['cell_type']}** — {format_status(existing)}"
    )
    st.image(str(row["image_path"]), use_container_width=True)

    def annotate(action: str, label: str | None = None) -> None:
        annotations[cell_id] = {"action": action, "label": label}
        save_annotations(annotations, annotation_dir)
        # Advance to next unannotated after annotating
        remaining = metadata.index[~metadata["cell_id"].isin(annotations)]
        next_idx = next((i for i in remaining if i > idx), None)
        if next_idx is not None:
            st.session_state.current_idx = int(next_idx)
        elif remaining.any():
            st.session_state.current_idx = int(remaining[0])

    select_key = f"correction_choice_{cell_id}"
    pending = st.session_state.get("pending_correction") == cell_id

    if pending:
        # The action buttons are removed while a correction is pending, so the
        # digit shortcuts below match nothing and cannot fire. The selector is
        # rendered under the pan
        # el to keep the image visible while choosing.
        st.markdown(f"**Correcting cell `{cell_id}`** — pick the right cell type:")
        sel_col, cancel_col, _ = st.columns([2, 1, 3])
        with sel_col:
            choice = st.selectbox(
                "Corrected cell type",
                [t for t in vocabulary if t != str(row["cell_type"])],
                index=None,
                placeholder="Type to filter…",
                label_visibility="collapsed",
                key=select_key,
            )
        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                st.session_state.pending_correction = None
                st.rerun()
        if choice is not None:
            annotate(CORRECT, label=choice)
            st.session_state.pending_correction = None
            st.rerun()
    else:
        cols = st.columns([1] * len(actions) + [6 - len(actions)])
        for col, action in zip(cols, actions):
            glyph, label = ACTION_STYLE[action]
            key_hint = actions.index(action) + 1
            with col:
                clicked = st.button(
                    f"{glyph} {label} [{key_hint}]",
                    type="primary" if action == ACCEPT else "secondary",
                    use_container_width=True,
                )
            if clicked:
                if action == CORRECT:
                    # Nothing is written yet: the cell stays unannotated until a
                    # label is picked.
                    st.session_state.pending_correction = cell_id
                    st.session_state.pop(select_key, None)
                else:
                    annotate(action)
                st.rerun()

    # Keyboard shortcuts: digits annotate, ←/→ navigate
    key_to_glyph = {
        str(i + 1): ACTION_STYLE[action][0] for i, action in enumerate(actions)
    }
    components.html(f"""
    <script>
    (function() {{
        const keyToGlyph = {json.dumps(key_to_glyph)};
        const doc = window.parent.document;
        doc.addEventListener('keydown', function(e) {{
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const glyph = keyToGlyph[e.key];
            doc.querySelectorAll('button').forEach(function(btn) {{
                const t = btn.textContent.trim();
                if (glyph && t.startsWith(glyph)) btn.click();
                if (e.key === 'ArrowLeft'  && t.startsWith('←')) btn.click();
                if (e.key === 'ArrowRight' && t.startsWith('Next')) btn.click();
            }});
        }}, true);
    }})();
    </script>
    """, height=0)


if __name__ == "__main__":
    main()
