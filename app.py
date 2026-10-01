"""
app.py — Trip Photo Organizer AI (v2 · Community-Cloud-safe)

Major changes from v1:
  • MAX_PHOTOS = 15 hard-cap with st.stop() enforcement.
  • YOLO / ReID removed — only InsightFace buffalo_s remains.
  • Human-in-the-loop: cluster rename, multi-select merge, live epsilon slider.
  • Gallery preview with EXIF metadata per folder.
  • Smart categories: Person_N, Group_Photos, Landscapes_or_NoFace.
"""

import os
import gc
import time
import asyncio

# ── Low-level env vars MUST come before any ML import ──────────
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np
import streamlit as st
from PIL import Image

from utils.detector import (
    get_face_analyzer,
    detect_and_draw_faces,
    extract_embeddings_from_files,
    MAX_PHOTOS,
)
from utils.cluster import cluster_face_embeddings
from utils.organizer import build_output_folders, extract_exif
from utils.zipper import create_zip_from_directory

# ── Layout constants ───────────────────────────────────────────
SCROLL_THRESHOLD = 12
GRID_HEIGHT = 560


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def load_image(uploaded_file):
    """Validate and load a PIL image from an uploaded file."""
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            img.verify()
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            return img.convert("RGB").copy(), None
    except Exception as e:
        return None, str(e)


def load_image_from_path(image_path):
    """Load a PIL image and detach pixel data from the on-disk file handle."""
    with Image.open(image_path) as img:
        return img.convert("RGB").copy()


def render_photo_grid(items, cols_count=4, height=None):
    """Render (image, caption, ok) tuples in a responsive column grid."""
    container = (
        st.container(height=height, border=True)
        if height
        else st.container(border=False)
    )
    with container:
        cols = st.columns(cols_count)
        for idx, (img, caption, ok) in enumerate(items):
            col = cols[idx % cols_count]
            if ok:
                col.image(img, caption=caption, use_container_width=True)
            else:
                col.error(caption)


# ───────────────────────────────────────────────────────────────
# Face-thumbnail helpers for the merge/rename UI
# ───────────────────────────────────────────────────────────────
def _crop_face_thumbnail(file_obj, bbox, size=96):
    """Return a small square crop of the face region for the merge UI."""
    try:
        file_obj.seek(0)
        with Image.open(file_obj) as img:
            img = img.convert("RGB")
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            crop = img.crop((x1, y1, x2, y2))
            crop.thumbnail((size, size))
            return crop
    except Exception:
        return None


def _build_cluster_preview(metadata, labels):
    """Group face metadata entries by cluster label and collect thumbnails.

    Returns
    -------
    cluster_thumbs : dict[int, list[PIL.Image]]
    cluster_files  : dict[int, set[str]]
    """
    cluster_thumbs: dict[int, list] = {}
    cluster_files: dict[int, set[str]] = {}

    label_idx = 0
    for meta in metadata:
        if meta["type"] != "face":
            continue
        cid = int(labels[label_idx])
        label_idx += 1

        cluster_thumbs.setdefault(cid, [])
        cluster_files.setdefault(cid, set())

        if meta["bbox"] is not None and len(cluster_thumbs[cid]) < 6:
            thumb = _crop_face_thumbnail(meta["file_obj"], meta["bbox"])
            if thumb:
                cluster_thumbs[cid].append(thumb)

        cluster_files[cid].add(meta["file_name"])

    return cluster_thumbs, cluster_files


# ───────────────────────────────────────────────────────────────
# Main application
# ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Trip Photo Organizer AI",
        page_icon="📸",
        layout="wide",
    )

    st.title("📸 Trip Photo Organizer AI")
    st.caption(
        "Automatically group trip photos by people using on-device facial recognition."
    )
    st.markdown("---")

    # ── Sidebar: clustering control ─────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        eps_val = st.slider(
            "Cluster Sensitivity (Epsilon)",
            min_value=0.30,
            max_value=0.70,
            value=0.50,
            step=0.01,
            help=(
                "Cosine distance threshold.  **Lower** = strict matching "
                "(more person folders, fewer false merges).  **Higher** = "
                "looser matching (fewer folders, tolerates lighting/pose changes)."
            ),
        )
        st.caption(f"Current ε = {eps_val:.2f}")
        st.markdown("---")
        st.info(
            f"📌 **Photo limit**: up to **{MAX_PHOTOS}** photos per batch to stay "
            "within the 1 GB RAM budget on Streamlit Community Cloud."
        )

    # ── Step 1: Upload ──────────────────────────────────────────
    st.header("1 ∙ Upload Trip Photos")
    uploaded_files = st.file_uploader(
        "Choose trip photos (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Please upload your photos above to get started.")
        return

    num_files = len(uploaded_files)

    # ── Enforce hard photo limit ────────────────────────────────
    if num_files > MAX_PHOTOS:
        st.error(
            f"🚫 You uploaded **{num_files}** photos — the maximum is "
            f"**{MAX_PHOTOS}** to avoid out-of-memory crashes.  "
            "Please remove some files and try again."
        )
        st.stop()

    st.success(f"✅ Loaded **{num_files}** photo(s)")

    # ── Step 2: Face Detection Preview ──────────────────────────
    st.markdown("---")
    st.header("2 ∙ Face Detection Preview")

    cache_key = tuple(sorted(f.name for f in uploaded_files))
    if st.session_state.get("preview_cache_key") != cache_key:
        with st.spinner("Loading face detection model (first run only)…"):
            analyzer = get_face_analyzer()

        grid_items = []
        total_faces = 0
        bar = st.progress(0, text="Scanning images…")

        for idx, file in enumerate(uploaded_files):
            bar.progress(
                (idx + 1) / num_files,
                text=f"Analysing {idx + 1}/{num_files}: `{file.name}`",
            )
            img, err = load_image(file)
            if img is not None:
                try:
                    annotated, faces, _ = detect_and_draw_faces(img, analyzer)
                    nf = len(faces)
                    total_faces += nf
                    cap = f"{file.name} ({nf} face{'s' if nf != 1 else ''})"
                    grid_items.append((annotated, cap, True))
                except Exception as e:
                    grid_items.append((None, f"Error: {file.name}: {e}", False))
            else:
                grid_items.append((None, f"Failed: {file.name}", False))
            del img
            gc.collect()

        bar.empty()
        st.session_state.update(
            preview_cache_key=cache_key,
            grid_items=grid_items,
            total_faces_found=total_faces,
        )

    grid_items = st.session_state["grid_items"]
    total_faces = st.session_state["total_faces_found"]
    analyzer = get_face_analyzer()  # cached — no re-load

    m1, m2, m3 = st.columns(3)
    m1.metric("Photos Uploaded", num_files)
    m2.metric("Faces Detected", total_faces)
    m3.metric("Avg Faces / Photo", round(total_faces / max(num_files, 1), 2))

    use_scroll = num_files > SCROLL_THRESHOLD
    if use_scroll:
        st.caption(f"Showing all {num_files} photos — scroll inside the box.")
        render_photo_grid(grid_items, cols_count=4, height=GRID_HEIGHT)
    else:
        render_photo_grid(grid_items, cols_count=4)

    # ── Step 3: Cluster + Organize ──────────────────────────────
    st.markdown("---")
    st.header("3 ∙ Clustering & Organisation")

    if st.button("🚀 Process & Organise Photos", type="primary"):
        pbar = st.progress(0, text="Starting pipeline…")
        try:
            pbar.progress(20, text="Extracting 512-D face embeddings…")
            embeddings, metadata = extract_embeddings_from_files(
                uploaded_files, analyzer
            )

            pbar.progress(50, text="Clustering with DBSCAN…")
            face_embeddings = embeddings  # already (N,512) from detector
            labels = (
                cluster_face_embeddings(face_embeddings, eps=eps_val, min_samples=1)
                if len(face_embeddings) > 0
                else np.array([], dtype=int)
            )

            # Store for the merge/rename UI
            st.session_state["embeddings"] = embeddings
            st.session_state["metadata"] = metadata
            st.session_state["labels"] = labels
            st.session_state["eps_val"] = eps_val

            pbar.progress(100, text="✅ Clustering complete!")
            time.sleep(0.4)
            pbar.empty()
            st.toast("Faces clustered — review below!", icon="🎉")

        except Exception as e:
            pbar.empty()
            st.error(f"Pipeline error: {e}")

        gc.collect()

    # ── Step 4: Merge / Rename UI ───────────────────────────────
    if "labels" not in st.session_state:
        return

    labels = st.session_state["labels"]
    metadata = st.session_state["metadata"]

    if len(labels) == 0:
        st.warning("No face embeddings were extracted — nothing to cluster.")
        return

    st.markdown("---")
    st.header("4 ∙ Review & Merge Person Clusters")

    cluster_thumbs, cluster_files = _build_cluster_preview(metadata, labels)
    unique_ids = sorted(cluster_thumbs.keys())

    if not unique_ids:
        st.info("No clusters to display.")
        return

    # ── Thumbnail overview ──────────────────────────────────────
    st.subheader("🧑‍🤝‍🧑 Detected Person Clusters")
    for cid in unique_ids:
        default_name = f"Person_{cid + 1}" if cid >= 0 else "Unclustered"
        n_files = len(cluster_files[cid])
        with st.expander(
            f"👤 {default_name}  —  {n_files} photo(s)", expanded=True
        ):
            thumb_cols = st.columns(min(len(cluster_thumbs[cid]), 6))
            for i, thumb in enumerate(cluster_thumbs[cid]):
                thumb_cols[i % len(thumb_cols)].image(
                    thumb, caption=f"Face {i+1}", use_container_width=True
                )

    # ── Rename inputs ───────────────────────────────────────────
    st.subheader("✏️ Rename Folders")
    custom_names: dict[int, str] = {}
    rename_cols = st.columns(min(len(unique_ids), 4))
    for i, cid in enumerate(unique_ids):
        default = f"Person_{cid + 1}" if cid >= 0 else "Unclustered"
        col = rename_cols[i % len(rename_cols)]
        new_name = col.text_input(
            f"Name for cluster {cid}",
            value=default,
            key=f"rename_{cid}",
        )
        if new_name and new_name != default:
            custom_names[cid] = new_name

    # ── Merge selector ──────────────────────────────────────────
    st.subheader("🔗 Merge Split Clusters")
    st.caption(
        "If the same person was split into multiple clusters (e.g. Person_1 and "
        "Person_5), select them here and merge."
    )

    cluster_options = [
        f"Person_{cid + 1}" if cid >= 0 else "Unclustered" for cid in unique_ids
    ]
    selected = st.multiselect(
        "Select clusters to merge",
        options=cluster_options,
        key="merge_select",
    )

    merge_map: dict[int, int] = {}
    if len(selected) >= 2 and st.button("🔗 Merge Selected Persons"):
        # Map all selected cluster IDs to the lowest one
        sel_ids = [unique_ids[cluster_options.index(s)] for s in selected]
        target = min(sel_ids)
        for sid in sel_ids:
            if sid != target:
                merge_map[sid] = target
        st.success(
            f"Merged {', '.join(selected)} → "
            f"{'Person_' + str(target + 1) if target >= 0 else 'Unclustered'}"
        )

    # ── Step 5: Build & Download ────────────────────────────────
    st.markdown("---")
    st.header("5 ∙ Build & Download")

    if st.button("📦 Build Organised ZIP", type="primary"):
        with st.spinner("Writing folders to disk…"):
            output_path, group_count = build_output_folders(
                uploaded_files,
                metadata,
                labels,
                custom_names=custom_names if custom_names else None,
                merge_map=merge_map if merge_map else None,
            )
            st.session_state["output_path"] = output_path

        st.toast("Folders created!", icon="✅")

    if "output_path" not in st.session_state:
        return

    output_path = st.session_state["output_path"]
    if not output_path.exists():
        return

    zip_buffer = create_zip_from_directory(output_path)
    st.download_button(
        label="📥 Download Organised Photos (.zip)",
        data=zip_buffer,
        file_name="organized_trip_photos.zip",
        mime="application/zip",
        type="primary",
    )

    # ── Step 6: Gallery Preview with EXIF ───────────────────────
    st.markdown("---")
    st.header("6 ∙ Gallery Preview")

    folders = sorted([f for f in output_path.iterdir() if f.is_dir()])
    st.caption(f"{len(folders)} folder(s) generated.")

    file_map = {f.name: f for f in uploaded_files}

    for folder in folders:
        files_in_dir = sorted(folder.glob("*"))
        with st.expander(
            f"📂 {folder.name}  ({len(files_in_dir)} image(s))", expanded=False
        ):
            for img_path in files_in_dir:
                col_img, col_meta = st.columns([2, 1])
                with col_img:
                    pil = load_image_from_path(img_path)
                    st.image(pil, caption=img_path.name, use_container_width=True)

                with col_meta:
                    st.markdown(f"**{img_path.name}**")
                    # Show EXIF if available
                    if img_path.name in file_map:
                        exif = extract_exif(file_map[img_path.name])
                        if exif:
                            ts = exif.get("DateTime", exif.get("DateTimeOriginal", "—"))
                            cam = exif.get("Model", exif.get("Make", "—"))
                            st.markdown(f"🕒 **Taken**: {ts}")
                            st.markdown(f"📷 **Camera**: {cam}")
                        else:
                            st.caption("No EXIF data available.")
                    else:
                        st.caption("No EXIF data available.")

                st.divider()

    gc.collect()


if __name__ == "__main__":
    main()