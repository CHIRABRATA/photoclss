"""
utils/organizer.py — Smart directory builder with three category tiers.

Output structure:
    output/
    ├── Person_1/          # (or user-renamed, e.g. "Alex")
    ├── Person_2/
    ├── Group_Photos/      # images with 2+ detected faces
    └── Landscapes_or_NoFace/  # images with 0 detected faces (scenery, etc.)

Design notes:
  • Group_Photos receives a *copy* of multi-face images — the originals still
    appear in their Person_N folders so no photo is lost.
  • Landscapes_or_NoFace catches every upload that had zero face detections.
  • EXIF extraction is done here once per file; the caller can display it.
"""

import shutil
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS


# ─────────────────────────────────────────────
# EXIF helper
# ─────────────────────────────────────────────
def extract_exif(file_obj) -> dict:
    """Return a dict of human-readable EXIF tags from an uploaded file.

    Returns an empty dict on failure so the caller never needs to guard.
    """
    try:
        file_obj.seek(0)
        with Image.open(file_obj) as img:
            raw = img.getexif()
            if not raw:
                return {}
            return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return {}


# ─────────────────────────────────────────────
# Safe directory removal (Windows-friendly)
# ─────────────────────────────────────────────
def _safe_remove_dir(dir_path: Path) -> None:
    """Remove a directory tree, silently skipping locked files."""
    if not dir_path.exists():
        return
    for p in dir_path.glob("**/*"):
        if p.is_file():
            try:
                p.unlink()
            except PermissionError:
                pass
    try:
        shutil.rmtree(dir_path)
    except Exception:
        pass


# ─────────────────────────────────────────────
# Image writer (detaches PIL handle cleanly)
# ─────────────────────────────────────────────
def _save_image(file_obj, dest_path: Path) -> None:
    """Read the upload once, save to disk, then release the handle."""
    file_obj.seek(0)
    with Image.open(file_obj) as img:
        img.save(dest_path)


# ─────────────────────────────────────────────
# Main folder builder
# ─────────────────────────────────────────────
def build_output_folders(
    uploaded_files,
    metadata_list: list[dict],
    labels,
    output_base: str = "output",
    custom_names: dict[int, str] | None = None,
    merge_map: dict[int, int] | None = None,
):
    """Create the organised output tree on disk.

    Parameters
    ----------
    uploaded_files : list
        Streamlit UploadedFile objects.
    metadata_list : list[dict]
        From detector.extract_embeddings_from_files.
    labels : ndarray
        Cluster labels (only for entries where type == "face").
    output_base : str
        Root directory name.
    custom_names : dict[int, str] | None
        Mapping of cluster_id → user-supplied name (e.g. {0: "Alex"}).
    merge_map : dict[int, int] | None
        Mapping of cluster_id → target_cluster_id for merged clusters.

    Returns
    -------
    (output_path, group_count) : tuple[Path, int]
    """
    base_path = Path(output_base)
    _safe_remove_dir(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    file_map = {f.name: f for f in uploaded_files}
    processed_filenames: set[str] = set()

    # ── 1. Build cluster → filenames map ──────────────────────────
    cluster_files: dict[int, set[str]] = {}
    file_face_counts: dict[str, int] = {}

    label_idx = 0  # labels only cover "face"-type metadata entries
    for meta in metadata_list:
        fname = meta["file_name"]
        if meta["type"] == "face":
            raw_label = int(labels[label_idx])
            label_idx += 1

            # Apply merge remapping if provided
            if merge_map and raw_label in merge_map:
                raw_label = merge_map[raw_label]

            processed_filenames.add(fname)
            file_face_counts[fname] = file_face_counts.get(fname, 0) + 1

            cluster_files.setdefault(raw_label, set()).add(fname)

        elif meta["type"] == "no_face":
            # Will be routed to Landscapes_or_NoFace below
            processed_filenames.add(fname)

    # ── 2. Write person folders ────────────────────────────────────
    for cluster_id, filenames in cluster_files.items():
        if custom_names and cluster_id in custom_names:
            folder_name = custom_names[cluster_id]
        else:
            folder_name = f"Person_{cluster_id + 1}" if cluster_id >= 0 else "Unclustered"

        folder_path = base_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        for fname in filenames:
            _save_image(file_map[fname], folder_path / fname)

    # ── 3. Group_Photos (images with 2+ faces) ────────────────────
    group_dir = base_path / "Group_Photos"
    group_count = 0
    for fname, count in file_face_counts.items():
        if count >= 2:
            group_dir.mkdir(parents=True, exist_ok=True)
            _save_image(file_map[fname], group_dir / fname)
            group_count += 1

    # ── 4. Landscapes_or_NoFace (0 detections) ────────────────────
    landscape_dir = base_path / "Landscapes_or_NoFace"
    no_face_files = {
        m["file_name"]
        for m in metadata_list
        if m["type"] == "no_face"
    }
    for fname in no_face_files:
        landscape_dir.mkdir(parents=True, exist_ok=True)
        _save_image(file_map[fname], landscape_dir / fname)

    # ── 5. Catch-all for any file missed entirely ─────────────────
    unclassified_dir = base_path / "Unclassified_Photos"
    all_uploaded_names = {f.name for f in uploaded_files}
    for fname in all_uploaded_names - processed_filenames:
        unclassified_dir.mkdir(parents=True, exist_ok=True)
        _save_image(file_map[fname], unclassified_dir / fname)

    return base_path, group_count