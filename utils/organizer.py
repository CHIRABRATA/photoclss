import os
import shutil
import time
from pathlib import Path
from PIL import Image

def _safe_remove_dir(dir_path):
    """
    Safely removes directory contents without throwing PermissionError if files are locked.
    """
    if not dir_path.exists():
        return

    for path in dir_path.glob("**/*"):
        if path.is_file():
            try:
                path.unlink()
            except PermissionError:
                pass  # Skip locked files if currently being read

    try:
        shutil.rmtree(dir_path)
    except Exception:
        pass


def build_output_folders(uploaded_files, metadata_list, labels, output_base="output"):
    """
    Creates structured output folders on disk and copies photos into respective directories.
    Handles file locking safely on Windows.
    """
    base_path = Path(output_base)
    
    # Safely clear previous output folder
    _safe_remove_dir(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # 1. Map each file object by name
    file_map = {f.name: f for f in uploaded_files}
    
    # 2. Track which files belong to which person cluster
    cluster_files = {}  # {folder_name: set_of_filenames}
    file_face_counts = {f.name: 0 for f in uploaded_files}

    for label, meta in zip(labels, metadata_list):
        fname = meta["file_name"]
        file_face_counts[fname] = file_face_counts.get(fname, 0) + 1
        
        folder_name = f"Person_{label + 1}" if label >= 0 else "Single_Unclustered"
        
        if folder_name not in cluster_files:
            cluster_files[folder_name] = set()
        cluster_files[folder_name].add(fname)

    # Helper function to save images cleanly and release file handles
    def save_image_to_dir(file_obj, dest_path):
        file_obj.seek(0)
        with Image.open(file_obj) as img:
            img.save(dest_path)

    # 3. Copy files to person folders
    for folder_name, filenames in cluster_files.items():
        folder_path = base_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        for fname in filenames:
            save_image_to_dir(file_map[fname], folder_path / fname)

    # 4. Save photos with 2+ faces to Group_Photos folder
    group_dir = base_path / "Group_Photos"
    group_count = 0
    for fname, count in file_face_counts.items():
        if count >= 2:
            group_dir.mkdir(parents=True, exist_ok=True)
            save_image_to_dir(file_map[fname], group_dir / fname)
            group_count += 1

    return base_path, group_count