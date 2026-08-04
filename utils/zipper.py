import os
import io
import zipfile
from pathlib import Path

def create_zip_from_directory(dir_path):
    """
    Compresses all folders and files inside `dir_path` into an in-memory ZIP buffer.
    Returns:
      - zip_buffer (BytesIO): binary buffer ready for st.download_button
    """
    dir_path = Path(dir_path)
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in dir_path.glob("**/*"):
            if file_path.is_file():
                # Store relative path inside the zip archive
                archive_name = file_path.relative_to(dir_path)
                zip_file.write(file_path, archive_name)

    zip_buffer.seek(0)
    return zip_buffer