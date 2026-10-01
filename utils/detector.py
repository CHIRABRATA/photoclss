"""
utils/detector.py — Memory-optimised face detection for Streamlit Community Cloud.

Key design decisions for 1 GB RAM ceiling:
  • buffalo_s model (~30 MB) with det_size=(320,320) keeps peak allocation <200 MB.
  • Every uploaded image is decoded→downscaled→processed→discarded in a streaming
    fashion so at most ONE image array lives in memory at any time.
  • Torch/YOLO/ResNet50-ReID are completely removed — they were the main OOM source.
  • gc.collect() is called per-image to return freed numpy buffers to the OS promptly.
"""

import gc
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from insightface.app import FaceAnalysis


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_PHOTOS = 15          # Hard cap for Community Cloud (1 GB RAM)
_DET_SIZE = (320, 320)   # Smaller ONNX input = less VRAM/RAM
_MAX_DIM = 1080          # Longest edge after downscale


# ─────────────────────────────────────────────
# Model loader (cached once across reruns)
# ─────────────────────────────────────────────
@st.cache_resource
def get_face_analyzer():
    """Load InsightFace buffalo_s on CPU with a small detection size."""
    app = FaceAnalysis(
        name="buffalo_s",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=_DET_SIZE)
    return app


# ─────────────────────────────────────────────
# Image pre-processing
# ─────────────────────────────────────────────
def preprocess_image(file_bytes: bytes, max_dim: int = _MAX_DIM) -> np.ndarray:
    """Decode raw upload bytes → BGR ndarray, downscaled so longest edge ≤ max_dim.

    Using cv2.imdecode avoids an intermediate PIL copy.  The resize is done
    in-place so the original full-res buffer can be freed immediately.
    """
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode returned None — corrupt or unsupported image")

    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img_bgr


# ─────────────────────────────────────────────
# Detection + annotation (for preview grid)
# ─────────────────────────────────────────────
def detect_and_draw_faces(pil_image, face_analyzer, _body_detector=None):
    """Return (annotated_PIL, faces_list, []) for backward-compat with app.py.

    The third element (body_boxes) is always an empty list now that YOLO has been
    removed.  Keeping the signature avoids breaking the caller.
    """
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Downscale for detection (keeps annotation on the downscaled copy)
    h, w = img_bgr.shape[:2]
    if max(h, w) > _MAX_DIM:
        scale = _MAX_DIM / max(h, w)
        img_bgr = cv2.resize(
            img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    faces = face_analyzer.get(img_bgr)

    annotated = img_bgr.copy()
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), faces, []


# ─────────────────────────────────────────────
# Embedding extraction (streaming, one-at-a-time)
# ─────────────────────────────────────────────
def extract_embeddings_from_files(uploaded_files, face_analyzer, _body_detector=None):
    """Extract pure 512-D face embeddings for every detected face.

    Returns
    -------
    embeddings : np.ndarray of shape (N, 512)
    metadata   : list[dict] with keys file_name, file_obj, bbox, type, face_count
    """
    embeddings: list[np.ndarray] = []
    metadata: list[dict] = []

    for file in uploaded_files:
        try:
            file.seek(0)
            raw_bytes = file.read()
            img_bgr = preprocess_image(raw_bytes, max_dim=_MAX_DIM)
            del raw_bytes  # free the upload buffer immediately

            faces = face_analyzer.get(img_bgr)
            face_count = len(faces)

            if face_count > 0:
                for face in faces:
                    if hasattr(face, "embedding") and face.embedding is not None:
                        embeddings.append(face.embedding.copy())
                        metadata.append({
                            "file_name": file.name,
                            "file_obj": file,
                            "bbox": face.bbox.astype(int),
                            "type": "face",
                            "face_count": face_count,
                        })
            else:
                # No face detected — still record so organizer can route to
                # Landscapes_or_NoFace later.
                metadata.append({
                    "file_name": file.name,
                    "file_obj": file,
                    "bbox": None,
                    "type": "no_face",
                    "face_count": 0,
                })

            # Aggressively free the image array before the next iteration
            del img_bgr, faces
            gc.collect()

        except Exception as e:
            st.error(f"⚠️ Error processing **{file.name}**: {e}")

    return np.array(embeddings) if embeddings else np.empty((0, 512)), metadata