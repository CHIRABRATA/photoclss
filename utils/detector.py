import numpy as np
import cv2
import streamlit as st
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

@st.cache_resource
def get_face_analyzer():
    """
    Initialize and cache the InsightFace FaceAnalysis engine.
    Uses 'buffalo_s' for fast, low-memory CPU execution.
    """
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def detect_and_draw_faces(pil_image, face_analyzer):
    """
    Detects faces in a PIL image and returns:
    1. OpenCV image (RGB) with bounding boxes drawn
    2. List of detected face objects from InsightFace
    """
    img_array = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    faces = face_analyzer.get(img_bgr)
    
    annotated_bgr = img_bgr.copy()
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(
            annotated_bgr, 
            (bbox[0], bbox[1]), 
            (bbox[2], bbox[3]), 
            (0, 255, 0), 
            2
        )
    
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), faces


def extract_embeddings_from_files(uploaded_files, face_analyzer):
    """
    Scans all uploaded files and extracts face embeddings.
    Returns:
      - embeddings: list of 512-D numpy vectors
      - face_metadata: list of dicts mapping each embedding to its original filename & face bounding box
    """
    embeddings = []
    face_metadata = []

    for file in uploaded_files:
        try:
            file.seek(0)
            pil_image = Image.open(file)
            img_array = np.array(pil_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            faces = face_analyzer.get(img_bgr)
            
            for face in faces:
                if hasattr(face, 'embedding') and face.embedding is not None:
                    # L2 Normalization ensures cosine similarity equals inner product
                    norm_embedding = face.embedding / np.linalg.norm(face.embedding)
                    embeddings.append(norm_embedding)
                    face_metadata.append({
                        "file_name": file.name,
                        "file_obj": file,
                        "bbox": face.bbox.astype(int),
                        "det_score": face.det_score
                    })
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    return np.array(embeddings), face_metadata