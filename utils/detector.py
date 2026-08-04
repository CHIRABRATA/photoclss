import cv2
import numpy as np
import streamlit as st
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from utils.reid import get_reid_extractor, extract_body_embedding


@st.cache_resource
def get_face_analyzer():
    """
    Initialize and cache InsightFace FaceAnalysis.
    """
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


@st.cache_resource
def get_body_detector():
    """
    Initialize and cache YOLOv8 for human body detection (front & back views).
    """
    return YOLO("yolov8n.pt")


def detect_and_draw_faces(pil_image, face_analyzer, body_detector=None):
    """
    Detects faces (Green) and full bodies (Blue) for UI visual preview.
    """
    img_array = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1. Detect faces
    faces = face_analyzer.get(img_bgr)
    
    # 2. Detect bodies (Class 0: Person)
    body_boxes = []
    if body_detector is not None:
        results = body_detector(img_array, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                    body_boxes.append(box.xyxy[0].cpu().numpy().astype(int))

    annotated_bgr = img_bgr.copy()
    
    # Draw body boxes (Blue)
    for bbox in body_boxes:
        cv2.rectangle(annotated_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    # Draw face boxes (Green)
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(annotated_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), faces, body_boxes


def extract_embeddings_from_files(uploaded_files, face_analyzer, body_detector=None):
    """
    Extracts face embeddings (512-D) and/or body embeddings (2048-D) for each photo.
    Handles back shots where no face is visible.
    """
    embeddings = []
    face_metadata = []
    reid_model = get_reid_extractor() if body_detector else None

    for file in uploaded_files:
        try:
            file.seek(0)
            pil_image = Image.open(file)
            img_array = np.array(pil_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = face_analyzer.get(img_bgr)
            
            # Detect human bodies (front + back)
            body_boxes = []
            if body_detector:
                results = body_detector(img_array, verbose=False)
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                            body_boxes.append(box.xyxy[0].cpu().numpy().astype(int))

            # Case A: Faces detected
            if len(faces) > 0:
                for face in faces:
                    if hasattr(face, 'embedding') and face.embedding is not None:
                        norm_face_emb = face.embedding / np.linalg.norm(face.embedding)
                        
                        # Find matching body crop to combine body/clothing context
                        body_emb = None
                        if reid_model and len(body_boxes) > 0:
                            body_emb = extract_body_embedding(pil_image, body_boxes[0], reid_model)

                        if body_emb is not None:
                            # Combine Face (512-D) + Body (2048-D) into a unified vector
                            combined = np.hstack([norm_face_emb, body_emb * 0.5])
                            combined = combined / np.linalg.norm(combined)
                            embeddings.append(combined)
                        else:
                            # Pad with zero vector so shape matches
                            padded = np.hstack([norm_face_emb, np.zeros(2048)])
                            padded = padded / np.linalg.norm(padded)
                            embeddings.append(padded)

                        face_metadata.append({
                            "file_name": file.name,
                            "file_obj": file,
                            "bbox": face.bbox.astype(int),
                            "type": "face"
                        })

            # Case B: Back view / No face visible, but body detected
            elif len(body_boxes) > 0 and reid_model:
                for bbox in body_boxes:
                    body_emb = extract_body_embedding(pil_image, bbox, reid_model)
                    if body_emb is not None:
                        # Dummy face vector so matrix dimensions match
                        dummy_face = np.zeros(512)
                        combined = np.hstack([dummy_face, body_emb * 0.5])
                        combined = combined / np.linalg.norm(combined)
                        
                        embeddings.append(combined)
                        face_metadata.append({
                            "file_name": file.name,
                            "file_obj": file,
                            "bbox": bbox,
                            "type": "body_back_view"
                        })

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    return np.array(embeddings), face_metadata