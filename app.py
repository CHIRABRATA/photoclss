import streamlit as st
from PIL import Image
from utils.detector import get_face_analyzer, detect_and_draw_faces

def load_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        return image, None
    except Exception as e:
        return None, str(e)


def main():
    st.set_page_config(
        page_title="Trip Photo Organizer AI", 
        page_icon="📸", 
        layout="wide"
    )

    st.title("📸 Trip Photo Organizer AI")
    st.caption("Automatically group trip photos by people using facial recognition.")
    st.markdown("---")

    st.header("1. Upload Trip Photos")
    uploaded_files = st.file_uploader(
        "Choose trip photos (JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Successfully loaded **{len(uploaded_files)}** photos!")
        
        st.markdown("---")
        st.header("2. Face Detection Preview")
        
        # Load and cache InsightFace model
        with st.spinner("Loading InsightFace detection model..."):
            analyzer = get_face_analyzer()

        cols = st.columns(4)
        total_faces_found = 0

        for idx, file in enumerate(uploaded_files):
            image, err = load_image(file)
            col = cols[idx % 4]
            
            if image is not None:
                # Detect faces & draw bounding boxes
                annotated_img, faces = detect_and_draw_faces(image, analyzer)
                num_faces = len(faces)
                total_faces_found += num_faces
                
                col.image(
                    annotated_img, 
                    caption=f"{file.name} ({num_faces} face{'s' if num_faces != 1 else ''})", 
                    use_container_width=True
                )
            else:
                col.error(f"Failed to load `{file.name}`")

        st.info(f"Total faces detected across all images: **{total_faces_found}**")
    else:
        st.info("Please upload your photos above to get started.")

if __name__ == "__main__":
    main()