import streamlit as st
from PIL import Image
from utils.detector import get_face_analyzer, detect_and_draw_faces, extract_embeddings_from_files
from utils.cluster import cluster_face_embeddings
from utils.organizer import build_output_folders
from utils.zipper import create_zip_from_directory

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
        
        with st.spinner("Loading InsightFace detection model..."):
            analyzer = get_face_analyzer()

        cols = st.columns(4)
        total_faces_found = 0

        for idx, file in enumerate(uploaded_files):
            image, err = load_image(file)
            col = cols[idx % 4]
            
            if image is not None:
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

        st.markdown("---")
        st.header("3. Face Clustering & Folder Generation")
        
        col1, col2 = st.columns(2)
        with col1:
            eps_val = st.slider("Clustering Distance Threshold (eps)", 0.20, 0.70, 0.50, 0.05)
        with col2:
            min_samples_val = st.number_input("Minimum Photos Per Person (min_samples)", 1, 10, 1)

        if st.button("Organize Photos into Folders"):
            with st.spinner("Extracting embeddings, clustering, and organizing disk folders..."):
                embeddings, metadata = extract_embeddings_from_files(uploaded_files, analyzer)
                
                if len(embeddings) > 0:
                    labels = cluster_face_embeddings(embeddings, eps=eps_val, min_samples=min_samples_val)
                    output_path, group_count = build_output_folders(uploaded_files, metadata, labels)
                    
                    st.success(f"Successfully generated folder hierarchy at `{output_path}`!")
                    
                    # Store generated path in session state
                    st.session_state["output_path"] = output_path
                else:
                    st.warning("No faces detected across uploaded images.")

        # Show Output & Download section if folders have been created
        if "output_path" in st.session_state and st.session_state["output_path"].exists():
            output_path = st.session_state["output_path"]
            
            st.markdown("---")
            st.header("4. Download Organized ZIP")
            
            # Prepare ZIP buffer
            zip_buffer = create_zip_from_directory(output_path)
            
            st.download_button(
                label="📦 Download Organized Photos (.zip)",
                data=zip_buffer,
                file_name="organized_trip_photos.zip",
                mime="application/zip",
                type="primary"
            )

            st.subheader("📁 Output Directory Structure:")
            for folder in sorted(output_path.iterdir()):
                if folder.is_dir():
                    files_in_dir = list(folder.glob("*"))
                    with st.expander(f"📂 {folder.name} ({len(files_in_dir)} image(s))", expanded=False):
                        img_cols = st.columns(4)
                        for idx, img_path in enumerate(files_in_dir):
                            img = Image.open(img_path)
                            col = img_cols[idx % 4]
                            col.image(img, caption=img_path.name, use_container_width=True)

    else:
        st.info("Please upload your photos above to get started.")

if __name__ == "__main__":
    main()