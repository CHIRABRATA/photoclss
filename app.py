import time
import streamlit as st
from PIL import Image
from utils.detector import get_face_analyzer, get_body_detector, detect_and_draw_faces, extract_embeddings_from_files
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
        
        with st.spinner("Initializing Face & Body detection engines..."):
            analyzer = get_face_analyzer()
            body_detector = get_body_detector()

        cols = st.columns(4)
        total_faces_found = 0

        # Progress bar for image scanning
        scan_bar = st.progress(0, text="Scanning uploaded images...")
        num_files = len(uploaded_files)

        for idx, file in enumerate(uploaded_files):
            # Update progress
            scan_bar.progress((idx + 1) / num_files, text=f"Analyzing image {idx + 1} of {num_files}: `{file.name}`")
            
            image, err = load_image(file)
            col = cols[idx % 4]
            
            if image is not None:
                try:
                    annotated_img, faces, bodies = detect_and_draw_faces(image, analyzer, body_detector)
                    num_faces = len(faces)
                    total_faces_found += num_faces
                    
                    col.image(
                        annotated_img, 
                        caption=f"{file.name} ({num_faces} face{'s' if num_faces != 1 else ''})", 
                        use_container_width=True
                    )
                except Exception as e:
                    col.error(f"Error analyzing faces in `{file.name}`: {e}")
            else:
                col.error(f"Failed to load `{file.name}`")

        scan_bar.empty()  # Clear progress bar after completion
        st.info(f"Scan complete! Detected **{total_faces_found}** face(s) across **{len(uploaded_files)}** photo(s).")

        st.markdown("---")
        st.header("3. Clustering Settings & Organization")
        
        col1, col2 = st.columns(2)
        with col1:
            eps_val = st.slider(
                "Clustering Distance Threshold (eps)", 
                0.20, 0.70, 0.50, 0.05,
                help="Lower value = strict face matching. Higher value = tolerates different angles/lighting."
            )
        with col2:
            min_samples_val = st.number_input(
                "Minimum Photos Per Person (min_samples)", 
                1, 10, 1,
                help="Minimum face detections required to form a dedicated person folder."
            )

        if st.button("🚀 Process & Organize Photos", type="primary"):
            progress_bar = st.progress(0, text="Starting pipeline...")
            
            try:
                # Step A: Embeddings
                progress_bar.progress(25, text="Extracting 512-D face embeddings...")
                embeddings, metadata = extract_embeddings_from_files(uploaded_files, analyzer)
                
                if len(embeddings) > 0:
                    # Step B: Clustering
                    progress_bar.progress(60, text="Clustering faces with DBSCAN...")
                    labels = cluster_face_embeddings(embeddings, eps=eps_val, min_samples=min_samples_val)
                    
                    # Step C: Disk folder creation
                    progress_bar.progress(85, text="Creating person directories on disk...")
                    output_path, group_count = build_output_folders(uploaded_files, metadata, labels)
                    
                    progress_bar.progress(100, text="Organization complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    st.toast("Photos organized successfully!", icon="🎉")
                    st.session_state["output_path"] = output_path
                else:
                    progress_bar.empty()
                    st.warning("No faces detected across uploaded images.")
            except Exception as e:
                progress_bar.empty()
                st.error(f"An unexpected error occurred during processing: {e}")

        # Section 4: Download & Summary Preview
        if "output_path" in st.session_state and st.session_state["output_path"].exists():
            output_path = st.session_state["output_path"]
            
            st.markdown("---")
            st.header("4. Download Organized ZIP")
            
            zip_buffer = create_zip_from_directory(output_path)
            
            st.download_button(
                label="📦 Download Organized Photos (.zip)",
                data=zip_buffer,
                file_name="organized_trip_photos.zip",
                mime="application/zip",
                type="primary"
            )

            st.subheader("📁 Output Directory Summary:")
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