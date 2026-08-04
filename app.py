import time
import streamlit as st
from PIL import Image
from utils.detector import get_face_analyzer, get_body_detector, detect_and_draw_faces, extract_embeddings_from_files
from utils.cluster import cluster_face_embeddings
from utils.organizer import build_output_folders
from utils.zipper import create_zip_from_directory
import time
import streamlit as st
from PIL import Image
from utils.detector import (
    get_face_analyzer, 
    get_body_detector, 
    detect_and_draw_faces, 
    extract_embeddings_from_files
)
from utils.cluster import cluster_face_embeddings
from utils.organizer import build_output_folders
from utils.zipper import create_zip_from_directory


# Above this many photos, the grid switches to a fixed-height scroll
# container instead of growing the page indefinitely.
SCROLL_THRESHOLD = 12
GRID_HEIGHT = 560
 
 
def load_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        return image, None
    except Exception as e:
        return None, str(e)
 
 
def render_photo_grid(items, cols_count=4, height=None):
    """Render (image, caption) tuples in a responsive grid.
    Wraps in a scrollable bordered container when `height` is set,
    so 100 photos don't turn the page into an endless scroll,
    while small batches (no height) just render naturally."""
    if height is None:
        container = st.container(border=False)
    else:
        container = st.container(height=height, border=True)
    with container:
        cols = st.columns(cols_count)
        for idx, (img, caption, ok) in enumerate(items):
            col = cols[idx % cols_count]
            if ok:
                col.image(img, caption=caption, use_container_width=True)
            else:
                col.error(caption)
 
 
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
        num_files = len(uploaded_files)
        st.success(f"Successfully loaded **{num_files}** photos!")
 
        st.markdown("---")
        st.header("2. Face Detection Preview")
 
        # Cache detection results across reruns (slider tweaks, button clicks)
        # so we don't redo face detection on every interaction — this matters
        # a lot once you're at 100 photos.
        cache_key = tuple(sorted(f.file_id for f in uploaded_files)) if hasattr(uploaded_files[0], "file_id") else tuple(f.name for f in uploaded_files)
        if st.session_state.get("preview_cache_key") != cache_key:
            with st.spinner("Initializing Face & Body detection engines..."):
                analyzer = get_face_analyzer()
                body_detector = get_body_detector()
 
            grid_items = []
            total_faces_found = 0
            scan_bar = st.progress(0, text="Scanning uploaded images...")
 
            for idx, file in enumerate(uploaded_files):
                scan_bar.progress((idx + 1) / num_files, text=f"Analyzing image {idx + 1} of {num_files}: `{file.name}`")
                image, err = load_image(file)
 
                if image is not None:
                    try:
                        annotated_img, faces, bodies = detect_and_draw_faces(image, analyzer, body_detector)
                        num_faces = len(faces)
                        total_faces_found += num_faces
                        caption = f"{file.name} ({num_faces} face{'s' if num_faces != 1 else ''})"
                        grid_items.append((annotated_img, caption, True))
                    except Exception as e:
                        grid_items.append((None, f"Error analyzing `{file.name}`: {e}", False))
                else:
                    grid_items.append((None, f"Failed to load `{file.name}`", False))
 
            scan_bar.empty()
            st.session_state["preview_cache_key"] = cache_key
            st.session_state["grid_items"] = grid_items
            st.session_state["total_faces_found"] = total_faces_found
            st.session_state["analyzer"] = analyzer
            st.session_state["body_detector"] = body_detector
 
        grid_items = st.session_state["grid_items"]
        total_faces_found = st.session_state["total_faces_found"]
        analyzer = st.session_state["analyzer"]
        body_detector = st.session_state["body_detector"]
 
        # Metrics dashboard instead of a single info line
        m1, m2, m3 = st.columns(3)
        m1.metric("Photos Uploaded", num_files)
        m2.metric("Faces Detected", total_faces_found)
        m3.metric("Avg Faces / Photo", round(total_faces_found / num_files, 2) if num_files else 0)
 
        # Only scroll-contain the grid once there's enough content to warrant it
        use_scroll = num_files > SCROLL_THRESHOLD
        if use_scroll:
            st.caption(f"Showing all {num_files} photos below — scroll within the box.")
            render_photo_grid(grid_items, cols_count=4, height=GRID_HEIGHT)
        else:
            render_photo_grid(grid_items, cols_count=4, height=None)
 
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
                progress_bar.progress(25, text="Extracting 512-D face embeddings...")
                embeddings, metadata = extract_embeddings_from_files(uploaded_files, analyzer)
 
                if len(embeddings) > 0:
                    progress_bar.progress(60, text="Clustering faces with DBSCAN...")
                    labels = cluster_face_embeddings(embeddings, eps=eps_val, min_samples=min_samples_val)
 
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
            folders = sorted([f for f in output_path.iterdir() if f.is_dir()])
            st.caption(f"{len(folders)} person folder(s) found.")
 
            for folder in folders:
                files_in_dir = sorted(folder.glob("*"))
                with st.expander(f"📂 {folder.name} ({len(files_in_dir)} image(s))", expanded=False):
                    thumb_items = [(Image.open(p), p.name, True) for p in files_in_dir]
                    # Same adaptive-scroll logic applied to each folder
                    if len(thumb_items) > SCROLL_THRESHOLD:
                        render_photo_grid(thumb_items, cols_count=4, height=420)
                    else:
                        render_photo_grid(thumb_items, cols_count=4, height=None)
 
    else:
        st.info("Please upload your photos above to get started.")
 
 
if __name__ == "__main__":
    main()
 