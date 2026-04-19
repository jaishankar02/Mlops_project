"""
Streamlit frontend for StyleSync Recommender.
Phase 1: Recommender only
"""
import streamlit as st
import requests
from PIL import Image
import io
from typing import List, Tuple
import logging
import os
import base64
import time
import json

# Configure page
st.set_page_config(
    page_title="StyleSync - Fashion Recommender",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
    }
    .similarity-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8001")


def _normalize_image_for_backend(
    image: Image.Image,
    min_dim: int = 128,
    max_dim: int = 1800,
    max_pixels: int = 3_200_000,
    pad_color: tuple = (255, 255, 255),
) -> Image.Image:
    """Normalize image dimensions to satisfy backend validation constraints."""
    img = image.convert("RGB")
    w, h = img.size

    # Upscale tiny dimensions by padding (avoid distortion).
    target_w = max(w, min_dim)
    target_h = max(h, min_dim)
    if target_w != w or target_h != h:
        canvas = Image.new("RGB", (target_w, target_h), pad_color)
        x = (target_w - w) // 2
        y = (target_h - h) // 2
        canvas.paste(img, (x, y))
        img = canvas
        w, h = img.size

    # Limit max side.
    if max(w, h) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        w, h = img.size

    # Limit total pixel count (backend estimates size from dimensions).
    if w * h > max_pixels:
        scale = (max_pixels / float(w * h)) ** 0.5
        new_w = max(min_dim, int(w * scale))
        new_h = max(min_dim, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def _is_backend_image_valid(image: Image.Image, max_size_mb: float = 10.0) -> bool:
    """Mirror backend validation logic to avoid request-time failures."""
    w, h = image.size
    if w < 100 or h < 100:
        return False
    estimated_size_mb = (w * h * 3) / (1024 * 1024)
    return estimated_size_mb <= max_size_mb


def _ensure_backend_safe_bytes(
    image_bytes: bytes,
    output_format: str = "PNG",
    pad_color: tuple = (255, 255, 255),
) -> bytes:
    """Aggressively normalize image until backend validation constraints are met."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # First pass.
    img = _normalize_image_for_backend(img, min_dim=128, max_dim=1400, max_pixels=2_400_000, pad_color=pad_color)

    # If still invalid, force conservative dimensions.
    if not _is_backend_image_valid(img):
        img = _normalize_image_for_backend(img, min_dim=128, max_dim=1024, max_pixels=1_800_000, pad_color=pad_color)

    out = io.BytesIO()
    if output_format.upper() == "JPEG":
        img.save(out, format="JPEG", quality=88, optimize=True)
    else:
        img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _is_backend_bytes_valid(image_bytes: bytes) -> bool:
    """Validate encoded image bytes against backend size constraints."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return False
    return _is_backend_image_valid(img)


def _prepare_person_upload_bytes(person_bytes: bytes) -> bytes:
    """Prepare person image bytes for backend-side preprocessing and validation."""
    try:
        img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    except Exception:
        img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    safe = _ensure_backend_safe_bytes(out.getvalue(), output_format="PNG", pad_color=(255, 255, 255))
    return safe


def _prepare_recommender_upload_bytes(image_bytes: bytes, filename: str) -> tuple[bytes, str, str]:
    """Normalize recommender query/upload images so backend validation always passes."""
    lower = filename.lower()
    requested_png = lower.endswith(".png")
    out_format = "PNG" if requested_png else "JPEG"
    mime = "image/png" if requested_png else "image/jpeg"

    safe = _ensure_backend_safe_bytes(
        image_bytes,
        output_format=out_format,
        pad_color=(255, 255, 255),
    )

    if out_format == "JPEG" and not filename.lower().endswith(".jpg") and not filename.lower().endswith(".jpeg"):
        filename = f"{os.path.splitext(filename)[0]}.jpg"
    if out_format == "PNG" and not filename.lower().endswith(".png"):
        filename = f"{os.path.splitext(filename)[0]}.png"

    return safe, filename, mime


def _compress_garment_for_upload(
    garment_bytes: bytes,
    filename: str,
    max_size_mb: float = 9.5,
    max_dim: int = 2048,
) -> Tuple[bytes, str, str, bool]:
    """Compress garment image when needed to satisfy backend size validation.

    Returns: (content_bytes, output_filename, mime_type, was_compressed)
    """
    if len(garment_bytes) <= int(max_size_mb * 1024 * 1024):
        try:
            img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
            out = io.BytesIO()
            if filename.lower().endswith(".png"):
                img.save(out, format="PNG", optimize=True)
                safe = _ensure_backend_safe_bytes(out.getvalue(), output_format="PNG", pad_color=(255, 255, 255))
                return safe, filename, "image/png", False
            img.save(out, format="JPEG", quality=92, optimize=True)
            safe = _ensure_backend_safe_bytes(out.getvalue(), output_format="JPEG", pad_color=(255, 255, 255))
            return safe, filename, "image/jpeg", False
        except Exception:
            lower = filename.lower()
            mime = "image/png" if lower.endswith(".png") else "image/jpeg"
            return garment_bytes, filename, mime, False

    try:
        img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        img = _normalize_image_for_backend(img, min_dim=128, max_dim=max_dim, max_pixels=2_400_000)

        qualities = [90, 82, 74, 66, 58, 50, 42]
        scale = 1.0
        best = None

        for _ in range(4):
            if scale < 1.0:
                w, h = img.size
                new_w = max(256, int(w * scale))
                new_h = max(256, int(h * scale))
                trial_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                trial_img = img

            for q in qualities:
                out = io.BytesIO()
                trial_img.save(out, format="JPEG", quality=q, optimize=True)
                data = out.getvalue()
                best = data
                if len(data) <= int(max_size_mb * 1024 * 1024):
                    out_name = f"{os.path.splitext(filename)[0]}_compressed.jpg"
                    safe = _ensure_backend_safe_bytes(data, output_format="JPEG", pad_color=(255, 255, 255))
                    return safe, out_name, "image/jpeg", True

            scale *= 0.85

        # Return smallest attempt even if still large.
        out_name = f"{os.path.splitext(filename)[0]}_compressed.jpg"
        payload = best if best is not None else garment_bytes
        safe = _ensure_backend_safe_bytes(payload, output_format="JPEG", pad_color=(255, 255, 255))
        return safe, out_name, "image/jpeg", True
    except Exception:
        lower = filename.lower()
        mime = "image/png" if lower.endswith(".png") else "image/jpeg"
        return garment_bytes, filename, mime, False

def make_api_call(endpoint: str, method: str = "GET", files=None, params=None, api_prefix: str = "recommender"):
    """Make API call with error handling."""
    try:
        url = f"{API_URL}/api/{api_prefix}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, params=params, timeout=60)
            else:
                response = requests.post(url, json=params, timeout=30)
        
        response.raise_for_status()
        return response
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Is it running?")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        detail_msg = None
        response = getattr(e, "response", None)
        if response is not None:
            try:
                payload = response.json()
                detail_msg = payload.get("detail") if isinstance(payload, dict) else None
            except Exception:
                detail_msg = None

        if detail_msg:
            st.error(f"❌ API Error: {detail_msg}")
        else:
            st.error(f"❌ API Error: {str(e)}")
        return None


def upload_garment_ui():
    """UI for uploading a single garment."""
    st.subheader("📤 Upload Garment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a garment image",
            type=["jpg", "jpeg", "png"],
            key="single_upload"
        )
    
    with col2:
        st.markdown("**Additional Info (Optional)**")
        category = st.text_input("Category", placeholder="e.g., shirt, dress, pants")
        color = st.text_input("Color", placeholder="e.g., red, blue")
        size = st.selectbox("Size", ["", "XS", "S", "M", "L", "XL", "XXL"])
        price = st.number_input("Price ($)", min_value=0.0, step=0.01)
    
    if uploaded_file and st.button("🚀 Upload Garment", key="upload_btn"):
        with st.spinner("Uploading garment..."):
            safe_bytes, safe_name, safe_mime = _prepare_recommender_upload_bytes(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )
            files = {"file": (safe_name, safe_bytes, safe_mime)}
            params = {
                "category": category if category else None,
                "color": color if color else None,
                "size": size if size else None,
                "price": price if price > 0 else None,
            }
            
            response = make_api_call("/upload-garment", method="POST", files=files, params=params)
            
            if response:
                data = response.json()
                st.success(f"✅ Garment uploaded successfully!")
                st.info(f"**Garment ID:** `{data['garment_id']}`")
                st.balloons()


def bulk_upload_ui():
    """UI for bulk uploading garments."""
    st.subheader("📦 Bulk Upload")
    
    uploaded_files = st.file_uploader(
        "Choose multiple garment images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="bulk_upload"
    )
    
    if uploaded_files and st.button("🚀 Upload All", key="bulk_upload_btn"):
        with st.spinner(f"Uploading {len(uploaded_files)} garments..."):
            files = []
            for f in uploaded_files:
                safe_bytes, safe_name, safe_mime = _prepare_recommender_upload_bytes(
                    f.getvalue(),
                    f.name,
                )
                files.append(("files", (safe_name, safe_bytes, safe_mime)))
            response = make_api_call("/bulk-upload", method="POST", files=files)
            
            if response:
                data = response.json()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Successful", data["successful"])
                with col2:
                    st.metric("❌ Failed", data["failed"])
                with col3:
                    st.metric("📊 Total", data["total_uploaded"])
                
                if data["failed"] > 0:
                    st.warning("Some uploads failed:")
                    for item in data["failed_items"]:
                        st.text(f"- {item['filename']}: {item['error']}")
                
                if data["successful"] > 0:
                    st.balloons()


def search_recommender_ui():
    """UI for searching similar garments."""
    st.subheader("🔍 Search Similar Garments")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query_file = st.file_uploader(
            "Upload a garment to find similar items",
            type=["jpg", "jpeg", "png"],
            key="search_file"
        )
    
    with col2:
        st.markdown("**Options**")
        k = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
    
    if query_file:
        # Display query image
        query_image = Image.open(query_file)
        st.image(query_image, caption="Query Garment", width=200)
        
        if st.button("🔎 Find Similar", key="search_btn"):
            with st.spinner("Searching..."):
                safe_bytes, safe_name, safe_mime = _prepare_recommender_upload_bytes(
                    query_file.getvalue(),
                    query_file.name,
                )
                files = {"file": (safe_name, safe_bytes, safe_mime)}
                params = {"k": k}
                
                response = make_api_call("/search", method="POST", files=files, params=params)
                
                if response:
                    data = response.json()
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Results Found", data["total_count"])
                    with col2:
                        st.metric("⚡ Processing Time", f"{data['processing_time_ms']:.2f}ms")
                    with col3:
                        st.metric("🔧 Model", data["model_used"])
                    
                    st.divider()
                    
                    # Display recommendations
                    if data["recommendations"]:
                        st.subheader("📊 Recommendations")
                        for idx, rec in enumerate(data["recommendations"], 1):
                            with st.container(border=True):
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1:
                                    st.markdown(f"**#{idx}** `{rec['garment_id']}`")
                                    if rec.get("metadata"):
                                        meta = rec["metadata"]
                                        meta_str = []
                                        if meta.get("category"):
                                            meta_str.append(f"📌 {meta['category']}")
                                        if meta.get("color"):
                                            meta_str.append(f"🎨 {meta['color']}")
                                        if meta.get("size"):
                                            meta_str.append(f"📏 {meta['size']}")
                                        if meta.get("price"):
                                            meta_str.append(f"💰 ${meta['price']}")
                                        if meta_str:
                                            st.caption(" | ".join(meta_str))

                                with col2:
                                    if rec.get("image_base64"):
                                        try:
                                            preview_bytes = base64.b64decode(rec["image_base64"])
                                            st.image(preview_bytes, caption="Garment", width=180)
                                        except Exception:
                                            st.caption("Preview unavailable")
                                    else:
                                        st.caption("Preview unavailable")
                                
                                with col3:
                                    st.markdown(f"<div class='similarity-badge'>{rec['similarity_score']:.2%}</div>", 
                                              unsafe_allow_html=True)
                    else:
                        st.info("No similar garments found.")


def index_stats_ui():
    """UI for displaying index statistics."""
    st.subheader("📈 Index Statistics")
    
    if st.button("🔄 Refresh Stats"):
        with st.spinner("Loading statistics..."):
            response = make_api_call("/stats", method="GET")
            
            if response:
                data = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📦 Total Items", data["total_items"])
                with col2:
                    st.metric("🔢 Feature Dimension", data["feature_dimension"])
                with col3:
                    st.metric("💾 Index Size", f"{data['index_size_mb']:.2f} MB")
                with col4:
                    st.metric("🔧 Model", data["model_type"])


def tryon_ui():
    """UI for virtual try-on generation using recommender search results."""
    st.subheader("👗 Virtual Try-On (Using Recommended Garments)")
    st.caption("VITON-only mode is enabled: if VITON output quality is poor, the API returns an error instead of fallback output.")
    
    # Initialize session state for selected garment
    if 'selected_garment' not in st.session_state:
        st.session_state.selected_garment = None
        st.session_state.selected_garment_base64 = None
        st.session_state.search_results = None
    
    # Show selected garment at top if one is selected
    if st.session_state.selected_garment:
        st.info(f"✅ **Selected Garment:** `{st.session_state.selected_garment}`")
        
        if st.button("🔄 Select a Different Garment", key="reset_garment_btn"):
            st.session_state.selected_garment = None
            st.session_state.selected_garment_base64 = None
            st.session_state.search_results = None
            st.rerun()
    
    # Step 1: Search for garments (only show if no garment selected)
    if not st.session_state.selected_garment:
        st.markdown("### Step 1: Find a Garment")
        st.markdown("Search for a garment using the recommender engine")
        
        query_file = st.file_uploader(
            "Upload a reference image to find similar garments",
            type=["jpg", "jpeg", "png"],
            key="tryon_search_file",
        )
        k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
        
        if query_file:
            if st.button("🔎 Search for Similar Garments", key="tryon_search_btn"):
                with st.spinner("Searching for garments..."):
                    files = {"file": query_file.getvalue()}
                    params = {"k": k}
                    
                    response = make_api_call("/search", method="POST", files=files, params=params)
                    
                    if response:
                        data = response.json()
                        st.session_state.search_results = data
        
        # Display search results if available
        if st.session_state.search_results:
            data = st.session_state.search_results
            if data.get("recommendations"):
                st.success(f"Found {data['total_count']} similar garments!")
                st.markdown("**Select a garment to use for try-on:**")
                
                # Create columns for garment selection
                for idx, rec in enumerate(data["recommendations"], 1):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        if rec.get("image_base64"):
                            try:
                                preview_bytes = base64.b64decode(rec["image_base64"])
                                st.image(preview_bytes, caption=f"#{idx}", width=120)
                            except Exception:
                                st.caption("Preview unavailable")
                    
                    with col2:
                        st.markdown(f"**{rec['garment_id']}**")
                        st.markdown(f"Similarity: {rec['similarity_score']:.2%}")
                    
                    with col3:
                        if st.button(f"Select", key=f"select_garment_{idx}"):
                            st.session_state.selected_garment = rec['garment_id']
                            st.session_state.selected_garment_base64 = rec.get("image_base64")
                            st.success(f"✓ Selected: {rec['garment_id']}")
                            st.rerun()
            else:
                st.info("No similar garments found.")
    
    # Step 2: Upload person image and generate try-on
    # This supports either a selected recommender garment OR a directly uploaded garment for stable demos.
    if st.session_state.selected_garment or True:
        st.markdown("### Step 2: Generate Try-On")
        st.caption("Tip: Use a front-facing person photo and a clean product garment image on plain background.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Garment Source:**")
            uploaded_garment_file = st.file_uploader(
                "Upload garment image directly (recommended for demo)",
                type=["jpg", "jpeg", "png"],
                key="tryon_direct_garment",
            )

            if uploaded_garment_file is not None:
                st.markdown("**Garment Preview (Uploaded):**")
                try:
                    st.image(uploaded_garment_file.getvalue(), caption=uploaded_garment_file.name, width=220)
                except Exception:
                    st.caption("Unable to preview uploaded garment image")

            st.markdown("**Selected from Recommender:**")
            if st.session_state.selected_garment_base64:
                try:
                    garment_preview = base64.b64decode(st.session_state.selected_garment_base64)
                    st.image(garment_preview, caption=st.session_state.selected_garment, width=200)
                except Exception:
                    st.caption("Preview unavailable")
        
        with col2:
            st.markdown("**Upload Person Image:**")
            person_file = st.file_uploader(
                "Upload a person/model image",
                type=["jpg", "jpeg", "png"],
                key="tryon_person",
            )

            if person_file is not None:
                st.markdown("**Person Preview:**")
                try:
                    person_preview_bytes = _prepare_person_upload_bytes(person_file.getvalue())
                    st.image(person_preview_bytes, caption=f"{person_file.name} (upload preview)", width=220)
                except Exception:
                    st.caption("Unable to preview uploaded person image")
            
            mode = st.radio(
                "Try-on mode",
                ["Quality try-on (recommended)", "Fast fallback try-on"],
                index=0,
                horizontal=True,
            )
        
        if person_file and st.button("✨ Generate Try-On", key="tryon_btn"):
            processed_person_bytes = _prepare_person_upload_bytes(person_file.getvalue())
            # Priority: direct uploaded garment > recommender-selected garment.
            if uploaded_garment_file is not None:
                garment_payload, garment_name, garment_mime, garment_compressed = _compress_garment_for_upload(
                    uploaded_garment_file.getvalue(),
                    uploaded_garment_file.name,
                )
                if garment_compressed:
                    st.info("Garment image was too large; applied compression fallback for upload.")

                if not _is_backend_bytes_valid(garment_payload):
                    st.error("Garment image is still invalid after compression. Please reduce size and upload again.")
                    return

                files = [
                    ("person_file", (person_file.name, processed_person_bytes, "image/png")),
                    ("garment_file", (garment_name, garment_payload, garment_mime)),
                ]

                params = {
                    "use_gan": mode == "Quality try-on (recommended)",
                }

                # Use streaming endpoint with progress bar
                try:
                    url = f"{API_URL}/api/tryon/generate-streaming"
                    
                    # Create progress container
                    progress_container = st.container()
                    progress_bar = progress_container.progress(0)
                    status_text = progress_container.empty()
                    time_text = progress_container.empty()
                    
                    result_image_b64 = None
                    model_used = None
                    processing_time = None
                    
                    response = requests.post(url, files=files, params=params, stream=True, timeout=1200)
                    response.raise_for_status()
                    
                    start_time = time.time() if 'time' in dir() else None
                    for line in response.iter_lines():
                        if line:
                            msg = json.loads(line)
                            status = msg.get("status", "")
                            progress = msg.get("progress", 0)
                            message = msg.get("message", "")
                            estimated = msg.get("estimated_seconds", 0)
                            elapsed = msg.get("elapsed_seconds", 0)
                            
                            progress_bar.progress(min(100, progress))
                            status_text.markdown(f"**{message}**")
                            
                            if estimated > 0:
                                time_text.markdown(f"⏱️ Estimated time: ~{estimated}s")
                            elif elapsed > 0:
                                time_text.markdown(f"⏱️ Elapsed time: {elapsed}s")
                            
                            if status == "success":
                                result_image_b64 = msg.get("result_image_base64")
                                model_used = msg.get("model_used")
                                processing_time = msg.get("processing_time_ms")
                            elif status == "error":
                                status_text.error(f"❌ {message}")
                                break
                    
                    # Clear progress UI and show result
                    progress_container.empty()
                    
                    if result_image_b64:
                        result_bytes = base64.b64decode(result_image_b64)
                        st.markdown("### Result")
                        result_col_l, result_col_c, result_col_r = st.columns([1, 2, 1])
                        with result_col_c:
                            st.image(result_bytes, caption="Try-On Result", width=500)

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Model Used", model_used or "Unknown")
                        with col_b:
                            st.metric("Processing Time", f"{processing_time:.2f} ms" if processing_time else "N/A")
                            
                except Exception as e:
                    st.error(f"❌ Try-on generation failed: {str(e)}")
            
            elif st.session_state.selected_garment_base64:
                garment_bytes = base64.b64decode(st.session_state.selected_garment_base64)
                garment_filename = f"{st.session_state.selected_garment}.jpg"
                garment_payload, garment_name, garment_mime, garment_compressed = _compress_garment_for_upload(
                    garment_bytes,
                    garment_filename,
                )
                if garment_compressed:
                    st.info("Selected garment was compressed for upload stability.")

                if not _is_backend_bytes_valid(garment_payload):
                    st.error("Garment image is still invalid after compression. Please reduce size and upload again.")
                    return
                
                files = [
                    ("person_file", (person_file.name, processed_person_bytes, "image/png")),
                    ("garment_file", (garment_name, garment_payload, garment_mime)),
                ]
                
                params = {
                    "use_gan": mode == "Quality try-on (recommended)",
                }
                
                response = make_api_call("/generate", method="POST", files=files, params=params, api_prefix="tryon")
                if response:
                    data = response.json()
                    result_bytes = base64.b64decode(data["result_image_base64"])
                    st.markdown("### Result")
                    result_col_l, result_col_c, result_col_r = st.columns([1, 2, 1])
                    with result_col_c:
                        st.image(result_bytes, caption="Try-On Result", width=500)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Model Used", data["model_used"])
                    with col_b:
                        st.metric("Processing Time", f"{data['processing_time_ms']:.2f} ms")
                    
                    if data["fallback_used"]:
                        st.info("💡 Fallback mode was used for faster processing.")
            else:
                st.warning("Please upload a garment image or select one from recommender results.")


def main():
    """Main Streamlit app."""
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("👕 StyleSync")
        st.markdown("### AI-Powered Fashion Recommender System")
    
    with col2:
        st.markdown("**Phase 1: Recommender Only**")
        st.info("🎯 Find similar fashion items efficiently")
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["🔍 Search", "📤 Upload", "📦 Bulk Upload", "👗 Try-On", "📈 Statistics"],
            index=0
        )
        
        st.divider()
        
        st.markdown("**API Status**")
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Server Online")
            else:
                st.error("❌ API Server Error")
        except:
            st.error("❌ API Server Offline")
        
        st.markdown("**About**")
        st.markdown("""
        This is the scoped StyleSync project focusing on:
        - Visual fashion recommendations using CLIP
        - FAISS-based vector search
        - MLflow and WandB experiment tracking
        - Optional Hugging Face checkpoint loading
        - Docker containerization
        - Virtual try-on with a resource-friendly fallback
        """)
    
    # Main content
    if page == "🔍 Search":
        search_recommender_ui()
    elif page == "📤 Upload":
        upload_garment_ui()
    elif page == "📦 Bulk Upload":
        bulk_upload_ui()
    elif page == "👗 Try-On":
        tryon_ui()
    elif page == "📈 Statistics":
        index_stats_ui()
    
    st.divider()
    
    # Footer
    st.markdown("""
    ---
    **StyleSync** - MLOps Project | IIT Jodhpur | Phase 1 (Recommender)
    
    Addressing Prof. Feedback:
    - ✅ Scoped down with a fallback try-on path
    - ✅ MLflow and WandB experiment tracking integrated
    - ✅ Hugging Face checkpoint support added
    - ✅ Docker containerization
    """)


if __name__ == "__main__":
    main()
