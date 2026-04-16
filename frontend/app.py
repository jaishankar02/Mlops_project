"""
Streamlit frontend for StyleSync Recommender.
Phase 1: Recommender only
"""
import streamlit as st
import requests
from PIL import Image
import io
from typing import List
import logging
import os

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
API_URL = os.getenv("API_URL", "http://localhost:8000")

def make_api_call(endpoint: str, method: str = "GET", files=None, params=None):
    """Make API call with error handling."""
    try:
        url = f"{API_URL}/api/recommender{endpoint}"
        
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
            files = {"file": uploaded_file.getvalue()}
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
            files = [("files", f.getvalue()) for f in uploaded_files]
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
        st.image(query_image, caption="Query Garment", use_column_width=True, width=200)
        
        if st.button("🔎 Find Similar", key="search_btn"):
            with st.spinner("Searching..."):
                files = {"file": query_file.getvalue()}
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
                                col1, col2 = st.columns([3, 1])
                                
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
            ["🔍 Search", "📤 Upload", "📦 Bulk Upload", "📈 Statistics"],
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
        This is Phase 1 of the StyleSync project focusing on:
        - Visual fashion recommendations using CLIP
        - FAISS-based vector search
        - MLflow experiment tracking
        - Docker containerization
        
        **Phase 2** will add GAN-based virtual try-on.
        """)
    
    # Main content
    if page == "🔍 Search":
        search_recommender_ui()
    elif page == "📤 Upload":
        upload_garment_ui()
    elif page == "📦 Bulk Upload":
        bulk_upload_ui()
    elif page == "📈 Statistics":
        index_stats_ui()
    
    st.divider()
    
    # Footer
    st.markdown("""
    ---
    **StyleSync** - MLOps Project | IIT Jodhpur | Phase 1 (Recommender)
    
    Addressing Prof. Feedback:
    - ✅ Scoped down to recommender only (GAN as Phase 2)
    - ✅ MLflow experiment tracking integrated
    - ✅ CI/CD pipeline with GitHub Actions
    - ✅ Docker containerization
    """)


if __name__ == "__main__":
    main()
