"""
Example: Using StyleSync Recommender Programmatically
Demonstrates all major features of Phase 1
"""
import requests
from pathlib import Path
from PIL import Image
import io
import json

# API Configuration
API_URL = "http://localhost:8000"
RECOMMENDER_ENDPOINT = f"{API_URL}/api/recommender"


def example_1_upload_single_garment():
    """Example 1: Upload a single garment"""
    print("\n" + "="*60)
    print("Example 1: Upload Single Garment")
    print("="*60)
    
    # Create a sample image (solid color)
    img = Image.new('RGB', (256, 256), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Prepare request
    files = {"file": img_byte_arr}
    params = {
        "category": "shirt",
        "color": "red",
        "size": "M",
        "price": 29.99
    }
    
    # Upload
    response = requests.post(
        f"{RECOMMENDER_ENDPOINT}/upload-garment",
        files=files,
        params=params,
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Upload successful!")
        print(f"   Garment ID: {data['garment_id']}")
        return data['garment_id']
    else:
        print(f"❌ Upload failed: {response.text}")
        return None


def example_2_bulk_upload():
    """Example 2: Bulk upload multiple garments"""
    print("\n" + "="*60)
    print("Example 2: Bulk Upload")
    print("="*60)
    
    # Create 3 sample images
    files = []
    for i, color in enumerate(['red', 'blue', 'green']):
        img = Image.new('RGB', (256, 256), color=color)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        files.append(("files", img_byte_arr))
    
    # Bulk upload
    response = requests.post(
        f"{RECOMMENDER_ENDPOINT}/bulk-upload",
        files=files,
        timeout=60
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Bulk upload complete!")
        print(f"   Total: {data['total_uploaded']}")
        print(f"   Successful: {data['successful']}")
        print(f"   Failed: {data['failed']}")
    else:
        print(f"❌ Bulk upload failed: {response.text}")


def example_3_search_similar():
    """Example 3: Search for similar garments"""
    print("\n" + "="*60)
    print("Example 3: Search Similar Garments")
    print("="*60)
    
    # Create query image
    query_img = Image.new('RGB', (256, 256), color='red')
    img_byte_arr = io.BytesIO()
    query_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Search
    files = {"file": img_byte_arr}
    params = {"k": 5}
    
    response = requests.post(
        f"{RECOMMENDER_ENDPOINT}/search",
        files=files,
        params=params,
        timeout=60
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Search complete!")
        print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
        print(f"   Found: {data['total_count']} items")
        print(f"   Model used: {data['model_used']}")
        print("\n   Results:")
        for i, rec in enumerate(data['recommendations'], 1):
            print(f"   {i}. {rec['garment_id']} "
                  f"(similarity: {rec['similarity_score']:.2%})")
    else:
        print(f"❌ Search failed: {response.text}")


def example_4_get_statistics():
    """Example 4: Get index statistics"""
    print("\n" + "="*60)
    print("Example 4: Index Statistics")
    print("="*60)
    
    response = requests.get(
        f"{RECOMMENDER_ENDPOINT}/stats",
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Statistics retrieved!")
        print(f"   Total items: {data['total_items']}")
        print(f"   Feature dimension: {data['feature_dimension']}")
        print(f"   Model type: {data['model_type']}")
        print(f"   Index size: {data['index_size_mb']:.2f} MB")
    else:
        print(f"❌ Statistics request failed: {response.text}")


def example_5_check_api_health():
    """Example 5: Check API health"""
    print("\n" + "="*60)
    print("Example 5: API Health Check")
    print("="*60)
    
    # Health check
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        print(f"✅ API is healthy!")
        data = response.json()
        print(f"   Status: {data.get('status')}")
    else:
        print(f"❌ API is not responding")
    
    # Root endpoint
    response = requests.get(f"{API_URL}/", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ API Information:")
        print(f"   Service: {data.get('service')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Phase: {data.get('phase')}")


def example_6_api_documentation():
    """Example 6: Access API documentation"""
    print("\n" + "="*60)
    print("Example 6: API Documentation")
    print("="*60)
    
    print("API documentation available at:")
    print(f"  - Swagger UI: {API_URL}/docs")
    print(f"  - ReDoc: {API_URL}/redoc")
    print("\nEndpoints:")
    print("  POST   /api/recommender/upload-garment     - Upload single garment")
    print("  POST   /api/recommender/bulk-upload        - Upload multiple items")
    print("  POST   /api/recommender/search             - Search similar items")
    print("  GET    /api/recommender/stats              - Get index statistics")
    print("  POST   /api/recommender/save-index         - Save index to disk")
    print("  POST   /api/recommender/reset-index        - Reset index")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("StyleSync Phase 1: Example Usage")
    print("="*60)
    
    print("\n📝 Prerequisites:")
    print("  - Backend running: python -m backend.main")
    print("  - Database running: docker-compose up -d db")
    print("  - MLflow running: docker-compose up -d mlflow")
    
    # Check API availability
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        print(f"\n✅ API is available at {API_URL}")
    except Exception as e:
        print(f"\n❌ API not available: {e}")
        print("\n   Make sure the backend is running!")
        return
    
    # Run examples
    try:
        example_5_check_api_health()
        example_1_upload_single_garment()
        example_2_bulk_upload()
        example_3_search_similar()
        example_4_get_statistics()
        example_6_api_documentation()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
