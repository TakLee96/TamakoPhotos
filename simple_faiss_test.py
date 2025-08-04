import requests
import time

def test_faiss_service():
    service_url = "http://127.0.0.1:8002"
    
    print("=== Testing FAISS Face Detection Service ===")
    
    # Test 1: Service stats
    print("\n1. Testing service statistics...")
    try:
        response = requests.get(f"{service_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("SUCCESS: Service stats retrieved:")
            print(f"   Total faces: {stats.get('total_faces', 0)}")
            print(f"   PyTorch available: {stats.get('pytorch_available', False)}")
            print(f"   Device: {stats.get('device', 'N/A')}")
            
            faiss_info = stats.get('faiss', {})
            print(f"   FAISS available: {faiss_info.get('available', False)}")
            print(f"   Indexed embeddings: {faiss_info.get('indexed_embeddings', 0)}")
            print(f"   Embedding dimension: {faiss_info.get('embedding_dimension', 0)}")
        else:
            print(f"ERROR: Failed to get stats - {response.status_code}")
            return
    except Exception as e:
        print(f"ERROR: Cannot connect to service - {e}")
        return
    
    # Test 2: Face detection
    print("\n2. Testing face detection...")
    image_path = "C:\\Users\\jiaha\\Developer\\TamakoPhotos\\photos\\010.JPG"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'photo_id': 'faiss_test_001',
                'photo_path': image_path
            }
            
            response = requests.post(f"{service_url}/detect-faces", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                face_count = result.get('face_count', 0)
                print(f"SUCCESS: Detected {face_count} faces")
                
                if face_count > 0:
                    faces = result.get('faces', [])
                    print("   Face details:")
                    for i, face in enumerate(faces[:3]):
                        conf = face.get('confidence', 0)
                        loc = face.get('location', {})
                        print(f"   Face {i+1}: confidence={conf:.3f}")
            else:
                print(f"ERROR: Face detection failed - {response.status_code}")
                
    except Exception as e:
        print(f"ERROR: Face detection error - {e}")
    
    # Test 3: Clustering
    print("\n3. Testing FAISS clustering...")
    try:
        response = requests.get(f"{service_url}/face-clusters")
        
        if response.status_code == 200:
            result = response.json()
            cluster_count = result.get('cluster_count', 0)
            clusters = result.get('clusters', [])
            
            print(f"SUCCESS: Created {cluster_count} clusters")
            
            # Count cluster types
            single_clusters = sum(1 for c in clusters if c.get('face_count', 0) == 1)
            multi_clusters = sum(1 for c in clusters if c.get('face_count', 0) > 1)
            
            print(f"   Single-face clusters: {single_clusters}")
            print(f"   Multi-face clusters: {multi_clusters}")
            
            # Show top clusters
            clusters_sorted = sorted(clusters, key=lambda x: x.get('face_count', 0), reverse=True)
            print("   Largest clusters:")
            for i, cluster in enumerate(clusters_sorted[:5]):
                face_count = cluster.get('face_count', 0)
                cluster_id = cluster.get('cluster_id', i)
                print(f"     Cluster {cluster_id}: {face_count} faces")
                
        else:
            print(f"ERROR: Clustering failed - {response.status_code}")
            
    except Exception as e:
        print(f"ERROR: Clustering error - {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_faiss_service()