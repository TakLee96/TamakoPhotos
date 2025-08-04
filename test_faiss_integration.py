import os
import time
import requests
import cv2
import numpy as np
from PIL import Image
import json

def test_face_service_with_faiss():
    """Test the face detection service with FAISS integration"""
    
    # Service URL
    service_url = "http://127.0.0.1:8002"  # Service runs on 8002
    
    print("=== FAISS Integration Test ===")
    
    # Test 1: Check service stats
    print("\n1. Testing service statistics...")
    try:
        response = requests.get(f"{service_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("OK Service stats retrieved successfully:")
            print(f"   Total faces: {stats.get('total_faces', 0)}")
            print(f"   PyTorch available: {stats.get('pytorch_available', False)}")
            print(f"   Device: {stats.get('device', 'N/A')}")
            print(f"   Advanced models: {stats.get('advanced_models', False)}")
            
            faiss_info = stats.get('faiss', {})
            print(f"   FAISS available: {faiss_info.get('available', False)}")
            print(f"   Indexed embeddings: {faiss_info.get('indexed_embeddings', 0)}")
            print(f"   Embedding dimension: {faiss_info.get('embedding_dimension', 0)}")
        else:
            print(f"ERROR Failed to get stats: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR Error connecting to service: {e}")
        return False
    
    # Test 2: Test face detection with a sample image
    print("\n2. Testing face detection...")
    sample_image_path = "C:\\Users\\jiaha\\Developer\\TamakoPhotos\\photos\\010.JPG"
    
    if not os.path.exists(sample_image_path):
        print(f"ERROR Sample image not found: {sample_image_path}")
        return False
    
    try:
        with open(sample_image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'photo_id': 'test_faiss_001',
                'photo_path': sample_image_path
            }
            
            response = requests.post(f"{service_url}/detect-faces", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                face_count = result.get('face_count', 0)
                print(f"OK Face detection successful: {face_count} faces detected")
                
                if face_count > 0:
                    print("   Sample face info:")
                    faces = result.get('faces', [])
                    for i, face in enumerate(faces[:3]):  # Show first 3 faces
                        location = face.get('location', {})
                        confidence = face.get('confidence', 0)
                        print(f"   Face {i+1}: confidence={confidence:.3f}, location=({location.get('x', 0)}, {location.get('y', 0)})")
            else:
                print(f"ERROR Face detection failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"ERROR Error in face detection: {e}")
        return False
    
    # Test 3: Test face clustering
    print("\n3. Testing FAISS-based face clustering...")
    try:
        response = requests.get(f"{service_url}/face-clusters")
        
        if response.status_code == 200:
            result = response.json()
            cluster_count = result.get('cluster_count', 0)
            clusters = result.get('clusters', [])
            
            print(f"✅ Face clustering successful: {cluster_count} clusters created")
            
            # Show cluster statistics
            single_face_clusters = sum(1 for c in clusters if c.get('face_count', 0) == 1)
            multi_face_clusters = sum(1 for c in clusters if c.get('face_count', 0) > 1)
            
            print(f"   Single-face clusters: {single_face_clusters}")
            print(f"   Multi-face clusters: {multi_face_clusters}")
            
            # Show largest clusters
            clusters_by_size = sorted(clusters, key=lambda x: x.get('face_count', 0), reverse=True)
            print("   Top 5 largest clusters:")
            for i, cluster in enumerate(clusters_by_size[:5]):
                face_count = cluster.get('face_count', 0)
                cluster_id = cluster.get('cluster_id', i)
                print(f"     Cluster {cluster_id}: {face_count} face(s)")
                
        else:
            print(f"❌ Face clustering failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error in face clustering: {e}")
        return False
    
    # Test 4: Test similarity search (if we have faces)
    if face_count > 0:
        print("\n4. Testing FAISS similarity search...")
        try:
            with open(sample_image_path, 'rb') as f:
                files = {'file': f}
                data = {'k': 5}
                
                response = requests.post(f"{service_url}/find-similar-faces", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success', False):
                        similar_faces = result.get('similar_faces', [])
                        print(f"✅ Similarity search successful: {len(similar_faces)} similar faces found")
                        
                        for i, face in enumerate(similar_faces[:3]):  # Show top 3
                            similarity = face.get('similarity', 0)
                            photo_id = face.get('photo_id', 'unknown')
                            print(f"   Similar face {i+1}: similarity={similarity:.3f}, photo={photo_id}")
                    else:
                        print("⚠️  Similarity search returned no matches")
                else:
                    print(f"❌ Similarity search failed: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
    
    print("\n=== FAISS Integration Test Complete ===")
    return True

def benchmark_faiss_vs_traditional():
    """Benchmark FAISS performance against traditional methods"""
    print("\n=== Performance Benchmark ===")
    
    service_url = "http://127.0.0.1:8002"
    
    # Get current stats
    try:
        response = requests.get(f"{service_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            total_faces = stats.get('total_faces', 0)
            indexed_embeddings = stats.get('faiss', {}).get('indexed_embeddings', 0)
            
            print(f"Dataset size: {total_faces} faces")
            print(f"FAISS indexed: {indexed_embeddings} embeddings")
            
            if total_faces < 10:
                print("⚠️  Small dataset - benchmark may not be meaningful")
            
        else:
            print("❌ Could not get service stats for benchmark")
            return
            
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return
    
    # Benchmark clustering time
    print("\nBenchmarking clustering performance...")
    try:
        start_time = time.time()
        response = requests.get(f"{service_url}/face-clusters?threshold=0.6")
        clustering_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            cluster_count = result.get('cluster_count', 0)
            print(f"✅ FAISS clustering: {clustering_time:.3f}s for {cluster_count} clusters")
        else:
            print(f"❌ Clustering benchmark failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error benchmarking clustering: {e}")

if __name__ == "__main__":
    print("Starting FAISS integration tests...")
    
    # Wait a moment for service to be ready
    time.sleep(2)
    
    success = test_face_service_with_faiss()
    
    if success:
        benchmark_faiss_vs_traditional()
    
    print("\nTest complete!")