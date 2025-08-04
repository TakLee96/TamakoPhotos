from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import socket
import time
import cv2
import numpy as np
from PIL import Image
import json
import os
from typing import List, Dict, Any
import base64
from io import BytesIO
from sklearn.cluster import DBSCAN
import faiss

# PyTorch and FaceNet imports (avoid transformers for now due to protobuf conflict)
try:
    import torch
    import torch.nn.functional as F
    from facenet_pytorch import MTCNN, InceptionResnetV1
    PYTORCH_AVAILABLE = True
    print("PyTorch and FaceNet loaded successfully")
except ImportError as e:
    print(f"PyTorch/FaceNet not available: {e}")
    PYTORCH_AVAILABLE = False

app = FastAPI(title="Face Detection Service")

class FaceDetectionService:
    def __init__(self):
        self.face_metadata = []
        self.faces_dir = "faces"
        self.metadata_file = "face_metadata.json"
        self.faiss_index_file = "face_embeddings.index"
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.embedding_dimension = 512  # FaceNet embedding dimension
        
        # Initialize OpenCV face detector (fallback)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize advanced face recognition models
        self.mtcnn = None
        self.facenet = None
        self.device = None
        
        if PYTORCH_AVAILABLE:
            self.init_pytorch_models()
        
        os.makedirs(self.faces_dir, exist_ok=True)
        self.load_metadata()
        self.initialize_faiss_index()
    
    def init_pytorch_models(self):
        """Initialize PyTorch-based face recognition models"""
        try:
            # Set device (GPU if available, CPU otherwise)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Initialize MTCNN for face detection and cropping with optimized parameters
            self.mtcnn = MTCNN(
                image_size=160, 
                margin=20,  # Add margin around faces for better embedding quality
                min_face_size=15,  # Detect smaller faces
                thresholds=[0.5, 0.6, 0.7],  # More sensitive P-Net threshold
                factor=0.709, 
                post_process=True,
                device=self.device,
                keep_all=True,  # Keep all detected faces
                select_largest=False  # Don't just select the largest face
            )
            
            # Initialize FaceNet for embeddings
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            print("Advanced face recognition models initialized successfully")
            
        except Exception as e:
            print(f"Error initializing PyTorch models: {e}")
            self.mtcnn = None
            self.facenet = None
    
    def initialize_faiss_index(self):
        """Initialize or load FAISS index for fast similarity search"""
        try:
            if os.path.exists(self.faiss_index_file) and len(self.face_metadata) > 0:
                # Load existing index
                print("Loading existing FAISS index...")
                self.faiss_index = faiss.read_index(self.faiss_index_file)
                print(f"Loaded FAISS index with {self.faiss_index.ntotal} embeddings")
            else:
                # Create new index - using inner product for cosine similarity
                print("Creating new FAISS index...")
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
                
                # If we have existing face metadata, rebuild the index
                if len(self.face_metadata) > 0:
                    self.rebuild_faiss_index()
                    
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            # Fallback to creating empty index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
    
    def rebuild_faiss_index(self):
        """Rebuild FAISS index from existing face metadata"""
        try:
            print("Rebuilding FAISS index from existing metadata...")
            
            # Reset the index
            self.faiss_index.reset()
            
            # Extract valid embeddings
            valid_embeddings = []
            for face_data in self.face_metadata:
                if 'embedding' in face_data and face_data['embedding']:
                    embedding = np.array(face_data['embedding'], dtype=np.float32)
                    if len(embedding) == self.embedding_dimension and np.linalg.norm(embedding) > 0:
                        # Normalize for cosine similarity
                        embedding = embedding / np.linalg.norm(embedding)
                        valid_embeddings.append(embedding)
            
            if valid_embeddings:
                embeddings_matrix = np.vstack(valid_embeddings).astype(np.float32)
                self.faiss_index.add(embeddings_matrix)
                print(f"Added {len(valid_embeddings)} embeddings to FAISS index")
                
                # Save the index
                self.save_faiss_index()
            
        except Exception as e:
            print(f"Error rebuilding FAISS index: {e}")
    
    def save_faiss_index(self):
        """Save FAISS index to disk"""
        try:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, self.faiss_index_file)
                print(f"Saved FAISS index with {self.faiss_index.ntotal} embeddings")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def load_metadata(self):
        """Load existing face metadata"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.face_metadata = json.load(f)
                print(f"Loaded {len(self.face_metadata)} face metadata entries")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.face_metadata = []
    
    def save_metadata(self):
        """Save face metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.face_metadata, f)
            print("Saved face metadata")
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def extract_face_embedding(self, landmarks) -> np.ndarray:
        """Extract face embedding from MediaPipe landmarks"""
        try:
            # Use key facial landmarks as features
            key_points = [
                # Eye corners, nose tip, mouth corners, face outline points
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 
                157, 158, 159, 160, 161, 246, 9, 10, 151, 
                # Additional distinctive points
                1, 2, 5, 6, 19, 20, 94, 125, 141, 235, 31, 228, 
                229, 230, 231, 232, 233, 244, 245, 122, 6, 202,
                214, 234, 93, 132, 58, 172, 136, 150, 149, 176,
                148, 152, 377, 400, 378, 379, 365, 397, 288, 361,
                323
            ]
            
            # Extract normalized coordinates for key points
            embedding = []
            for point_idx in key_points[:50]:  # Use first 50 points
                if point_idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[point_idx]
                    embedding.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(embedding[:150])  # Fixed-size embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return np.zeros(150)
    
    def detect_faces(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using advanced PyTorch models or fallback to OpenCV"""
        try:
            print(f"Starting face detection on image of shape: {image_array.shape}")
            
            # Try advanced PyTorch-based detection first
            if self.mtcnn is not None and self.facenet is not None:
                return self.detect_faces_pytorch(image_array)
            else:
                print("Falling back to OpenCV face detection")
                return self.detect_faces_opencv(image_array)
                
        except Exception as e:
            print(f"Error in face detection: {e}")
            # Fallback to OpenCV if PyTorch fails
            return self.detect_faces_opencv(image_array)
    
    def detect_faces_pytorch(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Advanced face detection using MTCNN and FaceNet embeddings"""
        try:
            detection_start = time.time()
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Detect faces and get embeddings
            mtcnn_start = time.time()
            boxes, probs = self.mtcnn.detect(pil_image)
            mtcnn_time = time.time() - mtcnn_start
            print(f"MTCNN detection took {mtcnn_time:.3f}s")
            
            if boxes is None:
                print("MTCNN detected 0 faces")
                return []
            
            print(f"MTCNN detected {len(boxes)} faces")
            
            # Track processing metrics
            embedding_start = time.time()
            faces = []
            processed_faces = 0
            skipped_faces = 0
            
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.8:  # Skip low-confidence detections (increased threshold)
                    skipped_faces += 1
                    continue
                
                # Extract face coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x, y, w, h = x1, y1, x2-x1, y2-y1
                
                # Extract and preprocess face for embedding with improved handling
                try:
                    face_tensor = self.mtcnn.extract(pil_image, [box], save_path=None)
                    if face_tensor is None or len(face_tensor) == 0:
                        print(f"Failed to extract face tensor for face {i}")
                        continue
                    
                    # Generate embedding using FaceNet with enhanced preprocessing
                    with torch.no_grad():
                        # Ensure proper tensor format and device placement
                        if face_tensor.dim() == 3:
                            face_tensor = face_tensor.unsqueeze(0)
                        face_tensor = face_tensor.to(self.device)
                        
                        # Generate embedding with improved normalization
                        embedding = self.facenet(face_tensor)
                        embedding = F.normalize(embedding, p=2, dim=1)
                        
                        # Verify embedding quality
                        embedding_np = embedding.cpu().numpy().flatten()
                        if np.linalg.norm(embedding_np) < 0.1:  # Check for degenerate embeddings
                            print(f"Warning: Low-quality embedding for face {i}")
                            continue
                            
                        embedding = embedding_np.tolist()
                        
                except Exception as e:
                    print(f"Error generating embedding for face {i}: {e}")
                    skipped_faces += 1
                    continue
                
                # Extract face image for thumbnail
                face_image = image_array[max(0,y):min(image_array.shape[0],y+h), 
                                       max(0,x):min(image_array.shape[1],x+w)]
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                pil_face = Image.fromarray(face_rgb)
                buffer = BytesIO()
                pil_face.save(buffer, format="JPEG")
                face_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                faces.append({
                    'face_id': i,
                    'location': {'x': x, 'y': y, 'width': w, 'height': h},
                    'face_image': face_base64,
                    'embedding': embedding,
                    'confidence': float(prob)
                })
                processed_faces += 1
            
            # Log performance metrics
            embedding_time = time.time() - embedding_start
            total_time = time.time() - detection_start
            
            print(f"Face processing metrics:")
            print(f"  Total detection time: {total_time:.3f}s")
            print(f"  MTCNN time: {mtcnn_time:.3f}s")
            print(f"  Embedding time: {embedding_time:.3f}s")
            print(f"  Processed faces: {processed_faces}")
            print(f"  Skipped faces: {skipped_faces}")
            print(f"  Success rate: {processed_faces/(processed_faces+skipped_faces)*100:.1f}%" if (processed_faces+skipped_faces) > 0 else "N/A")
            
            return faces
            
        except Exception as e:
            print(f"Error in PyTorch face detection: {e}")
            return []
    
    def detect_faces_opencv(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback OpenCV face detection with improved embeddings"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            print(f"Converted to grayscale: {gray.shape}")
            
            # Detect faces with more aggressive parameters
            faces_rects = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  # More sensitive
                minNeighbors=3,    # Less strict
                minSize=(20, 20)   # Smaller minimum size
            )
            print(f"OpenCV detected {len(faces_rects)} faces")
            
            faces = []
            for i, (x, y, w, h) in enumerate(faces_rects):
                # Extract face image
                face_image = image_array[y:y+h, x:x+w]
                
                # Convert to RGB and then to base64
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                buffer = BytesIO()
                pil_image.save(buffer, format="JPEG")
                face_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Generate better embeddings using face features
                embedding = self.generate_face_embedding(face_image, image_array.shape)
                
                faces.append({
                    'face_id': i,
                    'location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'face_image': face_base64,
                    'embedding': embedding,
                    'confidence': 0.8  # Fixed confidence for OpenCV
                })
            
            return faces
        except Exception as e:
            print(f"Error in OpenCV face detection: {e}")
            return []
    
    def generate_face_embedding(self, face_image: np.ndarray, img_shape: tuple) -> List[float]:
        """Generate improved face embedding from face image"""
        try:
            img_height, img_width = img_shape[:2]
            face_h, face_w = face_image.shape[:2]
            
            # Basic geometric features
            geometric_features = [
                face_w / img_width,  # relative width
                face_h / img_height,  # relative height
                face_w / face_h if face_h > 0 else 1.0,  # aspect ratio
                (face_w * face_h) / (img_width * img_height),  # area ratio
            ]
            
            # Enhanced intensity and texture features
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            if face_gray.size > 0:
                # Resize to standard size
                face_resized = cv2.resize(face_gray, (64, 64))
                
                # Extract features from different regions
                h, w = face_resized.shape
                regions = {
                    'forehead': face_resized[:h//3, :],
                    'eyes': face_resized[h//3:2*h//3, :],
                    'mouth': face_resized[2*h//3:, :],
                    'left': face_resized[:, :w//2],
                    'right': face_resized[:, w//2:],
                    'center': face_resized[h//4:3*h//4, w//4:3*w//4]
                }
                
                intensity_features = []
                for region_name, region in regions.items():
                    if region.size > 0:
                        intensity_features.extend([
                            region.mean() / 255.0,  # average intensity
                            region.std() / 255.0,   # intensity variation
                            np.median(region) / 255.0,  # median intensity
                        ])
                    else:
                        intensity_features.extend([0.5, 0.1, 0.5])  # default values
                
                # Add histogram features
                hist = cv2.calcHist([face_resized], [0], None, [16], [0, 256])
                hist_features = (hist.flatten() / hist.sum()).tolist()  # normalize
                
            else:
                intensity_features = [0.5] * 18  # 6 regions × 3 features
                hist_features = [1.0/16] * 16  # uniform histogram
            
            # Combine all features
            all_features = geometric_features + intensity_features + hist_features
            
            # Pad or truncate to fixed size
            target_size = 512
            if len(all_features) < target_size:
                all_features.extend([0.0] * (target_size - len(all_features)))
            else:
                all_features = all_features[:target_size]
            
            return all_features
            
        except Exception as e:
            print(f"Error generating face embedding: {e}")
            return [0.0] * 512  # Return zeros on error
    
    def add_faces_to_metadata(self, faces: List[Dict[str, Any]], photo_id: str, photo_path: str):
        """Add detected faces to metadata storage and FAISS index"""
        try:
            new_embeddings = []
            
            for face in faces:
                # Add metadata
                face_metadata = {
                    'face_id': len(self.face_metadata),
                    'photo_id': photo_id,
                    'photo_path': photo_path,
                    'location': face['location'],
                    'face_image': face['face_image'],
                    'embedding': face.get('embedding', []),
                    'confidence': face.get('confidence', 0.5)
                }
                self.face_metadata.append(face_metadata)
                
                # Prepare embedding for FAISS index
                embedding = face.get('embedding', [])
                if embedding and len(embedding) == self.embedding_dimension:
                    embedding_np = np.array(embedding, dtype=np.float32)
                    if np.linalg.norm(embedding_np) > 0:
                        # Normalize for cosine similarity
                        embedding_np = embedding_np / np.linalg.norm(embedding_np)
                        new_embeddings.append(embedding_np)
            
            # Add new embeddings to FAISS index
            if new_embeddings and self.faiss_index is not None:
                embeddings_matrix = np.vstack(new_embeddings).astype(np.float32)
                self.faiss_index.add(embeddings_matrix)
                print(f"Added {len(new_embeddings)} new embeddings to FAISS index")
                self.save_faiss_index()
            
            self.save_metadata()
            return True
        except Exception as e:
            print(f"Error adding faces to metadata: {e}")
            return False
    
    def find_similar_faces(self, query_faces: List[Dict[str, Any]], k: int = 10, threshold: float = 0.6):
        """Find similar faces using FAISS for fast similarity search"""
        try:
            if not query_faces or not self.face_metadata or self.faiss_index is None:
                return []
            
            search_start = time.time()
            query_embedding = np.array(query_faces[0]['embedding'], dtype=np.float32)
            
            # Normalize query embedding for cosine similarity
            if np.linalg.norm(query_embedding) > 0:
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            else:
                print("Query embedding has zero norm")
                return []
            
            print(f"FAISS searching for similar faces among {self.faiss_index.ntotal} indexed embeddings")
            
            # Search using FAISS (returns inner product scores, which equal cosine similarity for normalized vectors)
            search_k = min(k * 2, self.faiss_index.ntotal)  # Search more to account for filtering
            scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), search_k)
            
            similar_faces = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.face_metadata):  # Valid index
                    cosine_sim = float(score)  # FAISS inner product = cosine similarity for normalized vectors
                    
                    if cosine_sim > threshold:
                        face_data = self.face_metadata[idx].copy()
                        face_data['similarity'] = cosine_sim
                        face_data['faiss_score'] = cosine_sim
                        similar_faces.append(face_data)
                        
                        if len(similar_faces) >= k:  # Got enough results
                            break
            
            search_time = time.time() - search_start
            print(f"FAISS similarity search completed in {search_time:.3f}s, found {len(similar_faces)} matches above threshold {threshold}")
            
            return similar_faces
        except Exception as e:
            print(f"Error finding similar faces with FAISS: {e}")
            return self._fallback_similarity_search(query_faces, k, threshold)
    
    def _fallback_similarity_search(self, query_faces: List[Dict[str, Any]], k: int = 10, threshold: float = 0.6):
        """Fallback similarity search without FAISS"""
        try:
            if not query_faces or not self.face_metadata:
                return []
            
            query_embedding = np.array(query_faces[0]['embedding'])
            similarities = []
            
            for i, face_data in enumerate(self.face_metadata):
                if 'embedding' in face_data and face_data['embedding']:
                    stored_embedding = np.array(face_data['embedding'])
                    
                    if np.linalg.norm(query_embedding) > 0 and np.linalg.norm(stored_embedding) > 0:
                        cosine_sim = np.dot(query_embedding, stored_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                        )
                        similarities.append((cosine_sim, face_data))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            similar_faces = []
            
            for cosine_sim, face_data in similarities[:k]:
                if cosine_sim > threshold:
                    face_copy = face_data.copy()
                    face_copy['similarity'] = float(cosine_sim)
                    similar_faces.append(face_copy)
            
            return similar_faces
        except Exception as e:
            print(f"Error in fallback similarity search: {e}")
            return []
    
    def get_face_clusters(self, similarity_threshold: float = 0.6):
        """Group faces into clusters using FAISS-based similarity clustering"""
        try:
            print(f"DEBUG: Starting FAISS-based clustering with {len(self.face_metadata)} faces")
            if not self.face_metadata or self.faiss_index is None:
                print("DEBUG: No face metadata or FAISS index found")
                return []
            
            # Extract valid embeddings and faces
            valid_embeddings = []
            valid_faces = []
            
            for i, face in enumerate(self.face_metadata):
                if 'embedding' in face and face['embedding']:
                    embedding = np.array(face['embedding'], dtype=np.float32)
                    if len(embedding) == self.embedding_dimension and np.linalg.norm(embedding) > 0:
                        # Normalize for cosine similarity
                        embedding = embedding / np.linalg.norm(embedding)
                        valid_embeddings.append(embedding)
                        valid_faces.append(face)
            
            print(f"DEBUG: Found {len(valid_faces)} valid faces for clustering")
            
            if len(valid_faces) == 0:
                return []
            
            # Create clusters using FAISS similarity search
            clusters = []
            visited = set()
            cluster_id = 0
            
            for i, face in enumerate(valid_faces):
                if i in visited:
                    continue
                
                # Start new cluster with this face
                cluster_faces = [face]
                visited.add(i)
                
                # Find similar faces using FAISS
                query_embedding = valid_embeddings[i].reshape(1, -1)
                scores, indices = self.faiss_index.search(query_embedding, min(50, len(valid_faces)))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(valid_faces) and idx not in visited:
                        if score >= similarity_threshold:  # Similar enough to be in same cluster
                            cluster_faces.append(valid_faces[idx])
                            visited.add(idx)
                
                # Create cluster (always include, even single-face clusters)
                clusters.append({
                    'cluster_id': cluster_id,
                    'face_count': len(cluster_faces),
                    'faces': cluster_faces
                })
                cluster_id += 1
                
                print(f"DEBUG: Created cluster {cluster_id-1} with {len(cluster_faces)} faces")
            
            print(f"DEBUG: Final FAISS-based clustering result: {len(clusters)} clusters")
            print(f"DEBUG: Cluster sizes: {[c['face_count'] for c in clusters]}")
            
            # Sort clusters by size (largest first) but keep all clusters including single-face ones
            clusters.sort(key=lambda x: x['face_count'], reverse=True)
            
            return clusters
            
        except Exception as e:
            print(f"Error creating FAISS face clusters: {e}")
            return self._fallback_clustering(similarity_threshold)
    
    def _fallback_clustering(self, similarity_threshold: float = 0.6):
        """Fallback clustering without FAISS - ensures single-face clusters are preserved"""
        try:
            print("DEBUG: Using fallback clustering method")
            if not self.face_metadata:
                return []
            
            # Simple similarity-based clustering
            valid_faces = []
            for face in self.face_metadata:
                if 'embedding' in face and face['embedding']:
                    embedding = np.array(face['embedding'])
                    if np.linalg.norm(embedding) > 0:
                        valid_faces.append(face)
            
            if not valid_faces:
                return []
            
            clusters = []
            visited = set()
            cluster_id = 0
            
            for i, face1 in enumerate(valid_faces):
                if i in visited:
                    continue
                
                cluster_faces = [face1]
                visited.add(i)
                
                embedding1 = np.array(face1['embedding'])
                norm1 = np.linalg.norm(embedding1)
                
                # Find similar faces
                for j, face2 in enumerate(valid_faces):
                    if j <= i or j in visited:
                        continue
                    
                    embedding2 = np.array(face2['embedding'])
                    norm2 = np.linalg.norm(embedding2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
                        if similarity >= similarity_threshold:
                            cluster_faces.append(face2)
                            visited.add(j)
                
                # Always create cluster (including single-face clusters)
                clusters.append({
                    'cluster_id': cluster_id,
                    'face_count': len(cluster_faces),
                    'faces': cluster_faces
                })
                cluster_id += 1
            
            print(f"DEBUG: Fallback clustering created {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            print(f"Error in fallback clustering: {e}")
            # Last resort: create individual clusters for each face
            clusters = []
            for i, face in enumerate(self.face_metadata):
                if 'embedding' in face and face['embedding']:
                    clusters.append({
                        'cluster_id': i,
                        'face_count': 1,
                        'faces': [face]
                    })
            return clusters

# Global service instance
face_service = FaceDetectionService()

@app.post("/detect-faces")
async def detect_faces_endpoint(
    file: UploadFile = File(...),
    photo_id: str = None,
    photo_path: str = None
):
    """Detect faces in uploaded image"""
    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        faces = face_service.detect_faces(image)
        
        # Add faces to metadata if photo_id provided
        if photo_id and photo_path:
            if faces:
                print(f"SUCCESS: Adding {len(faces)} faces to metadata for photo {photo_id}")
                success = face_service.add_faces_to_metadata(faces, photo_id, photo_path)
                print(f"Metadata save result: {success}")
                print(f"Total faces in metadata now: {len(face_service.face_metadata)}")
            else:
                print(f"WARNING: No faces detected in photo {photo_id}. Image processed successfully but no faces found.")
        else:
            print(f"ERROR: Missing photo_id or photo_path for face detection")
        
        return JSONResponse({
            "success": True,
            "face_count": len(faces),
            "faces": faces
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/face-clusters")
async def get_face_clusters_endpoint(threshold: float = 0.6):
    """Get face clusters"""
    try:
        print(f"Getting face clusters with threshold {threshold}")
        print(f"Total faces in metadata: {len(face_service.face_metadata)}")
        
        clusters = face_service.get_face_clusters(threshold)
        print(f"Generated {len(clusters)} clusters")
        
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i}: {cluster['face_count']} faces")
        
        return JSONResponse({
            "success": True,
            "cluster_count": len(clusters),
            "clusters": clusters
        })
    except Exception as e:
        print(f"ERROR in get_face_clusters_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-faces")
async def find_similar_faces_endpoint(file: UploadFile = File(...), k: int = 10):
    """Find similar faces to uploaded image"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces in query image
        faces = face_service.detect_faces(image)
        if not faces:
            return JSONResponse({
                "success": False,
                "message": "No faces detected in query image"
            })
        
        # Use detected faces for similarity search
        similar_faces = face_service.find_similar_faces(faces, k)
        
        return JSONResponse({
            "success": True,
            "similar_faces": similar_faces
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    faiss_stats = {
        "available": face_service.faiss_index is not None,
        "indexed_embeddings": face_service.faiss_index.ntotal if face_service.faiss_index else 0,
        "embedding_dimension": face_service.embedding_dimension
    }
    
    stats = {
        "total_faces": len(face_service.face_metadata),
        "metadata_count": len(face_service.face_metadata),
        "pytorch_available": PYTORCH_AVAILABLE,
        "device": str(face_service.device) if face_service.device else "N/A",
        "advanced_models": face_service.mtcnn is not None and face_service.facenet is not None,
        "faiss": faiss_stats
    }
    print(f"Stats requested: {stats}")
    return JSONResponse(stats)

def find_free_port(start_port=8000, max_port=8010):
    """Find a free port starting from start_port"""
    for port in range(start_port, max_port + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            sock.close()
            return port
        except OSError:
            continue
    return None

if __name__ == "__main__":
    print("Starting Face Detection Service...")
    print(f"Initial face metadata count: {len(face_service.face_metadata)}")
    
    # Find a free port
    port = find_free_port()
    if port is None:
        print("ERROR: No free ports found between 8000-8010")
        exit(1)
    
    print(f"Using port: {port}")
    try:
        uvicorn.run(app, host="127.0.0.1", port=port)
    except Exception as e:
        print(f"ERROR starting server: {e}")
        exit(1)