import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import matplotlib.pyplot as plt

def test_pytorch_cuda():
    """Test PyTorch CUDA setup"""
    print("=== PyTorch CUDA Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

def test_yolo_face_detection(image_path):
    """Test YOLO face detection"""
    print("=== YOLO Face Detection Test ===")
    try:
        # Load YOLO face detection model
        model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
        model = YOLO(model_path)
        
        start_time = time.time()
        results = model.predict(image_path, save=False, verbose=False)
        detection_time = time.time() - start_time
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                print(f"YOLO detected {len(boxes)} faces in {detection_time:.3f}s")
                for i, box in enumerate(boxes):
                    conf = box.conf[0].item()
                    print(f"  Face {i+1}: confidence = {conf:.3f}")
                return len(boxes), detection_time
            else:
                print("YOLO detected 0 faces")
                return 0, detection_time
        else:
            print("YOLO: No results returned")
            return 0, detection_time
    except Exception as e:
        print(f"YOLO face detection error: {e}")
        return 0, 0

def test_mtcnn_face_detection(image_path, device):
    """Test MTCNN face detection"""
    print("\n=== MTCNN Face Detection Test ===")
    try:
        # Initialize MTCNN with optimized parameters
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
            factor=0.709,
            post_process=True,
            device=device,
            select_largest=False,  # Detect all faces, not just largest
            keep_all=True
        )
        
        # Load and process image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        start_time = time.time()
        boxes, probs = mtcnn.detect(pil_image)
        detection_time = time.time() - start_time
        
        if boxes is not None:
            print(f"MTCNN detected {len(boxes)} faces in {detection_time:.3f}s")
            high_conf_faces = sum(1 for prob in probs if prob > 0.9)
            print(f"  High confidence (>0.9): {high_conf_faces}")
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                print(f"  Face {i+1}: confidence = {prob:.3f}")
            return len(boxes), detection_time, boxes, probs
        else:
            print("MTCNN detected 0 faces")
            return 0, detection_time, None, None
    except Exception as e:
        print(f"MTCNN face detection error: {e}")
        return 0, 0, None, None

def test_facenet_embeddings(image_path, device):
    """Test FaceNet embedding generation"""
    print("\n=== FaceNet Embedding Test ===")
    try:
        # Initialize models
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=True
        )
        
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Detect faces
        boxes, probs = mtcnn.detect(pil_image)
        if boxes is None:
            print("No faces detected for embedding")
            return []
        
        # Extract faces and generate embeddings
        start_time = time.time()
        face_tensors = mtcnn.extract(pil_image, boxes, save_path=None)
        if face_tensors is None:
            print("Failed to extract face tensors")
            return []
        
        embeddings = []
        with torch.no_grad():
            for i, face_tensor in enumerate(face_tensors):
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    embedding = facenet(face_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    embeddings.append(embedding.cpu().numpy().flatten())
        
        embedding_time = time.time() - start_time
        
        print(f"Generated {len(embeddings)} embeddings in {embedding_time:.3f}s")
        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            print(f"  Embedding {i+1}: dimension={len(emb)}, norm={norm:.3f}")
        
        return embeddings
    except Exception as e:
        print(f"FaceNet embedding error: {e}")
        return []

def test_embedding_similarity(embeddings):
    """Test similarity calculation between embeddings"""
    print("\n=== Embedding Similarity Test ===")
    if len(embeddings) < 2:
        print("Need at least 2 embeddings for similarity test")
        return
    
    try:
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                emb1, emb2 = embeddings[i], embeddings[j]
                
                # Cosine similarity
                cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                # Euclidean distance
                euclidean_dist = np.linalg.norm(emb1 - emb2)
                
                print(f"Face {i+1} vs Face {j+1}:")
                print(f"  Cosine similarity: {cosine_sim:.3f}")
                print(f"  Euclidean distance: {euclidean_dist:.3f}")
                
                # Interpretation
                if cosine_sim > 0.8:
                    print("  -> Likely same person")
                elif cosine_sim > 0.6:
                    print("  -> Possibly same person")
                else:
                    print("  -> Likely different people")
    except Exception as e:
        print(f"Similarity calculation error: {e}")

def main():
    # Test image path
    image_path = "C:\\Users\\jiaha\\Developer\\TamakoPhotos\\photos\\010.JPG"
    
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing with image: {image_path}")
    print(f"Using device: {device}")
    print("="*50)
    
    # Run tests
    test_pytorch_cuda()
    
    yolo_faces, yolo_time = test_yolo_face_detection(image_path)
    mtcnn_faces, mtcnn_time, boxes, probs = test_mtcnn_face_detection(image_path, device)
    
    # Test embeddings if faces were detected
    if mtcnn_faces > 0:
        embeddings = test_facenet_embeddings(image_path, device)
        if len(embeddings) > 0:
            test_embedding_similarity(embeddings)
    
    # Summary
    print("\n" + "="*50)
    print("=== SUMMARY ===")
    print(f"YOLO detection: {yolo_faces} faces in {yolo_time:.3f}s")
    print(f"MTCNN detection: {mtcnn_faces} faces in {mtcnn_time:.3f}s")
    
    if yolo_faces != mtcnn_faces:
        print("⚠️  Detection count mismatch - consider parameter tuning")
    
    if device.type == 'cuda':
        print("✅ GPU acceleration enabled")
    else:
        print("⚠️  Running on CPU - consider GPU setup")

if __name__ == "__main__":
    main()