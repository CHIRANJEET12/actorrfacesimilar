import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import cv2
import tempfile
import os

MODEL = "Facenet"

# Load the actor database
try:
    with open("embeddings.pkl", "rb") as f:
        actor_db = pickle.load(f)
    print(f"Loaded database with {len(actor_db)} actors")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    actor_db = {}

def preprocess_image(image_path):
    """Preprocess image to improve face detection"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Save to temp file (DeepFace works better with file paths)
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, img)
        return temp_file.name
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return image_path

def predict_actor(img_path, top_k=5):
    """Find the most similar Bollywood actor"""
    try:
        print(f"Starting prediction for: {img_path}")
        
        # Preprocess the image
        processed_path = preprocess_image(img_path)
        
        # Try different face detectors
        detectors = ["opencv", "mtcnn", "retinaface"]
        embedding = None
        
        for detector in detectors:
            try:
                print(f"Trying {detector} detector...")
                result = DeepFace.represent(
                    img_path=processed_path,
                    model_name=MODEL,
                    detector_backend=detector,
                    enforce_detection=False,  # Set to False to handle cases without clear faces
                    align=True
                )
                
                if result and len(result) > 0:
                    embedding = result[0]["embedding"]
                    print(f"Successfully extracted embedding using {detector}")
                    break
                    
            except Exception as e:
                print(f"Detector {detector} failed: {str(e)[:100]}")
                continue
        
        # Clean up temp file if created
        if processed_path != img_path and os.path.exists(processed_path):
            os.unlink(processed_path)
        
        if embedding is None:
            raise Exception("Could not analyze the face in the image. Please upload a clearer photo.")
        
        # Compare with database
        results = []
        
        for actor, actor_data in actor_db.items():
            for data in actor_data:
                try:
                    db_embedding = data["embedding"]
                    image_path = data["image_path"]
                    
                    # Calculate similarity
                    distance = cosine(embedding, db_embedding)
                    similarity = round((1 - distance) * 100, 2)
                    
                    results.append({
                        "actor": actor,
                        "similarity": similarity,
                        "image_path": image_path
                    })
                except Exception as e:
                    print(f"Error comparing with {actor}: {e}")
                    continue
        
        if not results:
            return []
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get unique actors
        final_results = []
        seen_actors = set()
        
        for result in results:
            if result["actor"] not in seen_actors:
                final_results.append(result)
                seen_actors.add(result["actor"])
            
            if len(final_results) >= top_k:
                break
        
        print(f"Found {len(final_results)} matches")
        return final_results
        
    except Exception as e:
        print(f"Error in predict_actor: {str(e)}")
        raise Exception(str(e))