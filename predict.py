import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from PIL import Image
import cv2
import tempfile
import os

MODEL = "Facenet"
DETECTOR = "opencv"  # Changed to opencv which is more lenient
IMAGE_SIZE = (224, 224)

with open("embeddings.pkl", "rb") as f:
    actor_db = pickle.load(f)

def validate_and_preprocess_image(img_path):
    """Validate image and preprocess it for better face detection"""
    try:
        # Read image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Cannot read image file")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save a temporary copy for DeepFace
        temp_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(temp_path, img_rgb)
        
        return temp_path, img_rgb
        
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def predict_actor(img_path, top_k=5):
    try:
        print(f"Processing image: {img_path}")
        
        # Step 1: Preprocess the image
        temp_path, img_rgb = validate_and_preprocess_image(img_path)
        
        # Step 2: Try multiple approaches for face detection
        query = None
        
        # Approach 1: Try with opencv detector (most lenient)
        try:
            print("Trying with opencv detector...")
            result = DeepFace.represent(
                img_path=temp_path,
                model_name=MODEL,
                detector_backend="opencv",
                enforce_detection=True,
                align=True
            )
            query = result[0]["embedding"]
            print("Success with opencv detector")
        except Exception as e1:
            print(f"Opencv failed: {str(e1)[:100]}")
            
            # Approach 2: Try with mtcnn
            try:
                print("Trying with mtcnn detector...")
                result = DeepFace.represent(
                    img_path=temp_path,
                    model_name=MODEL,
                    detector_backend="mtcnn",
                    enforce_detection=True,
                    align=True
                )
                query = result[0]["embedding"]
                print("Success with mtcnn detector")
            except Exception as e2:
                print(f"MTCNN failed: {str(e2)[:100]}")
                
                # Approach 3: Try without face detection (use entire image)
                try:
                    print("Trying without face detection...")
                    result = DeepFace.represent(
                        img_path=temp_path,
                        model_name=MODEL,
                        detector_backend="skip",  # Skip face detection
                        enforce_detection=False,
                        align=False
                    )
                    query = result[0]["embedding"]
                    print("Success with skip detector")
                except Exception as e3:
                    print(f"Skip detector failed: {str(e3)[:100]}")
                    raise Exception("Face not detected. Please upload a clear image with a visible face.")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        if query is None:
            raise Exception("Could not extract facial features from the image")
        
        # Step 3: Compare with database
        results = []

        for actor, actor_data in actor_db.items():
            for data in actor_data:
                emb = data["embedding"]
                image_path = data["image_path"]

                dist = cosine(query, emb)
                similarity = round((1 - dist) * 100, 2)

                results.append({
                    "actor": actor,
                    "similarity": similarity,
                    "image_path": image_path
                })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Get unique actors only
        final = []
        seen = set()
        for r in results:
            if r["actor"] not in seen:
                final.append(r)
                seen.add(r["actor"])
            if len(final) == top_k:
                break

        return final
        
    except Exception as e:
        print(f"Error in predict_actor: {str(e)}")
        raise Exception(str(e))