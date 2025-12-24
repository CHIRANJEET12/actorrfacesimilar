import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from PIL import Image
import cv2
import tempfile

MODEL = "Facenet"
DETECTOR = "opencv"
IMAGE_SIZE = (224, 224)

with open("embeddings.pkl", "rb") as f:
    actor_db = pickle.load(f)

def predict_actor(img_path, top_k=5):
    try:
        # First, ensure the image has a face by trying to detect it
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        
        # Save to temp file for DeepFace processing
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_path = tmp.name
            img.save(temp_path, format="JPEG", quality=95)
        
        try:
            # Try to detect face first
            face_detected = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=DETECTOR,
                enforce_detection=True  # This will raise an exception if no face
            )
            
            # If we get here, face was detected
            query = DeepFace.represent(
                img_path=temp_path,
                model_name=MODEL,
                detector_backend=DETECTOR,
                enforce_detection=True
            )[0]["embedding"]
            
        except:
            # If no face detected with enforce_detection=True, try without
            try:
                query = DeepFace.represent(
                    img_path=temp_path,
                    model_name=MODEL,
                    detector_backend=DETECTOR,
                    enforce_detection=False
                )[0]["embedding"]
            except:
                raise Exception("Face not detected. Please upload a clear image with a visible face.")
        
        # Clean up temp file
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
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

        # Sort correctly
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Unique actors only
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
        raise Exception(str(e))