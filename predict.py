import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from PIL import Image

MODEL = "Facenet"
DETECTOR = "retinaface"   # IMPORTANT
IMAGE_SIZE = (224, 224)

with open("embeddings.pkl", "rb") as f:
    actor_db = pickle.load(f)

def predict_actor(img_path, top_k=5):
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img)

    # Get embedding (DO NOT enforce detection)
    query = DeepFace.represent(
        img_path=img_array,
        model_name=MODEL,
        detector_backend=DETECTOR,
        enforce_detection=False
    )[0]["embedding"]

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
