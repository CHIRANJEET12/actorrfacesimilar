import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace
import tempfile

MODEL = "Facenet"
DETECTOR = "opencv"

print("Starting to build actor database...")

actors = os.listdir('actors')
print(f"Found {len(actors)} items in actors folder")

actlist = {}
processed_actors = 0
total_images = 0
MAX_FILES = 5
failed_images = 0


def resize_image_to_224x224(image_path):
    """Resize image to 224x224 pixels and save to temp file"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Resize image to 224x224
        resized_img = cv2.resize(img, (224, 224))

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, resized_img)
        return temp_file.name

    except Exception as e:
        print(f"  ⚠ Resize error for {image_path}: {str(e)[:50]}")
        return None


for actor in actors:
    actor_path = os.path.join('actors', actor)

    if not os.path.isdir(actor_path):
        print(f"Skipping {actor} (not a directory)")
        continue
    print(f"\nProcessing: {actor}")

    all_files = os.listdir(actor_path)

    image_files = []
    for f in all_files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            image_files.append(f)
    image_files = image_files[:MAX_FILES]
    embeddings = []

    for file in image_files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        file_path = os.path.join(actor_path, file)
        temp_file_path = None

        try:
            # Resize image to 224x224
            temp_file_path = resize_image_to_224x224(file_path)
            if temp_file_path is None:
                failed_images += 1
                print(f"  ✗ Failed to resize: {file}")
                continue

            # Process resized image
            rep = DeepFace.represent(
                img_path=temp_file_path,
                model_name=MODEL,
                detector_backend=DETECTOR,
                enforce_detection=True,
            )
            embeddings.append({
                'embedding': rep[0]['embedding'],
                'image_path': file_path,
                'resized_path': temp_file_path  # Optional: store temp path
            })
            total_images += 1
            print(f"  ✓ Processed (224x224): {file}")

        except Exception as e:
            failed_images += 1
            print(f"  ✗ Failed: {file} - {str(e)[:50]}...")

        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

    if embeddings:
        actlist[actor] = embeddings
        processed_actors += 1
        print(f"  ✅ Added {len(embeddings)} embeddings for {actor}")
    else:
        print(f"  ❌ No embeddings extracted for {actor}")

if actlist:
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(actlist, f)
        print(f"\n{'=' * 50}")
        print("DATABASE BUILD COMPLETE")
        print(f"{'=' * 50}")
        print(f"Total actors processed: {processed_actors}")
        print(f"Total images successfully embedded: {total_images}")
        print(f"Total images failed: {failed_images}")
        print("All images resized to: 224x224 pixels")
        print(f"Database saved to: embeddings.pkl")

        # Quick check
        file_size = os.path.getsize('embeddings.pkl')
        print(f"File size: {file_size:,} bytes")

else:
    print("\n❌ No embeddings were created!")
    print("Possible reasons:")
    print("1. No valid images in actors folder")
    print("2. Images don't contain faces")
    print("3. DeepFace can't detect faces in the images")