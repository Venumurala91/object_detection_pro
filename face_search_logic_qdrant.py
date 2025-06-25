# face_search_logic_qdrant.py (FIXED)
import numpy as np
import cv2
import os
import insightface
from insightface.app import FaceAnalysis
import uuid
import math
from qdrant_client import QdrantClient, models

# --- CONFIGURATION ---
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "face_db_collection_qdrant"
FACE_DATABASE_PATH = "face_database"
MODEL_NAME = 'buffalo_l'
VECTOR_DIMENSION = 512
DISTANCE_METRIC = models.Distance.EUCLID


class FaceSearchEngine:
    """Business logic for face search using InsightFace and Qdrant."""

    def __init__(self):
        self.app_model = None
        self.client = None
        print("FaceSearchEngine (using Qdrant) instance created.")

    def load_models(self):
        """Loads the InsightFace model."""
        print("Initializing InsightFace model...")
        self.app_model = FaceAnalysis(name=MODEL_NAME, allowed_modules=['detection', 'recognition'])
        self.app_model.prepare(ctx_id=-1, det_size=(640, 640))
        print("InsightFace model loaded successfully.")

    def connect_to_qdrant(self):
        """Establishes connection to the Qdrant server."""
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print("Successfully connected to Qdrant.")

    def disconnect_from_qdrant(self):
        """Closes the connection to Qdrant."""
        if self.client:
            print("Disconnecting from Qdrant...")
            self.client.close()
            print("Successfully disconnected from Qdrant.")

    def load_or_create_collection(self):
        """Connects to Qdrant and ensures the collection is ready."""
        self.connect_to_qdrant()
        collections_response = self.client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]

        if COLLECTION_NAME in existing_collections:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
        else:
            print(f"Creating collection '{COLLECTION_NAME}'...")
            self.client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC),
            )
            print("Collection created.")
        print(f"Collection '{COLLECTION_NAME}' is ready.")

    def _extract_all_embeddings(self, image_np):
        """Extracts embeddings for ALL detected faces in an image."""
        faces = self.app_model.get(image_np)
        if not faces:
            return []
        return [face.normed_embedding for face in faces]

    def search_person(self, query_image_np, top_k=5):
        """
        Searches for all faces in the query image and returns combined, unique results.
        """
        query_embeddings = self._extract_all_embeddings(query_image_np)
        if not query_embeddings:
            return {"status": "No face detected in the uploaded image.", "results": []}

        all_hits = []
        # --- Perform a search for each face found in the query image ---
        for embedding in query_embeddings:
            hits = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=embedding,
                limit=top_k
            )
            all_hits.extend(hits)

        # --- De-duplicate and process the results ---
        unique_hits = {}
        for hit in all_hits:
            image_path = hit.payload["image_path"]
            # If we haven't seen this image before, or if the new hit is a closer match
            if image_path not in unique_hits or hit.score < unique_hits[image_path].score:
                unique_hits[image_path] = hit

        # Convert back to a list and sort by score (distance)
        sorted_unique_hits = sorted(list(unique_hits.values()), key=lambda h: h.score)

        # --- Format the final results with the distance threshold ---
        # Qdrant's EUCLID score is squared Euclidean distance. We take the sqrt to get L2 distance.
        # The original distance threshold was 1.1 for L2 distance.
        final_results = [
            {"image_path": hit.payload["image_path"], "distance": math.sqrt(hit.score)}
            for hit in sorted_unique_hits if math.sqrt(hit.score) < 1.1
        ]

        status_msg = f"Found {len(query_embeddings)} face(s) in query image. Search complete. Found {len(final_results)} potential matches."
        return {"status": status_msg, "results": final_results}

    def add_images_from_directory(self, progress=None):
        if not os.path.exists(FACE_DATABASE_PATH):
            return {"status": "Database directory not found.", "images_added": 0, "faces_added": 0}

        if progress: progress(0, desc="Querying existing images in Qdrant...")

        processed_paths = set()
        next_offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_offset
            )
            for record in records:
                processed_paths.add(record.payload["image_path"])
            if next_offset is None:
                break

        all_db_images = [os.path.join(FACE_DATABASE_PATH, f) for f in os.listdir(FACE_DATABASE_PATH) if
                         f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        new_images = [img_path for img_path in all_db_images if img_path not in processed_paths]

        if not new_images:
            return {"status": "No new images found to add.", "images_added": 0, "faces_added": 0}

        points_to_insert = []
        images_processed_count = 0
        total_new_images = len(new_images)
        print(f"Processing {total_new_images} new images...")

        for i, img_path in enumerate(new_images):
            if progress:
                progress((i + 1) / total_new_images, desc=f"Processing {os.path.basename(img_path)}")
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                faces = self.app_model.get(img)
                if not faces: continue

                images_processed_count += 1
                for face in faces:
                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=face.normed_embedding.tolist(),
                        payload={"image_path": img_path}
                    )
                    points_to_insert.append(point)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if not points_to_insert:
            return {"status": "New images found, but no new faces could be extracted.",
                    "images_added": images_processed_count, "faces_added": 0}

        if progress: progress(0.95, desc="Inserting embeddings into Qdrant...")

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_insert,
            wait=True
        )

        if progress: progress(1.0, desc="Complete!")
        print(f"Upsert complete. Added {len(points_to_insert)} new face vectors.")

        return {"status": "Successfully added new faces.", "images_added": images_processed_count,
                "faces_added": len(points_to_insert)}