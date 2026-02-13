from sympy.testing.tests.test_code_quality import EXAMPLES_PATH
from PIL import Image
from lib.utils import load_movies
from sentence_transformers import SentenceTransformer
import numpy as np

def image_search(image):
    movies_dataset = load_movies()
    if movies_dataset is None:
        raise ValueError("No movies found")
    movies = movies_dataset.get("movies", [])
    ms = MultimodalSearch(documents=movies)
    similarity_scores = ms.search_with_image(image)
    return similarity_scores

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def verify_image_embedding(img_path):
    ms = MultimodalSearch()
    embeddings = ms.embed_image(img_path)
    if embeddings is not None:
        embedding = embeddings[0]
        print(f"Embedding shape: {embedding.shape[0]} dimensions")
    else:
        raise ValueError("No embeddings")

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=[]):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{document.get('title', '')}: {document.get('description','')}" for document in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image):
        image_embedding = None
        with Image.open(image) as f:
            image_embedding = self.model.encode([f], show_progress_bar=True)
            return image_embedding
        return image_embedding

    def search_with_image(self, image):
        image_embedding = self.embed_image(image)[0]
        cosine_similarities = []
        for e, text_embed in enumerate(self.text_embeddings, 0):
            similarity = cosine_similarity(text_embed, image_embedding)
            cosine_similarities.append((similarity, self.documents[e]))
        cosine_similarities.sort(key=lambda x: x[0], reverse=True)
        return cosine_similarities

