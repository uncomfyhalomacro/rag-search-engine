from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    s = SemanticSearch()
    embedding = s.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    with open("data/movies.json", "r") as f:
        j = json.load(f)
        movies = j["movies"]
        s = SemanticSearch()
        embeddings = s.load_or_create_embeddings(movies)
        print(f"Number of docs:   {len(s.documents)}")
        print(
            f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
        )


def embed_text(text):
    s = SemanticSearch()
    embedding = s.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = []
        self.documents = []
        self.document_map = dict()
        self.CACHE_DIR = "cache"
        self.MOVIE_EMBED_PATH = f"{self.CACHE_DIR}/movie_embeddings.npy"

    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embed = self.generate_embedding(query)
        cosine_similarities = []
        for i, doc_embed in enumerate(self.embeddings, start=1):
            similarity = cosine_similarity(query_embed, doc_embed)
            cosine_similarities.append((similarity, self.document_map[i]))
        cosine_similarities.sort(key=lambda x: x[0], reverse=True)
        if len(cosine_similarities) >= limit:
            return cosine_similarities[:limit]
        return cosine_similarities

    def __populate_docs(self, doc_representations, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
            doc_representation = f"{document['title']}: {document['description']}"
            doc_representations.append(doc_representation)

    def build_embeddings(self, documents):
        self.documents = documents
        doc_representations = []
        self.__populate_docs(doc_representations, documents)
        self.embeddings = self.model.encode(doc_representations, show_progress_bar=True)
        if not os.path.isdir(self.CACHE_DIR):
            os.mkdir(self.CACHE_DIR)
        np.save(self.MOVIE_EMBED_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        doc_representations = []
        self.__populate_docs(doc_representations, documents)
        if os.path.isfile(self.MOVIE_EMBED_PATH):
            self.embeddings = np.load(self.MOVIE_EMBED_PATH)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
            else:
                raise ValueError("length between documents and embeddings do not match")
        else:
            self.build_embeddings(documents)
        if len(self.embeddings) == len(self.documents):
            return self.embeddings
        else:
            raise ValueError("length between documents and embeddings do not match")

    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("text contains only whitespace or is an empty string")

        return self.model.encode([text])[0]
