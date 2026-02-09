from lib.semantic_search.semantic_search import SemanticSearch
import os
import numpy as np
import json
from typing import Dict, List
import lib.utils as util
from lib.semantic_search import cosine_similarity


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: List | None = None
        self.chunk_metadata: Dict | None = None
        self.CHUNK_EMBEDDINGS_PATH = f"{self.CACHE_DIR}/chunk_embeddings.npy"
        self.CHUNK_METADATA_PATH = f"{self.CACHE_DIR}/chunk_metadata.json"

    def search_chunks(self, query: str, limit: int = 10):
        embeddings = self.generate_embedding(query)
        chunk_scores = []
        if (
            self.chunk_embeddings is not None
            and self.chunk_metadata is not None
            and self.chunk_metadata.get("chunks") is not None
        ):
            for i in range(len(self.chunk_embeddings)):
                chunk_meta = self.chunk_metadata["chunks"][i]
                chunk_embedding = self.chunk_embeddings[i]
                movie_idx = chunk_meta["movie_idx"]
                chunk_idx = chunk_meta["chunk_idx"]
                score = cosine_similarity(chunk_embedding, embeddings)
                chunk_score = {
                    "chunk_idx": chunk_idx,
                    "movie_idx": movie_idx,
                    "score": score,
                }
                chunk_scores.append(chunk_score)

            movie_idx_scores = dict()
            for chunk_score in chunk_scores:
                movie_idx = chunk_score["movie_idx"]
                score = chunk_score["score"]
                if movie_idx not in movie_idx_scores:
                    movie_idx_scores[movie_idx] = score
                else:
                    old_score = movie_idx_scores[movie_idx]
                    if score > old_score:
                        movie_idx_scores[movie_idx] = score

            movie_idx_scores = sorted(
                movie_idx_scores.items(), key=lambda x: x[1], reverse=True
            )
            submovie_idx_scores = []
            if limit <= len(movie_idx_scores):
                submovie_idx_scores = movie_idx_scores[:limit]
            else:
                submovie_idx_scores = movie_idx_scores

            items = []
            for sub in submovie_idx_scores:
                movie_idx = sub[0]
                score = round(sub[1], 4)
                document = self.documents[movie_idx]
                document["description"] = document["description"][:100]
                document["score"] = score
                document["metadata"] = self.chunk_metadata["chunks"][movie_idx] or {}
                items.append(document)
            return items

        raise ValueError("embeddings not generated. please run the appropriate command")

    def __populate_docs(self, doc_representations, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
            doc_representation = f"{document['title']}: {document['description']}"
            doc_representations.append(doc_representation)

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        doc_representations = []
        self.__populate_docs(doc_representations, documents)
        chunks = []
        chunk_metadata = []
        for movie_idx, document in enumerate(self.documents, 0):
            description = document.get("description")
            if description is None:
                continue
            if description.strip() == "":
                continue
            _chunks = util.semantic_chunker(description, overlap=1)
            chunks.extend([c[1] for c in _chunks])
            for i in range(len(_chunks)):
                metadata = {
                    "movie_idx": movie_idx,
                    "chunk_idx": _chunks[i][0],
                    "total_chunks": len(_chunks[i][1]),
                }
                chunk_metadata.append(metadata)
        self.chunk_embeddings = self.model.encode([" ".join(c) for c in chunks])
        np.save(self.CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(self.CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(_chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        doc_representations = []
        self.__populate_docs(doc_representations, documents)
        if os.path.isfile(self.CHUNK_EMBEDDINGS_PATH) and os.path.isfile(
            self.CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(self.CHUNK_EMBEDDINGS_PATH)
            with open(self.CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        self.chunk_embeddings = self.build_chunk_embeddings(documents)
        return self.chunk_embeddings
