from lib.semantic_search.semantic_search import SemanticSearch
import os
import numpy as np
import json
from typing import Dict, List
import lib.utils as util


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: List | None = None
        self.chunk_metadata: List | None = None
        self.CHUNK_EMBEDDINGS_PATH = f"{self.CACHE_DIR}/chunk_embeddings.npy"
        self.CHUNK_METADATA_PATH = f"{self.CACHE_DIR}/chunk_metadata.json"

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
        for document in self.documents:
            description = document.get("description")
            if description is None:
                continue
            if description.strip() == "":
                continue
            _chunks = util.semantic_chunker(description, overlap=1)
            chunks.extend([c[1] for c in _chunks])
            for i in range(len(_chunks)):
                metadata = {
                    "movie_idx": document["id"],
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
