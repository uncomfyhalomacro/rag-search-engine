import os

from lib.semantic_search import chunked_semantic_search as chunked_ss
from lib.keyword_search.inverted_index import InvertedIndex
from lib import utils


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.doc_hybridized_scores = dict()
        self.semantic_search = chunked_ss.ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_scores = self._bm25_search(query, limit=limit * 500)
        semantic_scores = self.semantic_search.search_chunks(query, limit=limit * 500)
        _k_scores = [k[1] for k in bm25_scores]
        _s_scores = [s["score"] for s in semantic_scores]
        _k_scores.extend(_s_scores)
        normal_scores = utils.normalise_scores(_k_scores)
        for normal_score, k_meta, s_meta in zip(
            normal_scores, bm25_scores, semantic_scores
        ):
            k_score = k_meta[1]
            s_score = s_meta["score"]
            h_score = utils.hybrid_score(k_score, s_score, alpha)
            title = s_meta["title"]
            description = s_meta["description"]
            doc_id = s_meta["id"]
            item = {
                "id": doc_id,
                "description": description,
                "title": title,
                "semantic_score": s_score,
                "bm25_score": k_score,
                "hybrid_score": h_score,
            }
            self.doc_hybridized_scores[doc_id] = item
        res = sorted(
            self.doc_hybridized_scores.items(),
            key=lambda x: x[1]["hybrid_score"],
            reverse=True,
        )
        if limit <= len(res):
            return res[:limit]
        return res

        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
