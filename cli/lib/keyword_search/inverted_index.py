import pickle
import os
from lib.keyword_search.utils import tokenize_text
from lib.keyword_search.constants import BM25_K1, BM25_B
from collections import Counter
import math


class InvertedIndex:
    def __init__(self):
        # token -> set(). a token mapped to a set of doc ids
        self.index = dict()
        # doc ids -> document object
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()
        self.CACHE_DIR = "cache"
        self.INDEX_PATH = f"{self.CACHE_DIR}/index.pkl"
        self.DOCMAP_PATH = f"{self.CACHE_DIR}/docmap.pkl"
        self.TF_PATH = f"{self.CACHE_DIR}/term_frequencies.pkl"
        self.DOC_LENGTHS_PATH = f"{self.CACHE_DIR}/doc_lengths.pkl"

    def tfidf(self, doc_id, term) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.idf(term)
        return tf * idf

    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        print(tokens)
        scores = dict()
        for token in tokens:
            doc_ids = self.index.get(token, set())
            for doc_id in doc_ids:
                if scores.get(doc_id) is None:
                    scores[doc_id] = 0
                bm25_score = self.bm25(doc_id, token)
                scores[doc_id] += bm25_score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if limit < len(sorted_scores):
            return sorted_scores[:limit]
        return sorted_scores

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        avg_len = self.__get_avg_doc_length()
        if avg_len == 0:
            return 0.0  # or just tf, but 0.0 is fine if no docs
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_len)
        tf = self.get_tf(doc_id, term)
        sat_value = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return sat_value

    def get_bm25_idf(self, term: str) -> float:
        words = term.split()
        if len(words) > 1:
            raise Exception("only one token is expected for this term")
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise Exception("only one token is expected for this term")
        token = tokens[0]
        total_documents = len(self.docmap.keys())
        doc_ids = self.index.get(token, set())
        df = len(doc_ids)
        return math.log((total_documents - df + 0.5) / (df + 0.5) + 1)

    def idf(self, term) -> float:
        words = term.split()
        if len(words) > 1:
            raise Exception("only one token is expected for this term")
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise Exception("only one token is expected for this term")
        token = tokens[0]
        doc_ids = self.index.get(token)
        len_doc_ids = len(doc_ids) if doc_ids is not None else 0
        return math.log((len(self.docmap) + 1) / (len_doc_ids + 1))

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = Counter(tokens).total()

    def __get_avg_doc_length(self) -> float:
        num_docs = len(self.doc_lengths)
        if num_docs == 0:
            return 0.0
        total_len = sum(self.doc_lengths.values())
        return total_len / num_docs

    def get_tf(self, doc_id, term) -> int | float:
        words = term.split()
        if len(words) > 1:
            raise Exception("only one token is expected for this term")
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise Exception("only one token is expected for this term")
        token = tokens[0]
        doc_object = self.term_frequencies.get(doc_id)
        if doc_object is not None:
            tf = doc_object.get(token)
            if tf is not None:
                return tf
        return 0

    def get_documents(self, term, limit=5) -> list[int]:
        try:
            res = list(self.index[term.lower()])
            res.sort()
            if len(res) > limit:
                return res[:limit]
            return res
        except Exception as e:
            print(e)
            return []

    def build(self, movies=[]):
        for movie in movies:
            title = movie["title"]
            description = movie["description"]
            doc_id = movie["id"]
            title_and_description = f"{title} {description}"
            self.__add_document(doc_id, title_and_description)
            self.docmap[doc_id] = movie

    def save(self):
        if not os.path.isdir(self.CACHE_DIR):
            os.mkdir(self.CACHE_DIR)
        with open(self.INDEX_PATH, "wb") as index_pkl:
            pickle.dump(self.index, index_pkl)
            index_pkl.close()
        with open(self.DOCMAP_PATH, "wb") as docmap_pkl:
            pickle.dump(self.docmap, docmap_pkl)
            docmap_pkl.close()
        with open(self.TF_PATH, "wb") as term_frequencies_pkl:
            pickle.dump(self.term_frequencies, term_frequencies_pkl)
            term_frequencies_pkl.close()
        with open(self.DOC_LENGTHS_PATH, "wb") as doc_lengths_pkl:
            pickle.dump(self.doc_lengths, doc_lengths_pkl)
            doc_lengths_pkl.close()

    def load(self):
        if not os.path.isfile(self.DOCMAP_PATH):
            raise FileNotFoundError("file not found: cache/docmap.pkl")
        if not os.path.isfile(self.INDEX_PATH):
            raise FileNotFoundError("file not found: cache/index.pkl")
        if not os.path.isfile(self.TF_PATH):
            raise FileNotFoundError("file not found: cache/term_frequencies.pkl")
        if not os.path.isfile(self.DOC_LENGTHS_PATH):
            raise FileNotFoundError("file not found: cache/doc_lengths.pkl")

        with open(self.INDEX_PATH, "rb") as index_pkl:
            self.index = pickle.load(index_pkl)

        with open(self.DOCMAP_PATH, "rb") as docmap_pkl:
            self.docmap = pickle.load(docmap_pkl)

        with open(self.TF_PATH, "rb") as term_frequencies_pkl:
            self.term_frequencies = pickle.load(term_frequencies_pkl)

        with open(self.DOC_LENGTHS_PATH, "rb") as doc_lengths_pkl:
            self.doc_lengths = pickle.load(doc_lengths_pkl)
