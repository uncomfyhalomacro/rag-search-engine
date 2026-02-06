#!/usr/bin/env python3

from pprint import pprint
import argparse
import json
from lib.inverted_index import InvertedIndex
from lib.constants import BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build search index")
    idf_parser = subparsers.add_parser("idf", help="Calculate IDF for a given term")
    idf_parser.add_argument("term", help="The term to calculate IDF")
    tf_parser = subparsers.add_parser(
        "tf", help="Find term frequencies from a term and a doc id"
    )
    tf_params = tf_parser.add_argument_group("params")
    tf_params.add_argument(
        "doc_id",
        type=int,
        help="Pass the doc id to get a specific document's term frequency",
    )
    tf_params.add_argument(
        "term", type=str, help="Returns the frequency of the passed term value"
    )
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Find IDF from a term and a doc id"
    )
    tfidf_params = tfidf_parser.add_argument_group("params")
    tfidf_params.add_argument(
        "doc_id", type=int, help="Pass the doc id to get a specific document's term IDF"
    )
    tfidf_params.add_argument(
        "term", type=str, help="Returns the IDF of the passed term value"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    args = parser.parse_args()
    index = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            index.load()
            tokens = args.query.split()
            for token in tokens:
                doc_ids = index.get_documents(token)
                matches = [index.docmap[doc_id] for doc_id in doc_ids]
                pprint(matches)
            pass
        case "build":
            with open("data/movies.json", "r") as f:
                j = json.load(f)
                movies = j["movies"]
                index.build(movies=movies)
                index.save()
        case "tf":
            print(f"Getting term frequency of {args.term} by doc id {args.doc_id}")
            index.load()
            print(index.get_tf(args.doc_id, args.term))
        case "tfidf":
            print(f"Getting term IDF of {args.term} by doc id {args.doc_id}")
            index.load()
            tf_idf = index.tfidf(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "idf":
            print(f"Getting term IDF of {args.term}")
            index.load()
            idf = index.idf(term=args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "bm25idf":
            print(f"Getting term bm25 IDF of {args.term}")
            index.load()
            bm25idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            print(
                f"Getting term bm25 adjusted frequency of {args.term} by doc id {args.doc_id}"
            )
            index.load()
            bm25tf = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
