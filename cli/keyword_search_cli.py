#!/usr/bin/env python3

from pprint import pprint
import argparse
import json
from lib.inverted_index import InvertedIndex


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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
