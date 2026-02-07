#!/usr/bin/env python3

import argparse
import json
from lib.semantic_search import (
    SemanticSearch,
    embed_text,
    verify_embeddings,
    embed_query_text,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify currently used model")
    subparsers.add_parser("verify_embeddings", help="Verify currently used model")
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query text")
    embed_query_parser.add_argument("query", help="Query text to generate embedding")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedded text")
    embed_parser.add_argument("text", help="Text to use for embed generation")
    search_parser = subparsers.add_parser("search", help="Semantic search a query")
    search_parser.add_argument("query", type=str, nargs="?", help="The query to search")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit the number of search results shown in descending order",
    )

    args = parser.parse_args()

    semantic_search = SemanticSearch()

    match args.command:
        case "verify":
            semantic_search = SemanticSearch()
            semantic_search.verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            with open("data/movies.json", "r") as f:
                j = json.load(f)
                movies = j["movies"]
                semantic_search.load_or_create_embeddings(movies)
            result = semantic_search.search(args.query, args.limit)
            for score, document in result:
                title = document["title"]
                description = document["description"]
                id = document["id"]
                print(f"{id}. {title} (score: {score})")
                print(f"{description}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
