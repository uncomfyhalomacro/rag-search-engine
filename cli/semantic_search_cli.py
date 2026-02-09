#!/usr/bin/env python3

import argparse
import json
from lib.utils import ordinary_chunker, semantic_chunker
from lib.semantic_search import (
    SemanticSearch,
    embed_text,
    verify_embeddings,
    embed_query_text,
    CHUNK_SIZE,
    MAX_SEM_CHUNK_SIZE,
)

from lib.semantic_search.chunked_semantic_search import ChunkedSemanticSearch


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify currently used model")
    chunk_parser = subparsers.add_parser("chunk", help="Split text in chunks")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE, help="Customise the chunk size"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Customise the overlap size"
    )
    chunk_parser.add_argument("text", help="Text to chunks")
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text in semantic chunks"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=MAX_SEM_CHUNK_SIZE,
        help="Customise the max semantic chunk size",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Customise the overlap size"
    )
    semantic_chunk_parser.add_argument("text", help="Text to chunks")
    subparsers.add_parser("verify_embeddings", help="Verify currently used model")
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query text")
    embed_query_parser.add_argument("query", help="Query text to generate embedding")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedded text")
    embed_parser.add_argument("text", help="Text to use for embed generation")
    _embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Generate embedded semantic chunked text"
    )
    search_parser = subparsers.add_parser("search", help="Semantic search a query")
    search_parser.add_argument("query", type=str, nargs="?", help="The query to search")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit the number of search results shown in descending order",
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Chunked semantic search a query"
    )
    search_chunked_parser.add_argument(
        "query", type=str, nargs="?", help="The query to search"
    )
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit the number of search results shown in descending order",
    )
    args = parser.parse_args()

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
            semantic_search = SemanticSearch()
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
        case "chunk":
            print(
                f"Chunking {len(args.text)} characters with chunk size: {args.chunk_size}"
            )
            chunks = ordinary_chunker(args.text, args.chunk_size, args.overlap)
            for idx, chunk in enumerate(chunks, 1):
                print(f"{idx}. {chunk}")
        case "semantic_chunk":
            print(
                f"Semantically chunking {len(args.text)} characters with chunk size: {args.max_chunk_size}"
            )
            chunks = semantic_chunker(args.text, args.max_chunk_size, args.overlap)
            for idx, chunk in enumerate(chunks, 1):
                chunk = " ".join(chunk[1])
                print(f"{idx}. {chunk}")
        case "embed_chunks":
            cc_search = ChunkedSemanticSearch()
            with open("data/movies.json", "r") as f:
                j = json.load(f)
                movies = j["movies"]
                cc_search.load_or_create_chunk_embeddings(movies)
            embeddings = cc_search.chunk_embeddings
            embeddings = embeddings if embeddings is not None else []
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            cc_search = ChunkedSemanticSearch()
            with open("data/movies.json", "r") as f:
                j = json.load(f)
                movies = j["movies"]
                cc_search.load_or_create_chunk_embeddings(movies)
                results = cc_search.search_chunks(args.query, args.limit)
                for i, result in enumerate(results, 1):
                    TITLE = result["title"]
                    SCORE = result["score"]
                    DESCRIPTION = result["description"]
                    print(f"\n{i}. {TITLE} (score: {SCORE:.4f})")
                    print(f"   {DESCRIPTION}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
