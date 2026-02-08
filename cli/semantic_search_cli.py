#!/usr/bin/env python3
from mpmath.libmp.libmpi import MAX
from torch.cuda import init
import re

import argparse
import json
from lib.semantic_search import (
    SemanticSearch,
    embed_text,
    verify_embeddings,
    embed_query_text,
    CHUNK_SIZE,
    MAX_SEM_CHUNK_SIZE
)

def ordinary_chunker(text, size=CHUNK_SIZE, overlap=0):
    chunks_storer = []
    initial_chunks = text.split(None, maxsplit=size)
    while len(initial_chunks) > 0:
        if len(initial_chunks) == 1:
            chunks_storer[-1] += " " + initial_chunks.pop()
        else:
            chunk = " ".join(initial_chunks[:-1])
            chunks_storer.append(chunk)
            initial_chunks = initial_chunks[-1].split(
                None, maxsplit=size
            )
            if overlap > 0:
                if len(initial_chunks) > overlap:
                    overlap = chunk.rsplit(None, maxsplit=overlap)[1:]
                    overlap.extend(initial_chunks)
                    initial_chunks = overlap

    return chunks_storer

def semantic_chunker(text, size=MAX_SEM_CHUNK_SIZE, overlap=0):
    splits = re.split(r"(?<=[.!?])\s+", text)
    chunks_storer = []
    start = 0
    for i in range(len(splits)):
        if start + size <= len(splits):
            if overlap > 0 and start+size+overlap <= len(splits):
                chunk = splits[start:start+size+overlap-1]
                chunks_storer.append(" ".join(chunk))
            else:
                chunk = splits[start:start+size]
                chunks_storer.append(" ".join(chunk))
            start += size - overlap
        else:
            chunk = []
            if overlap > 0:
                chunk = splits[start:-overlap]
            else:
                chunk = splits[start:]
            if len(chunk) > 0:
                chunks_storer.append(" ".join(chunk))
            break


    return chunks_storer


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
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text in semantic chunks")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=MAX_SEM_CHUNK_SIZE, help="Customise the max semantic chunk size"
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
    search_parser = subparsers.add_parser("search", help="Semantic search a query")
    search_parser.add_argument("query", type=str, nargs="?", help="The query to search")
    search_parser.add_argument(
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
                print(f"{idx}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
