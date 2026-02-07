#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch, embed_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify currently used model")
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedded text")
    embed_parser.add_argument("text", help="Text to use for embed generation")
    args = parser.parse_args()

    semantic_search = SemanticSearch()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
