#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify currently used model")
    args = parser.parse_args()

    semantic_search = SemanticSearch()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
