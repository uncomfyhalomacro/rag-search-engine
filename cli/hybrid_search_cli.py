import argparse
import json
from lib.utils import normalise_scores
from lib.hybrid_search import HYBRID_ALPHA, HYBRID_SEARCH_LIMIT, HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Score params")
    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Weighted search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=HYBRID_ALPHA, help="Adjustable alpha value"
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=HYBRID_SEARCH_LIMIT,
        help="Adjustable search limit value",
    )
    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument(
        "-k", "--k-value", type=int, default=60, help="K value"
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=HYBRID_SEARCH_LIMIT,
        help="Adjustable search limit value",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalise_scores(args.scores)
            for score in scores:
                print(f"* {score}")

        case "weighted-search":
            with open("data/movies.json", "r") as f:
                j = json.load(f)
                movies = j["movies"]
                hs = HybridSearch(movies)
                results = hs.weighted_search(
                    query=args.query, alpha=args.alpha, limit=args.limit
                )
                for i, result in enumerate(results, 1):
                    title = result[1]["title"]
                    hybrid_score = result[1]["hybrid_score"]
                    bm25_score = result[1]["bm25_score"]
                    semantic_score = result[1]["semantic_score"]
                    description = result[1]["description"]
                    print(f"{i}. {title}")
                    print(f"\t Hybrid Score: {hybrid_score}")
                    print(f"\t BM25 Score: {bm25_score}, Semantic: {semantic_score}")
                    print(f"\t {description}")

        case "rrf-search":
            with open("data/movies.json", "r") as f:
                j = json.load(f)
                movies = j["movies"]
                hs = HybridSearch(movies)
                results = hs.rrf_search(
                    query=args.query, k=args.k_value, limit=args.limit
                )
                for i, result in enumerate(results, 1):
                    res = result[1]
                    title = res["title"]
                    bm25_rank = res["bm25_rrf_rank"]
                    semantic_rank = res["semantic_rrf_rank"]
                    rrf_score = res["overall_rrf"]
                    description = result[1]["description"][:100]
                    print(f"{i}. {title}")
                    print(f"\t RRF Score: {rrf_score}")
                    print(f"\t BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                    print(f"\t {description}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
