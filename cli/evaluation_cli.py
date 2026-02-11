import argparse
from lib import hybrid_search as hs
from lib.utils import load_golden_dataset, load_movies


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    movies_dataset = load_movies()
    golden_dataset = load_golden_dataset()
    if golden_dataset is None or movies_dataset is None:
        raise Exception("No dataset found")

    movies = movies_dataset.get("movies", [])
    test_cases = golden_dataset.get("test_cases", [])
    hybrid_search = hs.HybridSearch(movies)
    print(f"k={limit}")  # refers to the limit, this is a different k
    for test_case in test_cases:
        query = test_case.get("query", "")
        relevant_docs = test_case.get("relevant_docs", [])
        results = hybrid_search.rrf_search(query, limit=limit, k=60)
        titles = [document.get("title", "") for _, document in results]
        relevant_count = sum([titles.count(rd) for rd in relevant_docs])
        precision = relevant_count / len(titles)
        print(f"Query: {query}")
        print(f"  - Precision@{limit}: {precision:0.4f}")
        print(f"  - Retrieved: {', '.join(titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")


if __name__ == "__main__":
    main()
