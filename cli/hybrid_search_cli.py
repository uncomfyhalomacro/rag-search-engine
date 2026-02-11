import argparse
import os
from sentence_transformers import CrossEncoder
import json
from lib.utils import normalise_scores
from lib.hybrid_search import HYBRID_ALPHA, HYBRID_SEARCH_LIMIT, HybridSearch
from dotenv import load_dotenv
from google import genai


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
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--evaluate", default=False, help="Evaluate search results", action="store_true"
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank method",
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
            query = args.query
            if args.enhance == "spell":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                system_prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{args.query}"

    If no errors, return the original query. Otherwise, return the correction but all should be lowercase."""
                response = client.models.generate_content(
                    model="gemini-2.5-flash", contents=system_prompt
                )
                query = response.text
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{response.text}'\n"
                )
            if args.enhance == "rewrite":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                system_prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Return the rewritten query, all in lowercase"""
            if args.enhance == "expand":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                system_prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"

Return the expanded query, all in lowercase
"""
                response = client.models.generate_content(
                    model="gemini-2.5-flash", contents=system_prompt
                )
                query = response.text
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{response.text}'\n"
                )
            with open("data/movies.json", "r") as f:
                j = json.load(f)
                movies = j["movies"]
                hs = HybridSearch(movies)
                results = hs.rrf_search(query=query, k=args.k_value, limit=args.limit)
                output = ""
                pairs = []
                for i, result in enumerate(results, 1):
                    res = result[1]
                    title = res["title"]
                    bm25_rank = res["bm25_rrf_rank"]
                    semantic_rank = res["semantic_rrf_rank"]
                    rrf_score = res["overall_rrf"]
                    description = result[1]["description"]
                    if args.rerank_method == "cross_encoder":
                        pairs.append([query, f"{title} - {description}"])
                    output += f"{i}. {title}\n"
                    if args.rerank_method == "individual":
                        load_dotenv()
                        api_key = os.environ.get("GEMINI_API_KEY")
                        client = genai.Client(api_key=api_key)
                        system_prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {title} - {description}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the rating in your response, no other text or explanation.
    """
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", contents=system_prompt
                        )
                        llm_score = response.text
                        output += f"\t Rerank Score: {llm_score}/10\n"

                    output += f"\t RRF Score: {rrf_score}\n"
                    output += (
                        f"\t BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}\n"
                    )
                    output += f"\t {description[:100]}\n"
                if args.rerank_method == "batch":
                    load_dotenv()
                    api_key = os.environ.get("GEMINI_API_KEY")
                    client = genai.Client(api_key=api_key)
                    system_prompt = f"""Rank these movies with existing ranks and scoring by relevance to the search query.

Query: "{query}"

Movies with ranks and scoring:
{output}

Return a similar output but with a "Rerank Rank" below the title and sort it by "Rerank Rank". For example:

1. This is a title
	Rerank Rank: 3
	RRF Score: 0.22
	BM25 Rank: 2, Semantic Rank: 10
	This is a description of the movie
"""

                    response = client.models.generate_content(
                        model="gemini-2.5-flash", contents=system_prompt
                    )
                    print(response.text)
                elif args.rerank_method == "cross_encoder":
                    cr = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                    cr_scores = cr.predict(pairs)
                    new_results = zip(cr_scores, results)
                    new_results = sorted(new_results, key=lambda x: x[0], reverse=True)
                    for idx, (cr_score, result) in enumerate(new_results, 1):
                        print(f"{idx}. {result[1]['title']}")
                        print(f"       Cross Encoder Score: {cr_score}")
                        print(f"       RRF Score: {result[1]['overall_rrf']}")
                        print(
                            f"       BM25 Rank: {result[1]['bm25_rrf_rank']}, Semantic Rank: {result[1]['semantic_rrf_rank']}"
                        )
                        print(f"       {result[1]['description']}")
                        print()
                else:
                    print(output)

                if args.evaluate:
                    load_dotenv()
                    api_key = os.environ.get("GEMINI_API_KEY")
                    client = genai.Client(api_key=api_key)
                    system_prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{output}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
                    response = client.models.generate_content(
                        model="gemini-2.5-flash", contents=system_prompt
                    )
                    scores = json.loads(
                        response.text if response.text is not None else "[]"
                    )
                    for i, (score, result) in enumerate(zip(scores, results), 1):
                        res = result[1]
                        title = res["title"]
                        print(f"{i}. {title}: {score}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
