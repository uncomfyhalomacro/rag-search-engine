import argparse
import os
import json
from lib.utils import load_movies
from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform summarization on search results (search + generate answer)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit number of documents or search results to n")

    args = parser.parse_args()
    movies_dataset = load_movies()
    if movies_dataset is None:
        raise Exception("No movies dataset")

    movies = movies_dataset.get("movies", [])
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    match args.command:
        case "rag":
            query = args.query
            hs = HybridSearch(documents=movies)
            results  = hs.rrf_search(query, k=60, limit=5)
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{results}

Provide a comprehensive answer that addresses the query:"""

            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            print("Search Results:")
            for doc_id, document in results:
                title = document.get("title", "")
                print(f"\t- {title}")
            print("RAG Response:")
            print(response.text)

        case "summarize":
            limit = args.limit
            query = args.query
            hs = HybridSearch(documents=movies)
            results  = hs.rrf_search(query, k=60, limit=limit)
            prompt=f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            print("Search Results:")
            for doc_id, document in results:
                title = document.get("title", "")
                print(f"\t- {title}")
            print("LLM Summary:")
            print(response.text)


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
