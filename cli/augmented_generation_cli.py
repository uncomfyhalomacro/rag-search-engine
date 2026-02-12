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
        "summarize",
        help="Perform summarization on search results (search + generate answer)",
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of documents or search results to n",
    )
    citations_parser = subparsers.add_parser(
        "citations",
        help="Perform summarization on search results (search + generate answer)",
    )
    citations_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of documents or search results to n",
    )
    question_parser = subparsers.add_parser(
        "question",
        help="Perform summarization on search results (search + generate answer)",
    )
    question_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of documents or search results to n",
    )

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
            results = hs.rrf_search(query, k=60, limit=5)
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
            results = hs.rrf_search(query, k=60, limit=limit)
            prompt = f"""
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

        case "citations":
            limit = args.limit
            query = args.query
            hs = HybridSearch(documents=movies)
            results = hs.rrf_search(query, k=60, limit=limit)
            documents = [document for _, document in results]
            prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
            print("Search Results:")
            for document in documents:
                title = document.get("title", "")
                print(f"\t- {title}")
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            print("LLM Answer:")
            print(response.text)
        case "question":
            limit = args.limit
            query = args.query
            hs = HybridSearch(documents=movies)
            results = hs.rrf_search(query, k=60, limit=limit)
            documents = [document for _, document in results]
            _documents = [
                f"Title: {document.get('title', '')}\nDescription: {document.get('description', '')}"
                for _, document in results
            ]
            context = "\n".join(_documents)
            prompt = f"""Answer the following question based on the provided documents.

Question: {query}

Documents:
{context}

General instructions:
- Answer directly and concisely
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
- Cite sources when possible
- You can find more information in the description section

Guidance on types of questions:
- Factual questions: Provide a direct answer
- Analytical questions: Compare and contrast information from the documents
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view

Answer:"""
            print("Search Results:")
            for document in documents:
                title = document.get("title", "")
                print(f"\t- {title}")
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            print("Answer:")
            print(response.text)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
