import argparse


def main() -> None:
	from .search import RAGSearch

	parser = argparse.ArgumentParser(description="Query the local RAG vector store.")
	parser.add_argument("query", help="Question to ask about the indexed documents")
	parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
	args = parser.parse_args()

	search = RAGSearch()
	print(search.search_and_summarize(args.query, top_k=args.top_k))


__all__ = ["main"]
