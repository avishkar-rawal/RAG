import os
from pathlib import Path
from dotenv import load_dotenv
from .vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        data_dir: str = "data",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str | None = None,
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.llm_model = llm_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.llm = None
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from .data_loader import load_all_documents

            docs = load_all_documents(data_dir)
            if not docs:
                raise ValueError(f"No supported documents found in {Path(data_dir).resolve()}")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

    def _get_llm(self) -> ChatGroq:
        if self.llm is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise RuntimeError("GROQ_API_KEY is required to summarize search results.")
            self.llm = ChatGroq(api_key=groq_api_key, model=self.llm_model)
            print(f"[INFO] Groq LLM initialized: {self.llm_model}")
        return self.llm

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the most relevant chunks without invoking an LLM."""
        return self.vectorstore.query(query, top_k=top_k)

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self._get_llm().invoke([prompt])
        return str(response.content)

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)