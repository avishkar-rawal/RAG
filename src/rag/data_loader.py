import json
from pathlib import Path
from typing import Any, Callable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


def _load_json(path: Path) -> List[Document]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return [
        Document(
            page_content=json.dumps(payload, ensure_ascii=False, indent=2),
            metadata={"source": str(path)},
        )
    ]


LOADERS: dict[str, Callable[..., Any]] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".json": _load_json,
}

def load_all_documents(data_dir: str) -> List[Document]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents: List[Document] = []
    files = [path for path in data_path.glob("**/*") if path.is_file() and path.suffix.lower() in LOADERS]
    print(f"[DEBUG] Found {len(files)} supported files in {data_path}")

    for file_path in files:
        loader_factory = LOADERS[file_path.suffix.lower()]
        print(f"[DEBUG] Loading {file_path}")
        try:
            loader = loader_factory(file_path) if file_path.suffix.lower() == ".json" else loader_factory(str(file_path))
            loaded = loader if isinstance(loader, list) else loader.load()
            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} documents from {file_path}")
        except Exception as error:
            print(f"[ERROR] Failed to load {file_path}: {error}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)