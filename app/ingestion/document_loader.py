from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


PDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "HiddenMindSolutions_Website_KnowledgeBase.pdf"
)


def load_documents() -> list[Document]:
    """
    Load the Hidden Mind Solutions knowledge base PDF.
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base PDF not found: {PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))

    documents = loader.load()

    return documents
