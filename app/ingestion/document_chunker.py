from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split PDF documents into smaller chunks for RAG retrieval.

    Args:
        documents: Documents loaded from the knowledge-base PDF.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        A list of chunked Document objects.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    return chunks