from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Location where ChromaDB will store data
VECTOR_DB_PATH = BASE_DIR / "data" / "vector_db"


def get_embeddings():
    """
    Initialize and return the embedding model.
    Downloads model if not already cached locally.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "local_files_only": True,
            },
        )
    except Exception:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

    return embeddings


def create_vector_store(chunks):
    """
    Create and persist a ChromaDB vector store
    using the provided document chunks.
    """

    print("Initializing embedding model...")

    embeddings = get_embeddings()

    print("Creating vector database...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
        collection_name="hms_knowledge_base"
    )

    print("Vector database created successfully.")
    print(f"Location: {VECTOR_DB_PATH}")

    return vector_store


def load_vector_store():
    """
    Load an existing ChromaDB vector store.
    """

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="hms_knowledge_base",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH)
    )

    return vector_store
