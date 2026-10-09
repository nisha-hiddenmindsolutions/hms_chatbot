from app.ingestion.document_loader import load_documents
from app.ingestion.document_cleaner import clean_documents
from app.ingestion.document_chunker import chunk_documents
from app.retrieval.vector_store import create_vector_store


# ============================================================
# PREPARE KNOWLEDGE BASE
# ============================================================

documents = load_documents()

print(f"Loaded {len(documents)} PDF pages")

cleaned_documents = clean_documents(documents)

print(f"Cleaned {len(cleaned_documents)} PDF pages")

chunks = chunk_documents(cleaned_documents)

print(f"Created {len(chunks)} chunks")


# ============================================================
# CREATE VECTOR STORE
# ============================================================

print("\nInitializing vector store...")

vector_store = create_vector_store(chunks)

print("Vector store ready.")


# ============================================================
# TEST QUESTIONS
# ============================================================

test_questions = [
    "Who is the CTO of Hidden Mind Solutions?",
    "Who is the founder of Hidden Mind Solutions?",
    "What services does Hidden Mind Solutions provide?",
    "What is the company's email address?",
    "What is the company's office address?",
]


# ============================================================
# SEARCH WITH SCORES
# ============================================================

for question in test_questions:

    print("\n")
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    results = vector_store.similarity_search_with_score(
        question,
        k=10,
    )

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 80)

        print(f"RESULT {index}")
        print(f"Score: {score:.4f}")

        print(
            f"Source: "
            f"{document.metadata.get('source', 'unknown')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page', 'unknown')}"
        )

        print(
            f"Characters: "
            f"{len(document.page_content)}"
        )

        print("\nContent:")
        print(document.page_content[:500])