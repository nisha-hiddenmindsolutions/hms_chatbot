from app.ingestion.document_loader import load_documents
from app.ingestion.document_cleaner import clean_documents
from app.ingestion.document_chunker import chunk_documents
from app.retrieval.retriever import get_retriever


# ============================================================
# LOAD PDF
# ============================================================

documents = load_documents()

print(f"Loaded {len(documents)} PDF pages")


# ============================================================
# CLEAN
# ============================================================

cleaned_documents = clean_documents(documents)

print(f"Cleaned {len(cleaned_documents)} PDF pages")


# ============================================================
# CHUNK
# ============================================================

chunks = chunk_documents(
    cleaned_documents,
    chunk_size=1000,
    chunk_overlap=150,
)

print(f"Created {len(chunks)} chunks")


# ============================================================
# CREATE RETRIEVER
# ============================================================

print("\nInitializing retriever...")

retriever = get_retriever(
    chunks,
    k=5,
)

print("Retriever ready.")


# ============================================================
# TEST QUESTIONS
# ============================================================

test_questions = [
    "What services does Hidden Mind Solutions provide?",
    "Tell me about the company's portfolio projects.",
    "Are there any career opportunities?",
    "Who is the founder of Hidden Mind Solutions?",
    "Who is the CTO of Hidden Mind Solutions?",
    "What is the company's office address?",
    "What is the contact number?",
    "What is the company's email address?",
    "What technologies does Hidden Mind Solutions use?",
    "What is the weather in Udaipur?",
]


# ============================================================
# TEST RETRIEVAL
# ============================================================

for question in test_questions:

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    results = retriever.invoke(question)

    if not results:
        print("❌ No documents retrieved.")
        continue

    for index, document in enumerate(
        results,
        start=1,
    ):

        print(f"\nRESULT {index}")
        print("-" * 70)

        print(
            "Source:",
            document.metadata.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Page:",
            document.metadata.get(
                "page",
                "Unknown",
            ),
        )

        print(
            "Characters:",
            len(document.page_content),
        )

        print("\nPreview:")

        print(
            document.page_content[:500]
            .replace("\n", " ")
        )