from app.ingestion.document_loader import load_documents
from app.ingestion.document_cleaner import clean_documents
from app.ingestion.document_chunker import chunk_documents


# ============================================================
# STEP 1: LOAD PDF DOCUMENTS
# ============================================================

documents = load_documents()

print(f"\nOriginal documents/pages: {len(documents)}")


# ============================================================
# STEP 2: CLEAN DOCUMENTS
# ============================================================

cleaned_documents = clean_documents(documents)

print(f"Cleaned documents/pages: {len(cleaned_documents)}")


# ============================================================
# STEP 3: SPLIT INTO CHUNKS
# ============================================================

chunks = chunk_documents(
    cleaned_documents,
    chunk_size=1000,
    chunk_overlap=150,
)

print(f"Total chunks created: {len(chunks)}")

print("=" * 60)


# ============================================================
# STEP 4: DISPLAY CHUNKS
# ============================================================

for index, chunk in enumerate(chunks, start=1):

    print(f"\nChunk {index}")

    # PDF metadata
    source = chunk.metadata.get(
        "source",
        "Unknown",
    )

    page = chunk.metadata.get(
        "page",
        "Unknown",
    )

    print(f"Source: {source}")
    print(f"Page: {page}")
    print(f"Characters: {len(chunk.page_content)}")

    print("Preview:")

    print(
        chunk.page_content[:300]
        .replace("\n", " ")
    )

    print("-" * 60)