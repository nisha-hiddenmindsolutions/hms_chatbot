from app.ingestion.document_loader import load_documents
from app.ingestion.document_cleaner import clean_documents
from app.ingestion.document_chunker import chunk_documents

from app.retrieval.vector_store import create_vector_store


print("\nLoading documents...")

documents = load_documents()

print(f"Loaded {len(documents)} PDF pages")


print("\nCleaning documents...")

cleaned_documents = clean_documents(documents)

print(f"Cleaned {len(cleaned_documents)} PDF pages")


print("\nCreating chunks...")

chunks = chunk_documents(cleaned_documents)

print(f"Created {len(chunks)} chunks")


print("\nCreating vector database...")

vector_store = create_vector_store(chunks)


print("\n" + "=" * 60)
print("VECTOR STORE TEST COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nTotal chunks stored: {len(chunks)}")