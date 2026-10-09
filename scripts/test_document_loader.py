from app.ingestion.document_loader import load_documents


documents = load_documents()

print(f"Successfully loaded {len(documents)} documents.")


for index, document in enumerate(documents, start=1):

    print("\n" + "=" * 70)
    print(f"DOCUMENT {index}")
    print("=" * 70)

    print(f"Source: {document.metadata.get('source', 'Unknown')}")
    print(f"Page: {document.metadata.get('page', 'Unknown')}")

    print("\nContent Preview:")
    print(document.page_content[:500])