from app.ingestion.document_loader import load_documents


documents = load_documents()

print(f"\nSuccessfully loaded {len(documents)} documents.\n")
print("=" * 60)

for document in documents:
    print(f"\nFile: {document.metadata['filename']}")
    print(f"Title: {document.metadata.get('title', 'Not available')}")
    print(f"Category: {document.metadata.get('category', 'Not available')}")
    print(f"Source URL: {document.metadata.get('source_url', 'Not available')}")
    print("\nContent preview:")
    print(document.page_content[:150].replace("\n", " "))
    print("-" * 60)