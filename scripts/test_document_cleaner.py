
from app.ingestion.document_loader import load_documents
from app.ingestion.document_cleaner import clean_documents


documents = load_documents()
cleaned_documents = clean_documents(documents)

print(f"Original documents: {len(documents)}")
print(f"Cleaned documents: {len(cleaned_documents)}\n")

for original, cleaned in zip(documents, cleaned_documents):
    print(f"File: {original.metadata['filename']}")
    print(f"Original characters: {len(original.page_content)}")
    print(f"Cleaned characters: {len(cleaned.page_content)}")
    print("-" * 50)