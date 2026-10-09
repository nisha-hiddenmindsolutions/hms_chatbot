from app.ingestion.document_loader import load_documents
from app.ingestion.document_validator import validate_documents


documents = load_documents()
validate_documents(documents)