from app.ingestion.document_loader import load_documents
from app.ingestion.document_cleaner import clean_documents
from app.ingestion.document_chunker import chunk_documents
from app.memory.conversation_memory import ConversationMemory
from app.rag.rag_pipeline import create_rag_pipeline, generate_answer


# Step 1: Load documents
documents = load_documents()

print(f"Loaded {len(documents)} documents")


# Step 2: Clean documents
cleaned_documents = clean_documents(documents)


# Step 3: Create chunks
chunks = chunk_documents(cleaned_documents)

print(f"Created {len(chunks)} chunks")


# Step 4: Create RAG pipeline
print("\nInitializing RAG pipeline...")

client, retriever = create_rag_pipeline(chunks)


# Step 5: Create conversation memory
memory = ConversationMemory()


# Step 6: Test conversation
questions = [
    "Who is the founder of Hidden Mind Solutions?",
    "What other information do you have about him?",
    "What services does the company provide?",
    "What is the weather in Udaipur?",
    "Who is the CTO of Hidden Mind Solutions?",
]


# Step 7: Ask questions
for question in questions:

    print("\n" + "=" * 70)
    print(f"USER: {question}")
    print("=" * 70)

    answer = generate_answer(
        question=question,
        client=client,
        retriever=retriever,
        memory=memory,
    )

    print(f"ASSISTANT: {answer}")