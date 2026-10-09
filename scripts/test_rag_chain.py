from app.retrieval.retriever import get_relevant_documents
from app.rag.rag_chain import build_prompt


def main():

    # Test question
    question = "Who is the founder of Hidden Mind Solutions?"

    print("\nQuestion:")
    print(question)

    # Retrieve relevant documents
    print("\nRetrieving relevant documents...")

    documents = get_relevant_documents(
        query=question,
        k=5
    )

    print(f"\nRetrieved {len(documents)} relevant documents")

    # Build RAG prompt
    prompt = build_prompt(
        question=question,
        documents=documents
    )

    # Display generated prompt
    print("\n" + "=" * 70)
    print("GENERATED RAG PROMPT")
    print("=" * 70)

    print(prompt)

    print("\n" + "=" * 70)
    print("RAG CHAIN TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()