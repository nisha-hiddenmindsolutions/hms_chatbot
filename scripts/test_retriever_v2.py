from app.retrieval.retriever import (
    retrieve_documents,
    get_relevant_documents
)


questions = [
    "What services does Hidden Mind Solutions provide?",
    "Who is the founder of Hidden Mind Solutions?",
    "Who is the CTO?",
    "What is the company's contact number?",
    "What technologies does the company use?",
    "What is the weather in Udaipur?"
]


for question in questions:

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    all_results = retrieve_documents(question, k=5)

    print("\nALL RETRIEVED RESULTS:\n")

    for index, (document, score) in enumerate(all_results, start=1):

        print(f"RESULT {index}")
        print(f"Relevance Score: {score:.4f}")
        print(f"Page: {document.metadata.get('page', 'Unknown')}")
        print(
            f"Preview: "
            f"{document.page_content[:250].replace(chr(10), ' ')}"
        )
        print("-" * 70)

    relevant_documents = get_relevant_documents(
        question,
        k=5,
        threshold=0.35
    )

    print("\nRELEVANT RESULTS AFTER FILTERING:")
    print(f"Total relevant documents: {len(relevant_documents)}")

    if not relevant_documents:
        print("No sufficiently relevant information found.")

    else:
        for index, document in enumerate(
            relevant_documents,
            start=1
        ):
            print(
                f"\nRelevant Result {index} "
                f"(Score: {document.metadata['relevance_score']})"
            )
            print(
                document.page_content[:200].replace("\n", " ")
            )


print("\n" + "=" * 70)
print("RETRIEVER V2 TEST COMPLETED")
print("=" * 70)