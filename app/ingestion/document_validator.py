from langchain_core.documents import Document


PLACEHOLDER_KEYWORDS = [
    "not yet available",
    "todo",
    "tbd",
    "placeholder",
]


def validate_documents(
    documents: list[Document],
    min_content_length: int = 100,
) -> list[Document]:
    """
    Validate documents before processing them for the RAG pipeline.

    Args:
        documents: List of LangChain Document objects.
        min_content_length: Minimum recommended content length.

    Returns:
        The original list of documents after reporting validation issues.
    """

    print("\nDocument Validation Report")
    print("=" * 50)

    valid_count = 0

    for document in documents:
        filename = document.metadata.get("filename", "Unknown file")
        content = document.page_content.strip()
        issues = []

        if not content:
            issues.append("Document is empty")

        elif len(content) < min_content_length:
            issues.append(
                f"Document is very short ({len(content)} characters)"
            )

        content_lower = content.lower()

        for keyword in PLACEHOLDER_KEYWORDS:
            if keyword in content_lower:
                issues.append(
                    f"Contains placeholder keyword: '{keyword}'"
                )

        if issues:
            print(f"\n {filename}")
            for issue in issues:
                print(f"   - {issue}")
        else:
            valid_count += 1
            print(f" {filename} — looks good")

    print("\n" + "=" * 50)
    print(f"Total documents: {len(documents)}")
    print(f"Documents without issues: {valid_count}")

    return documents