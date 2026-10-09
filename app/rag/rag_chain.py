from app.rag.prompt import RAG_PROMPT


def create_context(documents):
    """
    Combine retrieved documents into
    a single context string.
    """

    context_parts = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
DOCUMENT {index}
Source: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n\n".join(context_parts)


def build_prompt(question, documents):
    """
    Build the final prompt for the LLM.
    """

    context = create_context(documents)

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    return prompt   