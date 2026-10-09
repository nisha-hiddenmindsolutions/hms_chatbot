import os

from google.genai import errors


QUERY_REWRITER_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)


def rewrite_query(
    question: str,
    history: list[dict],
    client,
) -> str:
    """
    Convert a conversational question into a standalone
    question for document retrieval.
    """

    # No conversation history means the question
    # is already standalone.
    if not history:
        return question

    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in history
    )

    prompt = f"""
You are a query rewriting assistant for a RAG chatbot
about Hidden Mind Solutions.

Rewrite the latest user question into a standalone
search query for the Hidden Mind Solutions knowledge base.

Use the conversation history to understand references such as:
- it
- they
- them
- he
- she
- this
- that
- these
- those
-CTO
-CEO

Rules:
- Keep the original meaning.
- Do not answer the question.
- Do not add information that is not present in the conversation.
- If the question is already standalone, return it unchanged.
- Return ONLY the rewritten search query.
- Do not add explanations.

CONVERSATION HISTORY:
{history_text}

LATEST USER QUESTION:
{question}

STANDALONE SEARCH QUERY:
"""

    try:
        response = client.models.generate_content(
            model=QUERY_REWRITER_MODEL,
            contents=prompt,
        )

        rewritten_query = response.text.strip()

        if not rewritten_query:
            return question

        return rewritten_query

    except (errors.APIError, Exception) as e:
        print("\nQuery Rewriter API Error:")
        print("=" * 60)
        print(e)
        print("=" * 60)

        return question
