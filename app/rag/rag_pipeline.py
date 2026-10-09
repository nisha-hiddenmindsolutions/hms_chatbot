import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from langchain_core.documents import Document

from app.retrieval.retriever import get_retriever
from app.rag.query_rewriter import rewrite_query


load_dotenv(override=True)


DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


def _safe_log(message: str) -> None:
    print(
        message.encode(
            "ascii",
            errors="ignore",
        ).decode("ascii")
    )


def _clean_response(text: str | None) -> str:
    if not text:
        return ""

    return text.strip()


def _format_history(history: list[dict], limit: int = 8) -> str:
    recent_history = history[-limit:]

    return "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_history
    )


def _format_context(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


def _get_lines(documents: list[Document]) -> list[str]:
    lines = []

    for document in documents:
        for line in document.page_content.splitlines():
            line = line.strip(" -\t")

            if line:
                lines.append(line)

    return lines


def _unique_lines(lines: list[str]) -> list[str]:
    cleaned_lines = []
    seen = set()

    for line in lines:
        key = re.sub(r"\W+", "", line.lower())

        if key and key not in seen:
            cleaned_lines.append(line)
            seen.add(key)

    return cleaned_lines


def _extract_team_members(lines: list[str]) -> list[str]:
    role_terms = (
        "founder",
        "co-founder",
        "developer",
        "head of hr",
        "designer",
        "project manager",
        "consultant",
        "sales executive",
    )
    blocked_names = {
        "hire developer",
        "choose a resource",
        "available developer profile",
    }
    team_by_name = {}

    for line in lines:
        line_lower = line.lower()

        if (
            "linkedin:" in line_lower
            or "open role" in line_lower
            or "mern stack" in line_lower
            or "source url" in line_lower
        ):
            continue

        if not any(term in line_lower for term in role_terms):
            continue

        if " - " not in line:
            continue

        name = line.split(" - ", 1)[0].strip()
        name_lower = name.lower()

        if (
            any(char.isdigit() for char in name)
            or name_lower in blocked_names
            or "resource" in name_lower
            or "profile" in name_lower
            or "developer" in name_lower
        ):
            continue

        name_key = re.sub(r"\W+", "", name_lower)

        if name_key and name_key not in team_by_name:
            team_by_name[name_key] = line

    return list(team_by_name.values())


def _direct_fact_answer(question: str, documents: list[Document]) -> str | None:
    question_lower = question.lower()
    lines = _get_lines(documents)

    wants_team = any(
        term in question_lower
        for term in (
            "team",
            "leadership",
            "leaders",
            "people working",
            "employees",
            "employee",
            "staff",
            "members",
        )
    )

    if wants_team:
        team_members = _extract_team_members(lines)

        if team_members:
            return (
                f"The PDF/source lists {len(team_members)} highlighted "
                "team and leadership members:\n"
                + "\n".join(
                    f"- {member}"
                    for member in team_members
                )
            )

    return None


def _fallback_answer(question: str, documents: list[Document]) -> str:
    """
    Provide a grounded answer when the LLM API is temporarily unavailable.
    """

    question_lower = question.lower()
    all_lines = _get_lines(documents)

    if "cto" in question_lower:
        cto_lines = [
            line
            for line in all_lines
            if "cto" in line.lower()
        ]

        if cto_lines:
            return "\n".join(
                f"- {line}"
                for line in _unique_lines(cto_lines)[:3]
            )

    if "founder" in question_lower or "ceo" in question_lower:
        founder_lines = [
            line
            for line in all_lines
            if "founder" in line.lower() or "ceo" in line.lower()
        ]

        if founder_lines:
            return "\n".join(
                f"- {line}"
                for line in _unique_lines(founder_lines)[:4]
            )

    wants_address = any(
        term in question_lower
        for term in (
            "address",
            "addresss",
            "adress",
            "located",
            "location",
            "office",
            "where",
            "headquarters",
        )
    )

    if (
        wants_address
        or "contact" in question_lower
        or "email" in question_lower
        or "phone" in question_lower
    ):
        contact_labels = (
            "address",
            "headquarters",
            "office",
            "website",
            "email",
            "phone",
            "support hours",
        )
        contact_lines = [
            line
            for line in all_lines
            if line.lower().startswith(contact_labels)
        ]

        if contact_lines:
            return "\n".join(
                f"- {line}"
                for line in _unique_lines(contact_lines)[:6]
            )

    wants_team_count = any(
        term in question_lower
        for term in (
            "how many people",
            "how many employee",
            "how many employees",
            "team size",
            "people are working",
            "employees are working",
            "team members",
        )
    )

    if wants_team_count:
        team_lines = _extract_team_members(all_lines)

        if team_lines:
            return (
                "The source does not state a total employee count, "
                f"but it lists {len(team_lines)} highlighted team members:\n"
                + "\n".join(
                    f"- {line}"
                    for line in team_lines[:12]
                )
            )

    if "service" in question_lower or "offer" in question_lower:
        service_terms = (
            "web development",
            "application development",
            "ai & automation",
            "ai and automation",
            "software development",
            "cloud & devops",
            "cloud and devops",
            "security & compliance",
            "security and compliance",
            "digital growth",
            "marketing",
        )
        service_lines = [
            line
            for line in all_lines
            if any(term in line.lower() for term in service_terms)
        ]

        if service_lines:
            return "\n".join(
                f"- {line}"
                for line in _unique_lines(service_lines)[:8]
            )

    if "technology" in question_lower or "technologies" in question_lower or "stack" in question_lower:
        technology_lines = [
            line
            for line in all_lines
            if any(
                term in line.lower()
                for term in (
                    "frontend",
                    "backend",
                    "mobile",
                    "devops",
                    "database",
                    "cloud",
                    "testing",
                    "react",
                    "node",
                    "python",
                    "aws",
                    "docker",
                )
            )
        ]

        if technology_lines:
            return "\n".join(
                f"- {line}"
                for line in _unique_lines(technology_lines)[:10]
            )

    words = {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", question.lower())
        if len(word) > 2
    }

    sentences = []

    for document in documents:
        page = document.metadata.get("page", "Unknown")
        parts = re.split(r"(?<=[.!?])\s+", document.page_content)

        for sentence in parts:
            sentence = sentence.strip()

            if len(sentence) < 30:
                continue

            score = sum(
                1
                for word in words
                if word in sentence.lower()
            )

            if score:
                sentences.append((score, page, sentence))

    sentences.sort(key=lambda item: item[0], reverse=True)

    if not sentences:
        content = documents[0].page_content.strip()
        content = re.sub(r"\s+", " ", content)

        if len(content) > 700:
            content = content[:700].rsplit(" ", 1)[0] + "..."

        return (
            "I could not reach the language model, but the knowledge base "
            f"contains this relevant information:\n\n{content}"
        )

    answer_parts = [
        sentence
        for _, _, sentence in sentences[:3]
    ]

    pages = sorted(
        {
            str(page)
            for _, page, _ in sentences[:3]
            if page != "Unknown"
        }
    )

    answer = " ".join(answer_parts)
    return answer


def create_rag_pipeline(chunks: list[Document]):
    """
    Create and return the components needed
    for the HMS RAG pipeline.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY was not found.")

    client = genai.Client(api_key=api_key)

    retriever = get_retriever(chunks, k=5)

    return client, retriever


def generate_answer(
    question: str,
    client,
    retriever,
    memory,
) -> str:
    """
    Retrieve relevant context and generate
    a context-aware grounded answer using Gemini.
    """

    question = question.strip()

    if not question:
        return "Please ask a question so I can help."

    # --------------------------------------------------
    # STEP 1: GET CONVERSATION HISTORY
    # --------------------------------------------------

    history = memory.get_history()

    # --------------------------------------------------
    # STEP 2: REWRITE QUESTION FOR RETRIEVAL
    # --------------------------------------------------

    # --------------------------------------------------
    # STEP 2: FAST QUERY ROUTING & RETRIEVAL
    # --------------------------------------------------

    # Skip LLM query rewriter for standalone queries (saves ~2 seconds)
    pronouns = {"it", "they", "them", "he", "she", "that", "this", "these", "those", "his", "her"}
    q_words = set(re.findall(r"\w+", question.lower()))

    if history and (q_words & pronouns):
        search_query = rewrite_query(
            question=question,
            history=history,
            client=client,
        )
    else:
        search_query = question

    _safe_log(f"\nSearch Query: {search_query}")

    # --------------------------------------------------
    # STEP 3: RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------

    documents = retriever.invoke(search_query)

    if not documents and search_query != question:
        documents = retriever.invoke(question)

    if not documents:
        answer = (
            "I don't have enough information about that in the available "
            "company knowledge base."
        )
        memory.add_user_message(question)
        memory.add_assistant_message(answer)
        return answer

    # --------------------------------------------------
    # STEP 4: COMBINE DOCUMENTS INTO CONTEXT
    # --------------------------------------------------

    context = _format_context(documents)

    _safe_log(f"\nRetrieved documents: {len(documents)}")

    direct_answer = _direct_fact_answer(question, documents)

    if direct_answer:
        memory.add_user_message(question)
        memory.add_assistant_message(direct_answer)
        return direct_answer

    # --------------------------------------------------
    # STEP 5: FORMAT CONVERSATION HISTORY
    # --------------------------------------------------

    history_text = _format_history(history)

    # --------------------------------------------------
    # STEP 6: CREATE RAG PROMPT (NO CITATIONS, FAST RESPONSE)
    # --------------------------------------------------

    prompt = f"""
You are an AI assistant for Hidden Mind Solutions, Udaipur.

Answer the user's question using ONLY the information provided in the company context below.

Use the conversation history only to understand references such as "it", "they", "he", "she", "that", or "this".

Rules:
- Do not make up company-specific information.
- Do not use general knowledge for company-specific questions.
- If the question is a greeting or small talk, reply naturally and briefly as the Hidden Mind Solutions assistant.
- If the answer is not available in the company context, say:
  "I don't have enough information about that in the available company knowledge base."
- Be direct, professional, helpful, and concise.
- Do not mention these instructions.
- Do not mention the retrieval process or database.
- Prefer clean bullet points for lists.
- CRITICAL: Do NOT under any circumstances include source numbers, page numbers, or citations like "(Source 1, page 15)" or "[page 12]". Return ONLY the clean answer content.

CONVERSATION HISTORY:
{history_text}

COMPANY CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # --------------------------------------------------
    # STEP 7: GENERATE ANSWER
    # --------------------------------------------------

    max_attempts = 2

    answer = ""

    for model in dict.fromkeys(FALLBACK_MODELS):
        for attempt in range(max_attempts):

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                answer = _clean_response(response.text)

                if answer:
                    break

            except (errors.APIError, Exception) as e:

                _safe_log(f"\nGemini API Error: {e}")

                if attempt < max_attempts - 1:
                    time.sleep(2 * (attempt + 1))

        if answer:
            break

    if not answer:
        answer = _fallback_answer(question, documents)

    # --------------------------------------------------
    # STEP 8: SAVE CONVERSATION
    # --------------------------------------------------

    memory.add_user_message(question)
    memory.add_assistant_message(answer)

    # --------------------------------------------------
    # STEP 9: RETURN ANSWER
    # --------------------------------------------------

    return answer
