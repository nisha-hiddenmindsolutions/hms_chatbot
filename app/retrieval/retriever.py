import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.retrieval.vector_store import get_embeddings, load_vector_store


DEFAULT_RELEVANCE_THRESHOLD = 0.5
TERM_CORRECTIONS = {
    "adress": "address",
    "addres": "address",
    "addresss": "address",
    "adres": "address",
    "loction": "location",
    "locaton": "location",
    "located": "location",
}
QUERY_SYNONYMS = {
    "address": {"headquarters", "office", "location", "located", "udaipur", "sikar"},
    "headquarters": {"address", "office", "location", "udaipur"},
    "location": {"address", "headquarters", "office", "located", "udaipur", "sikar"},
    "office": {"address", "headquarters", "location", "udaipur"},
    "service": {"services", "offerings"},
    "services": {"service", "offerings"},
    "offerings": {"service", "services"},
    "founder": {"founders", "leadership"},
    "founders": {"founder", "leadership"},
    "cto": {"leadership", "co-founder"},
    "ceo": {"founder", "leadership"},
    "contact": {"email", "phone", "address"},
    "number": {"phone", "contact"},
    "technologies": {"technology", "stack"},
    "technology": {"technologies", "stack"},
    "people": {"team", "employee", "employees", "members", "leadership"},
    "person": {"team", "employee", "employees", "members"},
    "employee": {"team", "employees", "people", "members"},
    "employees": {"team", "employee", "people", "members"},
    "member": {"team", "members", "people", "employees"},
    "members": {"team", "member", "people", "employees"},
    "team": {"people", "employees", "members", "leadership"},
    "working": {"team", "employees", "people", "members"},
}
FACT_TERMS = {
    "address",
    "headquarters",
    "location",
    "office",
    "email",
    "phone",
    "contact",
    "website",
    "founder",
    "cto",
    "ceo",
    "service",
    "services",
    "technology",
    "technologies",
    "people",
    "employee",
    "employees",
    "member",
    "members",
    "team",
    "working",
}
STOP_WORDS = {
    "about",
    "anything",
    "company",
    "does",
    "give",
    "hidden",
    "hms",
    "mind",
    "provide",
    "solutions",
    "tell",
    "that",
    "the",
    "their",
    "this",
    "what",
    "where",
    "which",
    "who",
    "with",
}


def _query_terms(query: str) -> set[str]:
    terms = {
        TERM_CORRECTIONS.get(word, word)
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", query.lower())
        if len(word) > 2 and word not in STOP_WORDS
    }

    expanded_terms = set(terms)

    for term in terms:
        expanded_terms.update(QUERY_SYNONYMS.get(term, set()))

    return expanded_terms


def _lexical_hits(query_terms: set[str], text: str) -> int:
    text = text.lower()

    return sum(
        1
        for term in query_terms
        if term in text
    )


class HMSRetriever:
    """
    Small project retriever wrapper that returns LangChain documents
    while keeping relevance scores in document metadata.
    """

    def __init__(
        self,
        vector_store: Chroma,
        k: int = 6,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.vector_store = vector_store
        self.k = k
        self.threshold = threshold

    def invoke(self, query: str) -> list[Document]:
        terms = _query_terms(query)
        fetch_k = max(self.k * 5, 25)
        search_query = " ".join(sorted(terms)) or query

        results = self.vector_store.similarity_search_with_score(
            query=search_query,
            k=fetch_k,
        )

        ranked_documents = []

        for document, distance in results:
            semantic_score = 1 / (1 + max(distance, 0))
            lexical_score = _lexical_hits(
                terms,
                document.page_content,
            )

            combined_score = semantic_score + (lexical_score * 0.08)

            is_known_fact_query = bool(terms & FACT_TERMS)

            if (
                semantic_score >= self.threshold
                or lexical_score >= 2
                or (is_known_fact_query and lexical_score >= 1)
            ):
                document.metadata["relevance_score"] = round(
                    combined_score,
                    4,
                )
                ranked_documents.append(
                    (
                        combined_score,
                        document,
                    )
                )

        ranked_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            document
            for _, document in ranked_documents[:self.k]
        ]


def retrieve_documents(query, k=5):
    """
    Retrieve the most relevant documents from ChromaDB.

    Parameters:
        query (str): User's question.
        k (int): Number of documents to retrieve.

    Returns:
        list: Relevant documents with similarity scores.
    """

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=k
    )

    return results


def get_relevant_documents(query, k=5, threshold=0.35):
    """
    Retrieve only documents that meet the relevance threshold.
    """

    results = retrieve_documents(query, k)

    relevant_documents = []

    for document, score in results:

        if score >= threshold:

            document.metadata["relevance_score"] = round(score, 4)

            relevant_documents.append(document)

    return relevant_documents


def get_retriever(
    chunks: list[Document],
    k: int = 6,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> HMSRetriever:
    """
    Build an in-memory retriever for the current knowledge-base chunks.

    The app already loads and chunks the PDF at startup. Using an in-memory
    Chroma collection here avoids duplicating persisted records every time
    Streamlit reloads the app.
    """

    if not chunks:
        raise ValueError("No document chunks were provided to the retriever.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="hms_runtime_knowledge_base",
    )

    return HMSRetriever(
        vector_store=vector_store,
        k=k,
        threshold=threshold,
    )
