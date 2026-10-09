import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.document_loader import load_documents
from app.ingestion.document_cleaner import clean_documents
from app.ingestion.document_chunker import chunk_documents
from app.memory.conversation_memory import ConversationMemory
from app.rag.rag_pipeline import create_rag_pipeline, generate_answer


def run_all_pdf_tests():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE KNOWLEDGE BASE PDF RAG TEST SUITE")
    print("=" * 80)

    # 1. Load and process documents
    print("\n1. Loading knowledge base PDF...")
    raw_docs = load_documents()
    print(f"   Loaded {len(raw_docs)} pages.")

    print("\n2. Cleaning documents...")
    cleaned_docs = clean_documents(raw_docs)

    print("\n3. Chunking documents...")
    chunks = chunk_documents(cleaned_docs)
    print(f"   Created {len(chunks)} chunks.")

    # 4. Initialize RAG pipeline
    print("\n4. Initializing RAG pipeline with Chroma vector store...")
    client, retriever = create_rag_pipeline(chunks)
    memory = ConversationMemory()

    # 5. Define test cases covering all sections + out-of-context queries
    test_cases = [
        {
            "category": "Company Identity & Headquarters",
            "question": "Where is Hidden Mind Solutions located and what is its official website?",
            "expected_keywords": ["Udaipur", "hiddenmindsolutions.in"],
            "allow_out_of_context": False,
        },
        {
            "category": "Leadership & Founder",
            "question": "Who is the founder and CEO of Hidden Mind Solutions?",
            "expected_keywords": ["Himanshu Sanadhya", "Founder"],
            "allow_out_of_context": False,
        },
        {
            "category": "Co-Founder & CTO",
            "question": "Who is the CTO of Hidden Mind Solutions?",
            "expected_keywords": ["Tushar Vaghela", "CTO"],
            "allow_out_of_context": False,
        },
        {
            "category": "Services Offered",
            "question": "What services does Hidden Mind Solutions offer to its clients?",
            "expected_keywords": ["Web Development", "AI", "Automation"],
            "allow_out_of_context": False,
        },
        {
            "category": "Tech Stack",
            "question": "What technologies and frameworks does Hidden Mind Solutions use?",
            "expected_keywords": ["React", "AWS", "Python"],
            "allow_out_of_context": False,
        },
        {
            "category": "Portfolio & Projects",
            "question": "Tell me about the featured projects delivered by Hidden Mind Solutions.",
            "expected_keywords": ["Revolution Realty Capital", "FindYourSaaS"],
            "allow_out_of_context": False,
        },
        {
            "category": "Careers & Job Openings",
            "question": "What job positions is Hidden Mind Solutions hiring for?",
            "expected_keywords": ["MERN Stack Developer", "Udaipur"],
            "allow_out_of_context": False,
        },
        {
            "category": "Payment & Pricing Policies",
            "question": "What are the payment terms and GST rate for clients working with HMS?",
            "expected_keywords": ["advance", "18% GST"],
            "allow_out_of_context": False,
        },
        {
            "category": "Out-of-Context Question 1 (General Knowledge)",
            "question": "What is the capital city of France?",
            "expected_keywords": ["don't have enough information"],
            "allow_out_of_context": True,
        },
        {
            "category": "Out-of-Context Question 2 (Weather)",
            "question": "What is the current weather forecast for Tokyo today?",
            "expected_keywords": ["don't have enough information"],
            "allow_out_of_context": True,
        },
        {
            "category": "Out-of-Context Question 3 (Cooking)",
            "question": "How do I bake a chocolate cake at home?",
            "expected_keywords": ["don't have enough information"],
            "allow_out_of_context": True,
        },
    ]

    passed_count = 0
    total_count = len(test_cases)

    for idx, test in enumerate(test_cases, 1):
        question = test["question"]
        category = test["category"]
        expected = test["expected_keywords"]
        allow_ooc = test["allow_out_of_context"]

        print("\n" + "-" * 75)
        print(f"TEST {idx}/{total_count} [{category}]")
        print(f"QUESTION: {question}")

        answer = generate_answer(
            question=question,
            client=client,
            retriever=retriever,
            memory=memory,
        )

        print(f"ANSWER:\n{answer}")

        # Validation check
        answer_lower = answer.lower()
        if allow_ooc:
            is_ooc_ok = "don't have" in answer_lower or "not available" in answer_lower
            if is_ooc_ok:
                print(">> RESULT: PASSED (Correctly recognized as out-of-context)")
                passed_count += 1
            else:
                print(">> RESULT: FAILED (Should have returned out-of-context message)")
        else:
            found_any = any(kw.lower() in answer_lower for kw in expected)
            if found_any:
                print(f">> RESULT: PASSED (Found expected context info: {expected})")
                passed_count += 1
            else:
                print(f">> RESULT: FAILED (Expected one of {expected} in answer)")

        import time
        time.sleep(2)

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed_count}/{total_count} TESTS PASSED ({passed_count/total_count*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_pdf_tests()
