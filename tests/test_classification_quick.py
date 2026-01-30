"""
Test classification with gemini-1.5-flash
"""
import sys
sys.path.insert(0, '..')

from src.indexer import DocumentIndexer
from src.rag_qa import RAGQAEngine

# Initialize
indexer = DocumentIndexer()
qa_engine = RAGQAEngine(indexer.vector_store)

# Test questions
test_cases = [
    ("Hello", "CASUAL"),
    ("Hi, how are you?", "CASUAL"),
    ("Who are you?", "CASUAL"),
    ("What was NVIDIA's Q3 revenue?", "DOCUMENT_QUERY"),
    ("Tell me about the balance sheet", "DOCUMENT_QUERY"),
]

print("Testing Question Classification\n" + "="*60)

passed = 0
for question, expected in test_cases:
    result = qa_engine._classify_question(question)
    status = "✅" if result == expected else "❌"
    print(f"{status} Q: {question:40} Expected: {expected:15} Got: {result}")
    if result == expected:
        passed += 1

print(f"\n{passed}/{len(test_cases)} tests passed")
