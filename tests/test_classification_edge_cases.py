"""
Test improved classification with edge cases that might be misclassified
Now: General knowledge = CASUAL, Document-specific = DOCUMENT_QUERY
"""
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.indexer import DocumentIndexer
from src.rag_qa import RAGQAEngine

load_dotenv()

# Initialize
indexer = DocumentIndexer()
qa_engine = RAGQAEngine(indexer.vector_store)

# Test cases with new classification logic
test_cases = [
    # General knowledge - should be CASUAL (AI can answer from world knowledge)
    ("What is quantum computing?", "CASUAL"),
    ("How tall is the Eiffel Tower?", "CASUAL"),
    ("When was Apple founded?", "CASUAL"),
    ("What are some best practices for hiring?", "CASUAL"),
    ("Who invented the telephone?", "CASUAL"),
    
    # Document-specific questions - should be DOCUMENT_QUERY
    ("What was NVIDIA's Q3 revenue?", "DOCUMENT_QUERY"),
    ("What is the company's gross margin?", "DOCUMENT_QUERY"),
    ("What does the balance sheet show?", "DOCUMENT_QUERY"),
    ("What are the key highlights from the earnings report?", "DOCUMENT_QUERY"),
    ("How much did the company spend on R&D?", "DOCUMENT_QUERY"),
    
    # Greetings and AI questions - should be CASUAL
    ("Hello", "CASUAL"),
    ("Hi there", "CASUAL"),
    ("Who are you?", "CASUAL"),
    ("What can you do?", "CASUAL"),
    ("Tell me a joke", "CASUAL"),
]

print("Testing Refined Classification")
print("General Knowledge → CASUAL | Document-Specific → DOCUMENT_QUERY")
print("=" * 80)

passed = 0
failed_cases = []

for question, expected in test_cases:
    result = qa_engine._classify_question(question)
    status = "✅" if result == expected else "❌"
    
    print(f"{status} Q: {question:50} Expected: {expected:15} Got: {result}")
    
    if result == expected:
        passed += 1
    else:
        failed_cases.append((question, expected, result))

print("\n" + "=" * 80)
print(f"Results: {passed}/{len(test_cases)} tests passed ({passed/len(test_cases)*100:.1f}%)")

if failed_cases:
    print("\n❌ Failed Cases:")
    for q, exp, got in failed_cases:
        print(f"   Q: {q}")
        print(f"   Expected: {exp}, Got: {got}\n")
else:
    print("\n🎉 All tests passed!")
