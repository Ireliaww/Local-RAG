"""
Test script for hallucination detection and question classification
Tests with NVIDIA financial report related questions in English
"""
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.indexer import DocumentIndexer
from src.rag_qa import RAGQAEngine

load_dotenv()


def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_question(qa_engine, question, expected_type):
    """Test a single question and print results"""
    print(f"\n{'─' * 80}")
    print(f"Q: {question}")
    print(f"Expected Type: {expected_type}")
    print("─" * 80)
    
    result = qa_engine.answer_question(question, k=5)
    
    debug = result.get('debug', {})
    actual_type = debug.get('question_type', 'Unknown')
    
    # Print classification result
    type_match = "✅" if actual_type == expected_type else "❌"
    print(f"\n{type_match} Actual Type: {actual_type}")
    
    # Print debug info
    print(f"\nDebug Info:")
    print(f"  - Retrieved: {debug.get('total_retrieved', 0)} chunks")
    print(f"  - Relevant: {debug.get('relevant_count', 0)} chunks")
    print(f"  - Avg Similarity: {debug.get('avg_similarity', 0):.3f}")
    print(f"  - Threshold: {debug.get('threshold', 0.6)}")
    
    # Print answer (truncated if too long)
    answer = result.get('answer', '')
    print(f"\nAnswer:")
    if len(answer) > 300:
        print(f"  {answer[:300]}...")
    else:
        print(f"  {answer}")
    
    return actual_type == expected_type


def main():
    """Run test suite"""
    print_section("Hallucination Detection & Question Classification Test")
    print("Testing with NVIDIA Financial Reports")
    
    # Initialize system
    print("\nInitializing system...")
    indexer = DocumentIndexer()
    vector_store = indexer.vector_store
    qa_engine = RAGQAEngine(vector_store)
    
    # Check if documents are indexed
    doc_count = vector_store.get_collection_count()
    print(f"Documents in database: {doc_count}")
    
    if doc_count == 0:
        print("\n⚠️  WARNING: No documents indexed!")
        print("Please index NVIDIA financial reports first using the Gradio interface")
        print("or run: python main.py --index")
        return
    
    # Test cases
    test_cases = [
        # Casual conversations
        ("Hello", "CASUAL"),
        ("Hi, how are you?", "CASUAL"),
        ("Who are you?", "CASUAL"),
        ("What can you do?", "CASUAL"),
        ("Tell me a joke", "CASUAL"),
        
        # NVIDIA financial report queries (should find answers in docs)
        ("What was NVIDIA's revenue in Q3?", "DOCUMENT_QUERY"),
        ("What are the key highlights from NVIDIA's latest earnings report?", "DOCUMENT_QUERY"),
        ("How much did NVIDIA spend on research and development?", "DOCUMENT_QUERY"),
        ("What is NVIDIA's gross margin?", "DOCUMENT_QUERY"),
        ("Tell me about NVIDIA's data center revenue", "DOCUMENT_QUERY"),
        ("What is NVIDIA's operating income?", "DOCUMENT_QUERY"),
        
        # Document queries that might not have answers (depends on what's in the PDFs)
        ("What is NVIDIA's employee turnover rate?", "DOCUMENT_QUERY"),
        ("Who is NVIDIA's biggest competitor?", "DOCUMENT_QUERY"),
        ("What is NVIDIA's marketing budget?", "DOCUMENT_QUERY"),
    ]
    
    # Run tests
    print_section("Running Test Cases")
    
    results = {
        'total': len(test_cases),
        'passed': 0,
        'failed': 0
    }
    
    for question, expected_type in test_cases:
        passed = test_question(qa_engine, question, expected_type)
        if passed:
            results['passed'] += 1
        else:
            results['failed'] += 1
    
    # Print summary
    print_section("Test Summary")
    print(f"\nTotal Tests: {results['total']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"Success Rate: {results['passed'] / results['total'] * 100:.1f}%")
    
    # Additional check: Test "not found" responses
    print_section("Testing 'Not Found' Responses")
    print("\nTesting questions that should return 'not found' message...")
    
    # Questions unlikely to be in financial reports
    unlikely_questions = [
        "What is NVIDIA CEO's favorite color?",
        "How many cafeterias does NVIDIA have?",
        "What is the office layout of NVIDIA headquarters?",
    ]
    
    for question in unlikely_questions:
        print(f"\n{'─' * 80}")
        print(f"Q: {question}")
        result = qa_engine.answer_question(question, k=5)
        answer = result.get('answer', '')
        
        # Check if answer contains "not found" or similar language
        not_found_keywords = ["couldn't find", "not found", "unable to find", "no information"]
        is_not_found = any(keyword in answer.lower() for keyword in not_found_keywords)
        
        debug = result.get('debug', {})
        relevant_count = debug.get('relevant_count', 0)
        
        status = "✅" if is_not_found or relevant_count == 0 else "⚠️"
        print(f"{status} Relevant chunks: {relevant_count}")
        print(f"Answer: {answer[:200]}...")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
