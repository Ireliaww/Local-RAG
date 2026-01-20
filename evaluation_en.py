"""
评估脚本：预设10个英文问题，循环运行并打印结果以评估RAG质量

特性:
- 10个预设英文财务分析问题（针对NVIDIA财报）
- 详细的检索和生成结果输出
- 时间性能指标统计
- 相似度分数分析
"""
import time
from typing import List, Dict
from src.rag_qa import RAGQAEngine


# 预设的10个评估问题（英文，针对NVIDIA财务报表）
EVALUATION_QUESTIONS = [
    "What is the company's total revenue?",
    "What is the net income?",
    "What are the main risk factors for the company?",
    "What is the gross profit margin?",
    "What are the main business segments?",
    "What is the debt-to-asset ratio?",
    "How is the company's cash flow situation?",
    "Who are the main competitors?",
    "What is the R&D spending?",
    "What is the company's future growth strategy?"
]


def run_evaluation(qa_engine: RAGQAEngine, questions: List[str] = None) -> Dict:
    """
    运行评估脚本

    Args:
        qa_engine: RAGQAEngine实例
        questions: 问题列表，如果为None则使用默认问题

    Returns:
        评估结果字典，包含详细结果和统计信息
    """
    if questions is None:
        questions = EVALUATION_QUESTIONS

    print("=" * 80)
    print("Starting RAG System Evaluation")
    print(f"Total questions: {len(questions)}")
    print("=" * 80)

    results = []
    timing_data = []
    total_start_time = time.time()

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}/{len(questions)}: {question}")
        print("-" * 80)

        # 计时开始
        start_time = time.time()

        # 获取答案
        result = qa_engine.answer_question(
            question=question,
            k=5,
            threshold=0.6
        )

        # 计时结束
        elapsed_time = time.time() - start_time
        result['elapsed_time'] = elapsed_time
        timing_data.append(elapsed_time)

        # 打印结果
        print(f"\nAnswer:")
        print(result['answer'])
        print(f"\nRetrieval Info:")
        print(f"  - Document chunks used: {result['num_chunks']}")
        print(f"  - Response time: {elapsed_time:.2f} sec")

        if result['retrieved_chunks']:
            print(f"  - Related document chunks:")
            for j, chunk in enumerate(result['retrieved_chunks'], 1):
                print(f"    [{j}] Page {chunk['page_number']} | "
                      f"Similarity: {chunk['similarity']:.3f} | "
                      f"Tokens: {chunk['n_tokens']}")
                # 显示片段预览
                preview = chunk['text'][:100].replace('\n', ' ')
                print(f"       Preview: {preview}...")

        results.append(result)

        print("\n" + "-" * 80)

    total_elapsed_time = time.time() - total_start_time

    # 打印评估摘要
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    total_chunks = sum(r['num_chunks'] for r in results)
    avg_chunks = total_chunks / len(results) if results else 0

    print(f"\n📊 Basic Statistics:")
    print(f"  Total questions: {len(questions)}")
    print(f"  Avg document chunks used: {avg_chunks:.2f}")
    print(f"  Questions with chunks retrieved: {sum(1 for r in results if r['num_chunks'] > 0)}")
    print(f"  Questions without chunks: {sum(1 for r in results if r['num_chunks'] == 0)}")

    # 计算平均相似度
    all_similarities = []
    for result in results:
        for chunk in result['retrieved_chunks']:
            all_similarities.append(chunk['similarity'])

    if all_similarities:
        avg_similarity = sum(all_similarities) / len(all_similarities)
        print(f"\n📈 Similarity Statistics:")
        print(f"  Average similarity: {avg_similarity:.3f}")
        print(f"  Max similarity: {max(all_similarities):.3f}")
        print(f"  Min similarity: {min(all_similarities):.3f}")

    # 时间统计
    print(f"\n⏱️  Performance:")
    print(f"  Total time: {total_elapsed_time:.2f} sec")
    print(f"  Avg response time: {sum(timing_data) / len(timing_data):.2f} sec/question")
    print(f"  Fastest response: {min(timing_data):.2f} sec")
    print(f"  Slowest response: {max(timing_data):.2f} sec")

    # 检查错误
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"\n⚠️  Errors:")
        print(f"  Questions with errors: {len(errors)}")
        for r in errors:
            print(f"    - {r['question'][:30]}... : {r['error']}")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)

    # 返回完整的评估结果
    return {
        'results': results,
        'summary': {
            'total_questions': len(questions),
            'avg_chunks': avg_chunks,
            'successful_retrievals': sum(1 for r in results if r['num_chunks'] > 0),
            'failed_retrievals': sum(1 for r in results if r['num_chunks'] == 0),
            'avg_similarity': sum(all_similarities) / len(all_similarities) if all_similarities else 0,
            'total_time': total_elapsed_time,
            'avg_response_time': sum(timing_data) / len(timing_data),
            'min_response_time': min(timing_data),
            'max_response_time': max(timing_data),
            'errors': len(errors)
        }
    }


if __name__ == "__main__":
    # 如果直接运行，需要先初始化系统
    import os
    from dotenv import load_dotenv
    from src.vector_store import ChromaVectorStore
    from src.rag_qa import RAGQAEngine

    load_dotenv()

    vector_store = ChromaVectorStore()
    qa_engine = RAGQAEngine(vector_store)

    run_evaluation(qa_engine)
