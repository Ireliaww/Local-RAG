"""
评估脚本：预设10个问题，循环运行并打印结果以评估RAG质量

特性:
- 10个预设财务分析问题
- 详细的检索和生成结果输出
- 时间性能指标统计
- 相似度分数分析
"""
import time
from typing import List, Dict
from src.rag_qa import RAGQAEngine


# 预设的10个评估问题（针对财务报表）
EVALUATION_QUESTIONS = [
    "公司的总营收是多少？",
    "净利润是多少？",
    "公司的主要风险因素有哪些？",
    "公司的毛利率是多少？",
    "公司的主要业务板块是什么？",
    "公司的资产负债率是多少？",
    "公司的现金流情况如何？",
    "公司的主要竞争对手有哪些？",
    "公司的研发投入是多少？",
    "公司的未来发展战略是什么？"
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
    print("开始RAG系统评估")
    print(f"共 {len(questions)} 个问题")
    print("=" * 80)

    results = []
    timing_data = []
    total_start_time = time.time()

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"问题 {i}/{len(questions)}: {question}")
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
        print(f"\n答案:")
        print(result['answer'])
        print(f"\n检索信息:")
        print(f"  - 使用的文档片段数: {result['num_chunks']}")
        print(f"  - 响应时间: {elapsed_time:.2f} 秒")

        if result['retrieved_chunks']:
            print(f"  - 相关文档片段:")
            for j, chunk in enumerate(result['retrieved_chunks'], 1):
                print(f"    [{j}] 第{chunk['page_number']}页 | "
                      f"相似度: {chunk['similarity']:.3f} | "
                      f"Token数: {chunk['n_tokens']}")
                # 显示片段预览
                preview = chunk['text'][:100].replace('\n', ' ')
                print(f"       预览: {preview}...")

        results.append(result)

        print("\n" + "-" * 80)

    total_elapsed_time = time.time() - total_start_time

    # 打印评估摘要
    print("\n" + "=" * 80)
    print("评估摘要")
    print("=" * 80)

    total_chunks = sum(r['num_chunks'] for r in results)
    avg_chunks = total_chunks / len(results) if results else 0

    print(f"\n📊 基础统计:")
    print(f"  总问题数: {len(questions)}")
    print(f"  平均使用的文档片段数: {avg_chunks:.2f}")
    print(f"  成功检索到文档片段的问题数: {sum(1 for r in results if r['num_chunks'] > 0)}")
    print(f"  未检索到文档片段的问题数: {sum(1 for r in results if r['num_chunks'] == 0)}")

    # 计算平均相似度
    all_similarities = []
    for result in results:
        for chunk in result['retrieved_chunks']:
            all_similarities.append(chunk['similarity'])

    if all_similarities:
        avg_similarity = sum(all_similarities) / len(all_similarities)
        print(f"\n📈 相似度统计:")
        print(f"  平均相似度: {avg_similarity:.3f}")
        print(f"  最高相似度: {max(all_similarities):.3f}")
        print(f"  最低相似度: {min(all_similarities):.3f}")

    # 时间统计
    print(f"\n⏱️  时间性能:")
    print(f"  总耗时: {total_elapsed_time:.2f} 秒")
    print(f"  平均响应时间: {sum(timing_data) / len(timing_data):.2f} 秒/问题")
    print(f"  最快响应: {min(timing_data):.2f} 秒")
    print(f"  最慢响应: {max(timing_data):.2f} 秒")

    # 检查错误
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"\n⚠️  错误统计:")
        print(f"  出错问题数: {len(errors)}")
        for r in errors:
            print(f"    - {r['question'][:30]}... : {r['error']}")

    print("\n" + "=" * 80)
    print("评估完成！")
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
