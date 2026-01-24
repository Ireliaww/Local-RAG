#!/usr/bin/env python3
"""
RAG系统评估脚本

使用方法:
    python run_evaluation.py                    # 使用默认golden dataset
    python run_evaluation.py --dataset path.json  # 使用自定义dataset
    python run_evaluation.py --quick            # 快速模式 (前5个问题)
"""
import argparse
import json
from pathlib import Path

from src.indexer import DocumentIndexer
from src.rag_qa import RAGQAEngine
from src.evaluation import RAGEvaluator


def load_test_cases(dataset_path: str) -> list:
    """加载测试用例"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/golden_dataset.json",
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only evaluate first 5 cases"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG System Evaluation")
    print("=" * 60)

    # 加载测试用例
    print(f"\nLoading test cases from: {args.dataset}")
    test_cases = load_test_cases(args.dataset)

    if args.quick:
        test_cases = test_cases[:5]
        print(f"Quick mode: using {len(test_cases)} test cases")
    else:
        print(f"Loaded {len(test_cases)} test cases")

    # 初始化系统
    print("\nInitializing RAG system...")
    indexer = DocumentIndexer()
    vector_store = indexer.vector_store
    qa_engine = RAGQAEngine(vector_store)

    # 显示索引状态
    status = indexer.get_status()
    print(f"- Indexed documents: {status['indexed_pdfs']}")
    print(f"- Total chunks: {status['total_chunks']}")

    if status['total_chunks'] == 0:
        print("\nWARNING: No documents indexed! Add PDFs to data/pdfs/ and index them first.")
        print("Context-based evaluation will not be meaningful without documents.")

    # 初始化评估器
    print("\nInitializing evaluator...")
    evaluator = RAGEvaluator()

    # 运行评估
    print("\nRunning evaluation...")
    print("-" * 60)

    results, report = evaluator.evaluate_rag_system(
        qa_engine=qa_engine,
        test_cases=test_cases,
        delay_between_calls=0.5
    )

    # 打印报告
    print("\n" + "=" * 60)
    print(report.to_markdown())

    # 保存结果
    print("\n" + "-" * 60)
    evaluator.save_results(results, report, output_dir=args.output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
