"""
RAG问答系统主程序入口
"""
import os
import argparse
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessor, TextChunker
from src.vector_store import ChromaVectorStore
from src.rag_qa import RAGQAEngine

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="RAG问答系统 - 财务报表分析")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="PDF文件路径或URL"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="重新索引PDF（清空现有向量数据库）"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="单个问题（交互模式）"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="运行评估脚本（10个预设问题）"
    )
    
    args = parser.parse_args()
    
    # 初始化组件
    print("=" * 60)
    print("初始化RAG问答系统...")
    print("=" * 60)
    
    pdf_processor = PDFProcessor()
    text_chunker = TextChunker(min_tokens=300, max_tokens=500)
    vector_store = ChromaVectorStore()
    qa_engine = RAGQAEngine(vector_store)
    
    # 检查是否需要重新索引
    if args.reindex or vector_store.get_collection_count() == 0:
        print("\n开始处理PDF文档...")
        # 处理PDF
        pages = pdf_processor.process_pdf(args.pdf)
        
        # 文本分块
        print("\n开始文本分块...")
        chunks = text_chunker.chunk_text(pages)
        
        # 清空现有数据（如果需要）
        if args.reindex:
            vector_store.clear_collection()
        
        # 添加到向量数据库
        print("\n开始向量化存储...")
        vector_store.add_chunks(chunks)
        print(f"\n索引完成！共存储 {vector_store.get_collection_count()} 个文本块")
    else:
        print(f"\n使用现有索引（{vector_store.get_collection_count()} 个文本块）")
    
    # 问答模式
    if args.evaluate:
        # 运行评估
        from evaluation import run_evaluation
        run_evaluation(qa_engine)
    elif args.question:
        # 单个问题
        print(f"\n问题: {args.question}")
        print("-" * 60)
        result = qa_engine.answer_question(args.question)
        print(f"\n答案:\n{result['answer']}")
        print(f"\n使用了 {result['num_chunks']} 个相关文档片段")
    else:
        # 交互模式
        print("\n进入交互模式（输入 'quit' 退出）")
        print("-" * 60)
        while True:
            question = input("\n请输入您的问题: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue
            
            result = qa_engine.answer_question(question)
            print(f"\n答案:\n{result['answer']}")
            print(f"\n使用了 {result['num_chunks']} 个相关文档片段")
            if result['retrieved_chunks']:
                print("\n相关文档片段:")
                for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
                    print(f"  [{i}] 第{chunk['page_number']}页 (相似度: {chunk['similarity']:.3f})")


if __name__ == "__main__":
    main()
