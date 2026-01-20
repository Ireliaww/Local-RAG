"""
文档索引器：批量处理PDF文件并建立向量索引

特性:
- 扫描文件夹中的所有PDF文件
- 跳过已索引的文件（增量索引）
- 支持强制重新索引
"""
import os
import json
from typing import List, Optional
from pathlib import Path

from src.pdf_processor import PDFProcessor, TextChunker
from src.vector_store import ChromaVectorStore


class DocumentIndexer:
    """文档索引器，支持批量PDF处理和增量索引"""

    def __init__(
        self,
        pdf_dir: str = "data/pdfs",
        index_record_path: str = "data/indexed_files.json"
    ):
        """
        初始化文档索引器

        Args:
            pdf_dir: PDF文件目录
            index_record_path: 索引记录文件路径
        """
        self.pdf_dir = pdf_dir
        self.index_record_path = index_record_path

        self.pdf_processor = PDFProcessor(data_dir=pdf_dir)
        self.text_chunker = TextChunker(min_tokens=300, max_tokens=500)
        self.vector_store = ChromaVectorStore()

        # 加载已索引文件记录
        self._indexed_files = self._load_index_record()

    def _load_index_record(self) -> dict:
        """加载已索引文件记录"""
        if os.path.exists(self.index_record_path):
            try:
                with open(self.index_record_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_index_record(self):
        """保存索引记录"""
        os.makedirs(os.path.dirname(self.index_record_path), exist_ok=True)
        with open(self.index_record_path, 'w', encoding='utf-8') as f:
            json.dump(self._indexed_files, f, ensure_ascii=False, indent=2)

    def get_pdf_files(self) -> List[str]:
        """获取目录中的所有PDF文件"""
        pdf_dir = Path(self.pdf_dir)
        if not pdf_dir.exists():
            return []

        pdf_files = list(pdf_dir.glob("*.pdf"))
        return [str(f) for f in pdf_files]

    def get_unindexed_files(self) -> List[str]:
        """获取尚未索引的PDF文件"""
        all_files = self.get_pdf_files()
        return [f for f in all_files if os.path.basename(f) not in self._indexed_files]

    def get_indexed_files(self) -> List[str]:
        """获取已索引的文件列表"""
        return list(self._indexed_files.keys())

    def index_single_file(self, pdf_path: str, force: bool = False) -> dict:
        """
        索引单个PDF文件

        Args:
            pdf_path: PDF文件路径
            force: 是否强制重新索引

        Returns:
            索引结果信息
        """
        filename = os.path.basename(pdf_path)

        # 检查是否已索引
        if not force and filename in self._indexed_files:
            return {
                'file': filename,
                'status': 'skipped',
                'message': '文件已索引'
            }

        try:
            print(f"\n处理文件: {filename}")

            # 提取文本
            pages = self.pdf_processor.extract_text_from_pdf(pdf_path)

            # 文本分块
            chunks = self.text_chunker.chunk_text(pages)

            # 添加到向量数据库
            self.vector_store.add_chunks(chunks, source_file=filename)

            # 记录已索引
            self._indexed_files[filename] = {
                'path': pdf_path,
                'pages': len(pages),
                'chunks': len(chunks)
            }
            self._save_index_record()

            return {
                'file': filename,
                'status': 'success',
                'pages': len(pages),
                'chunks': len(chunks)
            }

        except Exception as e:
            return {
                'file': filename,
                'status': 'error',
                'message': str(e)
            }

    def index_all(self, force: bool = False) -> List[dict]:
        """
        索引目录中的所有PDF文件

        Args:
            force: 是否强制重新索引所有文件

        Returns:
            每个文件的索引结果列表
        """
        if force:
            # 清空向量数据库和索引记录
            self.vector_store.clear_collection()
            self._indexed_files = {}
            self._save_index_record()
            files_to_index = self.get_pdf_files()
        else:
            files_to_index = self.get_unindexed_files()

        if not files_to_index:
            print("没有需要索引的文件")
            return []

        print(f"发现 {len(files_to_index)} 个需要索引的PDF文件")

        results = []
        for pdf_path in files_to_index:
            result = self.index_single_file(pdf_path, force=force)
            results.append(result)

        # 打印摘要
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')

        print(f"\n索引完成: 成功 {success_count}, 失败 {error_count}, 跳过 {skipped_count}")
        print(f"向量数据库共有 {self.vector_store.get_collection_count()} 个文本块")

        return results

    def get_status(self) -> dict:
        """获取索引状态"""
        all_files = self.get_pdf_files()
        indexed_files = self.get_indexed_files()

        return {
            'total_pdfs': len(all_files),
            'indexed_pdfs': len(indexed_files),
            'pending_pdfs': len(all_files) - len(indexed_files),
            'total_chunks': self.vector_store.get_collection_count(),
            'indexed_files': indexed_files
        }
