"""
向量存储模块：使用ChromaDB存储文本向量，配置为Cosine Similarity

特性:
- 使用 cosine 相似度距离
- 支持批量添加文档 (自动分批处理大量文档)
- 使用唯一ID避免冲突 (基于内容hash)
- 查询时使用 RETRIEVAL_QUERY 任务类型
"""
import os
import hashlib
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from dotenv import load_dotenv
from .gemini_embedding import GeminiEmbeddingFunction

load_dotenv()

# 批量添加文档的大小
ADD_BATCH_SIZE = 50


class ChromaVectorStore:
    """ChromaDB向量存储，使用Google Gemini embedding和Cosine相似度"""

    def __init__(self, collection_name: str = "financial_reports", persist_directory: str = "chroma_db"):
        """
        初始化ChromaDB向量存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 获取Google API Key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("请设置GOOGLE_API_KEY环境变量")

        self._api_key = api_key

        # 初始化ChromaDB客户端
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 使用Gemini embedding function (默认为 RETRIEVAL_DOCUMENT)
        self.embedding_function = GeminiEmbeddingFunction(
            api_key=api_key,
            model_name="models/text-embedding-004",
            task_type="RETRIEVAL_DOCUMENT"
        )

        # 获取或创建集合（使用cosine距离）
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # 使用cosine相似度
        )

        print(f"ChromaDB向量存储已初始化: {collection_name}")

    @staticmethod
    def _generate_chunk_id(text: str, page_number: int, index: int, source_file: str = "") -> str:
        """
        生成唯一的chunk ID

        Args:
            text: 文本内容
            page_number: 页码
            index: 索引序号
            source_file: 源文件名

        Returns:
            唯一ID (基于内容hash)
        """
        content = f"{source_file}:{page_number}:{index}:{text[:200]}"
        return f"chunk_{hashlib.md5(content.encode()).hexdigest()[:16]}"
    
    def add_chunks(self, chunks: List[Dict], source_file: str = ""):
        """
        添加文本块到向量数据库

        Args:
            chunks: 文本块列表，每个元素包含text, page_number, n_tokens
            source_file: 源文件名（用于多文档追踪）

        Note:
            - 使用唯一ID避免重复添加时的冲突
            - 大量文档会自动分批处理
        """
        if not chunks:
            print("没有文本块需要添加")
            return

        # 准备数据
        texts = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            texts.append(chunk['text'])
            metadatas.append({
                'page_number': chunk['page_number'],
                'n_tokens': chunk['n_tokens'],
                'source_file': source_file or chunk.get('source_file', '')
            })
            # 使用唯一ID
            ids.append(self._generate_chunk_id(
                chunk['text'],
                chunk['page_number'],
                i,
                source_file
            ))

        # 分批添加 (避免大量文档导致的内存问题)
        total_added = 0
        for i in range(0, len(chunks), ADD_BATCH_SIZE):
            batch_texts = texts[i:i + ADD_BATCH_SIZE]
            batch_metadatas = metadatas[i:i + ADD_BATCH_SIZE]
            batch_ids = ids[i:i + ADD_BATCH_SIZE]

            try:
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                total_added += len(batch_texts)
                print(f"已添加 {total_added}/{len(chunks)} 个文本块...")
            except Exception as e:
                print(f"添加文本块时出错 (batch {i // ADD_BATCH_SIZE + 1}): {e}")
                raise

        print(f"成功添加 {total_added} 个文本块到向量数据库")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.6
    ) -> List[Dict]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回的top-k结果数量
            threshold: 相似度阈值，低于此值的将被过滤

        Returns:
            检索结果列表，每个元素包含text, page_number, n_tokens, distance, similarity

        Note:
            使用 RETRIEVAL_QUERY 任务类型生成查询 embedding，
            这与文档存储时使用的 RETRIEVAL_DOCUMENT 配对使用效果更好
        """
        # 使用 RETRIEVAL_QUERY 任务类型生成查询 embedding
        query_embedding = self.embedding_function.embed_query(query)

        # 使用 embedding 直接查询 (而不是 query_texts)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        # 处理结果
        retrieved_chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # 计算相似度分数 (cosine距离: 相似度 = 1 - distance)
                similarity = 1.0 - distance

                # 过滤低于阈值的结果
                if similarity >= threshold:
                    retrieved_chunks.append({
                        'text': doc,
                        'page_number': metadata.get('page_number', 0),
                        'n_tokens': metadata.get('n_tokens', 0),
                        'source_file': metadata.get('source_file', ''),
                        'distance': distance,
                        'similarity': similarity
                    })

        print(f"检索到 {len(retrieved_chunks)} 个相关文本块（阈值: {threshold}）")
        return retrieved_chunks
    
    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        return self.collection.count()
    
    def clear_collection(self):
        """清空集合"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        print("集合已清空")
