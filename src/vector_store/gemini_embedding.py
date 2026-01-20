"""
Google Gemini Embedding Function for ChromaDB

使用最新的 google-genai SDK (from google import genai)
支持批量 embedding 和不同的任务类型 (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)
"""
import os
import time
from google import genai
from google.genai import types
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Embedding 维度 (text-embedding-004)
EMBEDDING_DIMENSION = 768

# API 调用配置
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # 初始重试延迟(秒)
BATCH_SIZE = 100  # Gemini API 批量处理大小


class GeminiEmbeddingFunction:
    """ChromaDB兼容的Gemini Embedding函数，支持批量处理和不同任务类型"""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "models/text-embedding-004",
        task_type: str = "RETRIEVAL_DOCUMENT"
    ):
        """
        初始化Gemini Embedding函数

        Args:
            api_key: Google API Key，如果为None则从环境变量读取
            model_name: Gemini embedding模型名称，默认为text-embedding-004
            task_type: 任务类型
                - RETRIEVAL_DOCUMENT: 用于存储文档 (默认)
                - RETRIEVAL_QUERY: 用于查询检索
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("请设置GOOGLE_API_KEY环境变量")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.api_key = api_key
        self.task_type = task_type

    def name(self) -> str:
        """返回embedding function的名称，ChromaDB需要此方法"""
        return "gemini-embedding-function"

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        带重试的批量 embedding 生成

        Args:
            texts: 文本列表 (应该 <= BATCH_SIZE)

        Returns:
            embedding向量列表
        """
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=self.task_type
                    )
                )

                # 提取 embedding 向量
                if hasattr(result, 'embeddings') and result.embeddings:
                    return [emb.values for emb in result.embeddings]
                else:
                    raise ValueError(f"API返回格式异常: {result}")

            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # 指数退避
                    print(f"Embedding API调用失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                    print(f"等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)

        # 所有重试都失败
        raise RuntimeError(f"Embedding API调用失败，已重试 {MAX_RETRIES} 次: {last_exception}")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        对输入文本列表生成embedding (ChromaDB 调用接口)

        Args:
            input: 文本列表

        Returns:
            embedding向量列表
        """
        if not input:
            return []

        all_embeddings = []

        # 批量处理
        for i in range(0, len(input), BATCH_SIZE):
            batch = input[i:i + BATCH_SIZE]

            try:
                batch_embeddings = self._embed_with_retry(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # 批量处理失败，降级为逐个处理
                print(f"批量embedding失败，降级为逐个处理: {e}")
                for text in batch:
                    try:
                        single_result = self._embed_with_retry([text])
                        all_embeddings.extend(single_result)
                    except Exception as single_e:
                        print(f"生成embedding失败: {single_e}, 文本长度: {len(text)}")
                        # 返回零向量作为fallback
                        all_embeddings.append([0.0] * EMBEDDING_DIMENSION)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档生成 embedding (使用 RETRIEVAL_DOCUMENT 任务类型)

        Args:
            texts: 文档文本列表

        Returns:
            embedding向量列表
        """
        original_task_type = self.task_type
        self.task_type = "RETRIEVAL_DOCUMENT"
        try:
            return self.__call__(texts)
        finally:
            self.task_type = original_task_type

    def embed_query(self, query: str) -> List[float]:
        """
        为查询生成 embedding (使用 RETRIEVAL_QUERY 任务类型)

        Args:
            query: 查询文本

        Returns:
            embedding向量
        """
        original_task_type = self.task_type
        self.task_type = "RETRIEVAL_QUERY"
        try:
            result = self.__call__([query])
            return result[0] if result else [0.0] * EMBEDDING_DIMENSION
        finally:
            self.task_type = original_task_type
