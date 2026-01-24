"""
RAG问答引擎：Knowledge-Enhanced Generation

特性:
- 始终响应用户，无论检索结果如何
- 有相关文档时使用文档信息并引用
- 无相关文档时使用通用知识自然回答
- 对话式、自然的交互体验
"""
import os
import time
from typing import List, Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# API 调用配置
MAX_RETRIES = 3
RETRY_DELAY = 1.0

# 相似度阈值 - 只有高于此值的文档才会作为上下文
SIMILARITY_THRESHOLD = 0.6


class RAGQAEngine:
    """Knowledge-Enhanced QA Engine - 智能助手模式"""

    def __init__(
        self,
        vector_store,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        max_output_tokens: int = 2000
    ):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("请设置GOOGLE_API_KEY环境变量")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.vector_store = vector_store
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _build_system_instruction(self) -> str:
        """Constructs a structured system instruction for a professional Finance AI."""
        return """ROLE:
You are a highly skilled Financial Analyst AI. Your goal is to assist users by providing accurate information from the provided financial documents or your general knowledge.

CORE BEHAVIORS:

1. **Context-Aware Response (Priority)**:
   - When "RELEVANT DOCUMENT CONTEXT" is provided, prioritize this information.
   - Integrate the facts seamlessly into your response.
   - **Citation Style**: Use natural, professional citations (e.g., "The Q3 report indicates...", "Based on the balance sheet...", or "According to the filing..."). Do not use bracketed indices like [1] unless specifically asked.

2. **Graceful Fallback**:
   - If the context is missing, insufficient, or irrelevant to the user's specific query, leverage your internal expertise to provide a helpful answer.
   - **CRITICAL**: Never mention phrases like "The provided documents do not contain..." or "I don't have access to that information in the context." Simply answer the question as a knowledgeable assistant would.

3. **Tone and Style**:
   - Maintain a professional, objective, and conversational tone.
   - Avoid "AI-talk" (e.g., "I have analyzed the retrieved chunks for you...").
   - For greetings or casual queries, be warm and concise.

4. **Safety & Accuracy**:
   - Do not speculate on specific stock prices or provide personalized financial advice.
   - If a conflict exists between your general knowledge and the provided context, prioritize the provided context as the most recent/specific data.

Remember: You are a unified AI assistant. The knowledge base is an extension of your memory, not a separate, external tool you are looking into."""
    def _build_prompt(self, question: str, relevant_chunks: List[Dict]) -> str:
        """构建用户提示"""
        if relevant_chunks:
            # 有相关文档时，提供上下文 (使用清晰的分隔符)
            context_parts = []
            for chunk in relevant_chunks:
                source = chunk.get('source_file', 'Document')
                page = chunk.get('page_number', '?')
                text = chunk['text']
                context_parts.append(
                    f"--- START OF DOCUMENT: {source} (Page {page}) ---\n"
                    f"{text}\n"
                    f"--- END OF DOCUMENT ---"
                )
            context = "\n\n".join(context_parts)

            return f"""RELEVANT DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTION: Synthesize the information from the documents above to answer the question. Cite sources naturally (e.g., "The quarterly report shows..." or "According to page 5...")."""
        else:
            # 无相关文档时，直接提问
            return f"USER QUESTION: {question}"

    def _generate_with_retry(self, system_instruction: str, prompt: str) -> str:
        """带重试的内容生成"""
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens
                    )
                )
                return response.text

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                retryable = any(keyword in error_msg for keyword in [
                    'timeout', 'rate limit', '429', '503', '500',
                    'overloaded', 'temporarily', 'connection'
                ])

                if retryable and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    print(f"API调用失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                    time.sleep(delay)
                elif not retryable:
                    raise

        raise RuntimeError(f"API调用失败，已重试 {MAX_RETRIES} 次: {last_exception}")

    def answer_question(self, question: str, k: int = 5) -> Dict:
        """
        回答用户问题 - Knowledge-Enhanced模式

        始终调用LLM生成回答，根据检索结果决定是否提供文档上下文。
        """
        result = {
            'question': question,
            'answer': '',
            'retrieved_chunks': [],
            'relevant_chunks': [],
            'debug': {
                'total_retrieved': 0,
                'relevant_count': 0,
                'avg_similarity': 0.0,
                'threshold': SIMILARITY_THRESHOLD
            }
        }

        # 1. 尝试检索相关文档（如果向量库有数据）
        all_chunks = []
        try:
            if self.vector_store.get_collection_count() > 0:
                all_chunks = self.vector_store.similarity_search(
                    query=question,
                    k=k,
                    threshold=0.0  # 获取所有top-k用于debug
                )
        except Exception as e:
            print(f"检索警告: {e}")

        # 2. 过滤出相关文档（高于阈值的）
        relevant_chunks = [
            chunk for chunk in all_chunks
            if chunk.get('similarity', 0) >= SIMILARITY_THRESHOLD
        ]

        # 记录debug信息
        result['retrieved_chunks'] = all_chunks
        result['relevant_chunks'] = relevant_chunks
        result['debug']['total_retrieved'] = len(all_chunks)
        result['debug']['relevant_count'] = len(relevant_chunks)
        if all_chunks:
            result['debug']['avg_similarity'] = sum(
                c['similarity'] for c in all_chunks
            ) / len(all_chunks)

        # 3. 构建提示并调用LLM（始终调用！）
        system_instruction = self._build_system_instruction()
        prompt = self._build_prompt(question, relevant_chunks)

        try:
            answer = self._generate_with_retry(system_instruction, prompt)
            result['answer'] = answer
        except Exception as e:
            result['answer'] = f"Sorry, I encountered an error: {str(e)}"
            result['error'] = str(e)

        return result
