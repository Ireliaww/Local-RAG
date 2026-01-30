"""
RAG问答引擎：Knowledge-Enhanced Generation with LangChain-style Memory

特性:
- 始终响应用户，无论检索结果如何
- 有相关文档时使用文档信息并引用
- 无相关文档时使用通用知识自然回答
- 对话式、自然的交互体验
- LangChain-style conversation memory for context-aware responses
"""
import os
import time
from typing import List, Dict, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# API 调用配置
MAX_RETRIES = 3
RETRY_DELAY = 1.0

# 相似度阈值 - 只有高于此值的文档才会作为上下文
SIMILARITY_THRESHOLD = 0.6

# 幻觉检测配置
ENABLE_HALLUCINATION_CHECK = False  # 是否启用幻觉检测（更保守但更慢）


class RAGQAEngine:
    """Knowledge-Enhanced QA Engine with Conversation Memory"""

    def __init__(
        self,
        vector_store,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.1,
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

    def _build_system_instruction(self, has_conversation_history: bool = False) -> str:
        """Constructs a structured system instruction for a professional Finance AI with memory."""
        memory_instruction = ""
        if has_conversation_history:
            memory_instruction = """
5. **Conversation Memory**:
   - You have access to the recent conversation history.
   - Use this context to understand follow-up questions and maintain coherence.
   - Reference previous exchanges naturally when relevant (e.g., "As I mentioned earlier...", "Building on our previous discussion...").
   - Handle pronouns and references that depend on conversation context (e.g., "it", "that company", "the same period").
"""

        return f"""ROLE:
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
{memory_instruction}
Remember: You are a unified AI assistant. The knowledge base is an extension of your memory, not a separate, external tool you are looking into."""

    def _classify_question(self, question: str) -> str:
        """
        Classify question type: CASUAL or DOCUMENT_QUERY
        
        Args:
            question: User's question
            
        Returns:
            "CASUAL" - for greetings, chitchat, questions about the AI itself
            "DOCUMENT_QUERY" - for questions requiring document lookup
        """
        # Simplified prompt to avoid triggering thinking mode
        classification_prompt = f"""Is this casual chat or a document query?

"{question}"

Reply ONE WORD: CASUAL or DOCUMENT_QUERY"""

        try:
            # Use gemini-2.5-flash for faster classification without thinking mode
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=classification_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic classification
                    max_output_tokens=100  # Generous limit to avoid truncation
                )
            )
            
            # Handle None or empty response
            if not response or not response.candidates or len(response.candidates) == 0:
                print(f"Empty classification response, defaulting to DOCUMENT_QUERY")
                return "DOCUMENT_QUERY"
            
            # Get text from first candidate
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts or len(candidate.content.parts) == 0:
                print(f"No content in classification response, defaulting to DOCUMENT_QUERY")
                return "DOCUMENT_QUERY"
            
            classification_text = candidate.content.parts[0].text
            if not classification_text:
                print(f"Empty text in classification response, defaulting to DOCUMENT_QUERY")
                return "DOCUMENT_QUERY"
                
            classification = classification_text.strip().upper()
            
            # Extract first word if multi-word response
            first_word = classification.split()[0] if classification.split() else classification
            
            # Validate response - handle partial matches
            if "CAS" in first_word:  # Matches "CAS", "CASUAL"
                return "CASUAL"
            elif "DOC" in first_word or "QUERY" in first_word:  # Matches "DOC", "DOCUMENT", "DOCUMENT_QUERY"
                return "DOCUMENT_QUERY"
            else:
                # Default to DOCUMENT_QUERY if unclear
                print(f"Unexpected classification response: '{classification}', defaulting to DOCUMENT_QUERY")
                return "DOCUMENT_QUERY"
                
        except Exception as e:
            print(f"Classification error: {e}, defaulting to DOCUMENT_QUERY")
            return "DOCUMENT_QUERY"


    def _check_answer_relevance(self, question: str, answer: str, chunks: List[Dict]) -> bool:
        """
        Check if the generated answer is based on the provided document chunks.
        
        Args:
            question: User's question
            answer: Generated answer
            chunks: Retrieved document chunks
            
        Returns:
            True if answer is relevant to chunks, False if hallucinated
        """
        if not chunks:
            return True  # No chunks to validate against
            
        # Build context from chunks
        context_parts = []
        for chunk in chunks:
            context_parts.append(chunk['text'])
        context = "\n\n".join(context_parts)
        
        validation_prompt = f"""You are a fact-checker. Determine if the ANSWER is based on information from the DOCUMENT CONTEXT.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Is the answer based on facts from the document context above? 
- Respond "YES" if the answer's main claims can be found in or reasonably inferred from the context.
- Respond "NO" if the answer contains significant information not present in the context.

Respond with ONLY one word: "YES" or "NO"."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=validation_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=10
                )
            )
            result = response.text.strip().upper()
            return result == "YES"
            
        except Exception as e:
            print(f"Relevance check error: {e}, assuming answer is relevant")
            return True
    def _build_prompt(
        self,
        question: str,
        relevant_chunks: List[Dict],
        conversation_history: Optional[str] = None
    ) -> str:
        """构建用户提示，支持对话历史"""
        prompt_parts = []

        # Add conversation history if available
        if conversation_history:
            prompt_parts.append(f"""CONVERSATION HISTORY:
{conversation_history}
--- END OF CONVERSATION HISTORY ---
""")

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

            prompt_parts.append(f"""RELEVANT DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTION: Synthesize the information from the documents above to answer the question. Cite sources naturally (e.g., "The quarterly report shows..." or "According to page 5..."). Consider the conversation history for context if available.""")
        else:
            # 无相关文档时，直接提问
            if conversation_history:
                prompt_parts.append(f"USER QUESTION: {question}\n\nINSTRUCTION: Answer the question considering the conversation history above for context.")
            else:
                prompt_parts.append(f"USER QUESTION: {question}")

        return "\n".join(prompt_parts)

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

    def answer_question(
        self,
        question: str,
        k: int = 5,
        conversation_history: Optional[str] = None
    ) -> Dict:
        """
        回答用户问题 - Knowledge-Enhanced模式 with Conversation Memory and Hallucination Control
        
        Args:
            question: 用户问题
            k: 检索的文档数量
            conversation_history: LangChain-style对话历史字符串
        
        根据问题类型智能响应：
        - CASUAL问题：直接生成答案
        - DOCUMENT_QUERY问题：需要相关文档支持，否则返回"未找到"
        """
        result = {
            'question': question,
            'answer': '',
            'retrieved_chunks': [],
            'relevant_chunks': [],
            'debug': {
                'question_type': '',
                'total_retrieved': 0,
                'relevant_count': 0,
                'avg_similarity': 0.0,
                'threshold': SIMILARITY_THRESHOLD,
                'has_memory': conversation_history is not None,
                'hallucination_check': False
            }
        }
        
        # Step 1: Classify question type
        question_type = self._classify_question(question)
        result['debug']['question_type'] = question_type
        print(f"Question classified as: {question_type}")
        
        # Step 2: Retrieve documents (always attempt retrieval)
        all_chunks = []
        try:
            if self.vector_store.get_collection_count() > 0:
                all_chunks = self.vector_store.similarity_search(
                    query=question,
                    k=k,
                    threshold=0.0  # Get all top-k for debug
                )
        except Exception as e:
            print(f"Retrieval warning: {e}")
        
        # Step 3: Filter relevant documents (above threshold)
        relevant_chunks = [
            chunk for chunk in all_chunks
            if chunk.get('similarity', 0) >= SIMILARITY_THRESHOLD
        ]
        
        # Record debug info
        result['retrieved_chunks'] = all_chunks
        result['relevant_chunks'] = relevant_chunks
        result['debug']['total_retrieved'] = len(all_chunks)
        result['debug']['relevant_count'] = len(relevant_chunks)
        if all_chunks:
            result['debug']['avg_similarity'] = sum(
                c['similarity'] for c in all_chunks
            ) / len(all_chunks)
        
        # Step 4: Generate response based on question type
        if question_type == "CASUAL":
            # For casual questions, always generate an answer
            has_history = conversation_history is not None and len(conversation_history.strip()) > 0
            system_instruction = self._build_system_instruction(has_conversation_history=has_history)
            prompt = self._build_prompt(question, [], conversation_history)  # No document context
            
            try:
                answer = self._generate_with_retry(system_instruction, prompt)
                result['answer'] = answer
            except Exception as e:
                result['answer'] = f"Sorry, I encountered an error: {str(e)}"
                result['error'] = str(e)
                
        elif question_type == "DOCUMENT_QUERY":
            # For document queries, require relevant documents
            if not relevant_chunks:
                # No relevant documents found - return "not found" message
                result['answer'] = (
                    "I apologize, but I couldn't find relevant information about this in the available documents. "
                    "Please try rephrasing your question or upload additional documents that may contain this information."
                )
                print(f"No relevant documents found for DOCUMENT_QUERY (max similarity: {all_chunks[0]['similarity'] if all_chunks else 0:.3f})")
            else:
                # Generate answer based on documents
                has_history = conversation_history is not None and len(conversation_history.strip()) > 0
                system_instruction = self._build_system_instruction(has_conversation_history=has_history)
                prompt = self._build_prompt(question, relevant_chunks, conversation_history)
                
                try:
                    answer = self._generate_with_retry(system_instruction, prompt)
                    
                    # Optional: Check for hallucinations
                    if ENABLE_HALLUCINATION_CHECK:
                        is_relevant = self._check_answer_relevance(question, answer, relevant_chunks)
                        result['debug']['hallucination_check'] = True
                        
                        if not is_relevant:
                            # Answer appears to be hallucinated
                            result['answer'] = (
                                "I apologize, but I couldn't find reliable information about this in the available documents. "
                                "Please try rephrasing your question or upload additional documents."
                            )
                            result['debug']['hallucination_detected'] = True
                            print("Hallucination detected - returning 'not found' message")
                        else:
                            result['answer'] = answer
                    else:
                        result['answer'] = answer
                        
                except Exception as e:
                    result['answer'] = f"Sorry, I encountered an error: {str(e)}"
                    result['error'] = str(e)
        
        return result

