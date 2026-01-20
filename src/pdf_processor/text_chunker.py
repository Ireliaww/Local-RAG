"""
文本分块模块：按照段落切分文本，使用tiktoken确保每个块在300-500 tokens之间

特性:
- 使用 tiktoken (cl100k_base) 进行 token 计数
- 支持中英文标点符号
- 智能段落和句子切分
"""
import tiktoken
from typing import List, Dict
import re


# 中英文句子结束符
SENTENCE_ENDINGS = r'[.!?。！？；]'
# 中英文段落分隔
PARAGRAPH_SEPARATORS = r'\n\s*\n+'


class TextChunker:
    """文本分块器，使用tiktoken进行token计数，支持中英文"""

    def __init__(self, min_tokens: int = 300, max_tokens: int = 500):
        """
        初始化文本分块器

        Args:
            min_tokens: 每个块的最小token数
            max_tokens: 每个块的最大token数
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        # 使用cl100k_base编码 (GPT-4/Claude 使用的编码，对中文支持较好)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.encoding.encode(text))

    def split_by_sentences(self, text: str) -> List[str]:
        """
        按句子切分文本，支持中英文标点

        Args:
            text: 原始文本

        Returns:
            句子列表
        """
        # 使用正则切分，保留分隔符
        parts = re.split(f'({SENTENCE_ENDINGS}\\s*)', text)

        sentences = []
        current = ""

        for i, part in enumerate(parts):
            current += part
            # 如果是句子结束符后面的部分，或者是最后一部分
            if re.match(f'{SENTENCE_ENDINGS}\\s*$', part) or i == len(parts) - 1:
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        return sentences

    def split_by_paragraphs(self, text: str) -> List[str]:
        """
        按段落切分文本

        Args:
            text: 原始文本

        Returns:
            段落列表
        """
        # 按双换行符或段落标记分割
        paragraphs = re.split(PARAGRAPH_SEPARATORS, text.strip())
        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def chunk_text(self, pages: List[Dict]) -> List[Dict]:
        """
        将页面文本分块，确保每个块在300-500 tokens之间

        Args:
            pages: 页面列表，每个元素包含page_number和text

        Returns:
            文本块列表，每个元素包含text, page_number, n_tokens
        """
        chunks = []

        for page_info in pages:
            page_number = page_info['page_number']
            text = page_info['text']

            # 按段落分割
            paragraphs = self.split_by_paragraphs(text)

            current_chunk = ""
            current_tokens = 0

            for paragraph in paragraphs:
                para_tokens = self.count_tokens(paragraph)

                # 如果单个段落就超过max_tokens，需要进一步分割
                if para_tokens > self.max_tokens:
                    # 先保存当前chunk（如果有）
                    if current_chunk and current_tokens >= self.min_tokens:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page_number': page_number,
                            'n_tokens': current_tokens
                        })
                        current_chunk = ""
                        current_tokens = 0

                    # 分割超长段落（使用支持中英文的句子分割）
                    sentences = self.split_by_sentences(paragraph)
                    for sentence in sentences:
                        sent_tokens = self.count_tokens(sentence)

                        if current_tokens + sent_tokens > self.max_tokens:
                            if current_chunk and current_tokens >= self.min_tokens:
                                chunks.append({
                                    'text': current_chunk.strip(),
                                    'page_number': page_number,
                                    'n_tokens': current_tokens
                                })
                            current_chunk = sentence
                            current_tokens = sent_tokens
                        else:
                            # 使用空格连接英文，无空格连接中文
                            if current_chunk:
                                # 检查是否需要添加空格 (简单判断：如果上一段结尾或当前开头是ASCII字符)
                                if current_chunk[-1].isascii() or sentence[0].isascii():
                                    current_chunk += " " + sentence
                                else:
                                    current_chunk += sentence
                            else:
                                current_chunk = sentence
                            current_tokens += sent_tokens

                # 如果添加这个段落会超过max_tokens
                elif current_tokens + para_tokens > self.max_tokens:
                    # 保存当前chunk
                    if current_chunk and current_tokens >= self.min_tokens:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page_number': page_number,
                            'n_tokens': current_tokens
                        })
                    # 开始新chunk
                    current_chunk = paragraph
                    current_tokens = para_tokens

                else:
                    # 添加到当前chunk
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                    current_tokens += para_tokens

            # 保存最后一个chunk
            if current_chunk and current_tokens >= self.min_tokens:
                chunks.append({
                    'text': current_chunk.strip(),
                    'page_number': page_number,
                    'n_tokens': current_tokens
                })
            elif current_chunk:  # 如果不够min_tokens但还有内容，也保存
                chunks.append({
                    'text': current_chunk.strip(),
                    'page_number': page_number,
                    'n_tokens': current_tokens
                })

        print(f"文本分块完成，共生成 {len(chunks)} 个文本块")
        return chunks
