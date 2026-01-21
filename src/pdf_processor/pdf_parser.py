"""
PDF处理模块：下载PDF并提取文本

特性:
- 支持本地文件和HTTP(S) URL
- 下载超时控制和重试机制
- 使用pypdf进行轻量级PDF解析
"""
import os
import requests
from typing import List
from urllib.parse import urlparse
from pypdf import PdfReader

# 下载配置
DOWNLOAD_TIMEOUT = 60  # 下载超时(秒)
MAX_DOWNLOAD_RETRIES = 3


class PDFProcessingError(Exception):
    """PDF处理相关错误"""
    pass


class PDFProcessor:
    """PDF处理器，负责下载和解析PDF文件"""

    def __init__(self, data_dir: str = "data/pdfs"):
        """
        初始化PDF处理器

        Args:
            data_dir: PDF文件存储目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_pdf(self, url: str, filename: str = None) -> str:
        """
        下载PDF文件，带超时和重试机制

        Args:
            url: PDF文件的URL
            filename: 保存的文件名，如果为None则从URL提取

        Returns:
            保存的文件路径

        Raises:
            PDFProcessingError: 下载失败
        """
        if filename is None:
            # 从URL提取文件名
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.endswith(".pdf"):
                filename = "document.pdf"

        filepath = os.path.join(self.data_dir, filename)

        print(f"正在下载PDF: {url}")

        last_exception = None
        for attempt in range(MAX_DOWNLOAD_RETRIES):
            try:
                response = requests.get(
                    url,
                    stream=True,
                    timeout=DOWNLOAD_TIMEOUT,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; RAG-PDF-Processor/1.0)'}
                )
                response.raise_for_status()

                # 验证内容类型
                content_type = response.headers.get('content-type', '')
                if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                    print(f"警告: 内容类型可能不是PDF: {content_type}")

                # 下载文件
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r下载进度: {progress:.1f}%", end='', flush=True)

                print()  # 换行
                print(f"PDF已保存至: {filepath}")
                return filepath

            except requests.exceptions.Timeout:
                last_exception = PDFProcessingError(f"下载超时 (>{DOWNLOAD_TIMEOUT}秒)")
                print(f"下载超时，尝试 {attempt + 1}/{MAX_DOWNLOAD_RETRIES}")

            except requests.exceptions.RequestException as e:
                last_exception = PDFProcessingError(f"下载失败: {e}")
                print(f"下载失败: {e}，尝试 {attempt + 1}/{MAX_DOWNLOAD_RETRIES}")

        raise last_exception or PDFProcessingError("下载失败")

    def extract_text_from_pdf(self, pdf_path: str) -> List[dict]:
        """
        使用pypdf逐页提取PDF文本

        Args:
            pdf_path: PDF文件路径

        Returns:
            包含每页文本信息的列表，每个元素包含page_number和text

        Raises:
            PDFProcessingError: 解析失败
        """
        # 验证文件存在
        if not os.path.exists(pdf_path):
            raise PDFProcessingError(f"PDF文件不存在: {pdf_path}")

        # 验证文件大小
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            raise PDFProcessingError(f"PDF文件为空: {pdf_path}")

        print(f"正在解析PDF: {pdf_path} ({file_size / 1024 / 1024:.2f} MB)")

        try:
            reader = PdfReader(pdf_path)
            pages_text = []

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append({
                        'page_number': page_num,
                        'text': text
                    })

        except Exception as e:
            raise PDFProcessingError(f"PDF解析失败: {e}")

        if not pages_text:
            raise PDFProcessingError("PDF解析未提取到有效文本内容")

        print(f"成功提取 {len(pages_text)} 页内容")
        return pages_text

    def process_pdf(self, pdf_path_or_url: str) -> List[dict]:
        """
        处理PDF文件（如果是URL则先下载）

        Args:
            pdf_path_or_url: PDF文件路径或URL

        Returns:
            包含每页文本信息的列表

        Raises:
            PDFProcessingError: 处理失败
        """
        if pdf_path_or_url.startswith("http://") or pdf_path_or_url.startswith("https://"):
            pdf_path = self.download_pdf(pdf_path_or_url)
        else:
            pdf_path = pdf_path_or_url
            # 验证本地文件
            if not os.path.exists(pdf_path):
                raise PDFProcessingError(f"文件不存在: {pdf_path}")

        return self.extract_text_from_pdf(pdf_path)
