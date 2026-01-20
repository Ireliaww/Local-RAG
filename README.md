# RAG问答系统 - 财务报表分析

基于RAG（Retrieval-Augmented Generation）架构的财务报表问答系统，能够解析PDF财务报表，进行向量化存储，并实现语义搜索问答。

## 技术栈

- **PDF解析**: `unstructured` (逐页转换)
- **Tokenization**: `tiktoken` (cl100k_base编码)
- **Embedding**: Google Gemini `text-embedding-004`
- **向量数据库**: `chromadb` (Cosine Similarity)
- **LLM**: Google Gemini `gemini-2.5-flash` (用于问答生成)

## 项目结构

```
My-LLM-APP/
├── src/
│   ├── __init__.py
│   ├── pdf_processor/          # PDF处理模块
│   │   ├── __init__.py
│   │   ├── pdf_parser.py       # PDF下载和解析
│   │   └── text_chunker.py     # 文本分块（300-500 tokens）
│   ├── vector_store/           # 向量存储模块
│   │   ├── __init__.py
│   │   └── chroma_store.py     # ChromaDB向量存储
│   └── rag_qa/                 # RAG问答模块
│       ├── __init__.py
│       └── qa_engine.py        # RAG问答引擎
├── data/
│   └── pdfs/                   # PDF文件存储目录
├── chroma_db/                  # ChromaDB数据目录（自动创建）
├── main.py                     # 主程序入口
├── evaluation.py               # 评估脚本（10个预设问题）
├── requirements.txt            # 依赖包
├── .env.example                # 环境变量示例
└── README.md                   # 项目说明

```

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的Google API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```
GOOGLE_API_KEY=AIzaSyCX1YkKjWksGupxxxiS_Pbn2XpV26I2lwY
```

**获取Google API Key**: 访问 [Google AI Studio](https://aistudio.google.com/app/apikey) 创建API密钥

## 使用方法

### 1. 处理PDF并建立索引

```bash
# 从URL下载PDF并建立索引
python main.py --pdf "https://example.com/financial_report.pdf" --reindex

# 或使用本地PDF文件
python main.py --pdf "data/pdfs/report.pdf" --reindex
```

### 2. 交互式问答

```bash
# 使用现有索引进行问答（交互模式）
python main.py --pdf "data/pdfs/report.pdf"
```

### 3. 单个问题

```bash
python main.py --pdf "data/pdfs/report.pdf" --question "公司的总营收是多少？"
```

### 4. 运行评估脚本

```bash
# 运行10个预设问题的评估
python main.py --pdf "data/pdfs/report.pdf" --evaluate
```

## 工作流程

1. **PDF处理**
   - 下载或读取PDF文件
   - 使用`unstructured`逐页提取文本

2. **文本分块**
   - 按段落切分文本
   - 使用`tiktoken`确保每个块在300-500 tokens之间

3. **向量化存储**
   - 使用Google Gemini `text-embedding-004`生成向量
   - 存储到ChromaDB（Cosine Similarity）

4. **检索和问答**
   - 用户提问时，检索Top-K相关段落（默认k=5）
   - 过滤相似度低于阈值的结果（默认threshold=0.6）
   - 构造Prompt并调用Google Gemini LLM生成答案

## 评估问题

系统预设了10个评估问题，涵盖：
- 营收和利润
- 风险因素
- 财务指标（毛利率、资产负债率等）
- 业务板块
- 现金流
- 竞争对手
- 研发投入
- 发展战略

## 配置参数

### 文本分块参数
- `min_tokens`: 300（最小token数）
- `max_tokens`: 500（最大token数）

### 检索参数
- `k`: 5（检索的top-k数量）
- `threshold`: 0.6（相似度阈值）

### LLM参数
- `model`: gemini-2.5-flash（可修改为gemini-pro等）
- `temperature`: 0.3（降低温度以获得更准确的答案）

## 注意事项

1. 首次运行需要下载PDF并建立索引，可能需要一些时间
2. 确保有足够的Google API额度（Gemini API有免费额度）
3. ChromaDB数据会持久化在`chroma_db/`目录，如需重新索引可使用`--reindex`参数
4. PDF文件会保存在`data/pdfs/`目录
5. **注意**: 如果之前使用OpenAI API建立的索引，切换为Gemini后需要重新索引（使用`--reindex`参数），因为embedding模型不同

## 示例输出

```
问题: 公司的总营收是多少？

答案:
根据文档内容，公司在报告期内的总营收为XXX亿元，较去年同期增长XX%...

使用了 3 个相关文档片段

相关文档片段:
  [1] 第5页 (相似度: 0.852)
  [2] 第6页 (相似度: 0.789)
  [3] 第12页 (相似度: 0.734)
```

## 许可证

MIT License
