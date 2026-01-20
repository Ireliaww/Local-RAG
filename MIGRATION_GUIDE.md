# 从OpenAI迁移到Google Gemini指南

本文档说明如何将RAG问答系统从OpenAI API迁移到Google Gemini API。

## 主要变更

### 1. 依赖包变更

**之前 (OpenAI)**:
```txt
openai>=1.12.0
```

**现在 (Gemini)**:
```txt
google-generativeai>=0.3.0
```

### 2. 环境变量变更

**之前**:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**现在**:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

获取Google API Key: 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. 模型变更

| 功能 | OpenAI | Google Gemini |
|------|--------|---------------|
| Embedding | text-embedding-3-small | text-embedding-004 |
| LLM | gpt-4o-mini | gemini-2.5-flash |

### 4. 代码变更

#### Embedding模块
- **文件**: `src/vector_store/chroma_store.py`
- **变更**: 使用自定义的`GeminiEmbeddingFunction`替代`OpenAIEmbeddingFunction`
- **新增文件**: `src/vector_store/gemini_embedding.py`

#### LLM模块
- **文件**: `src/rag_qa/qa_engine.py`
- **变更**: 
  - 使用`google.generativeai`替代`openai`
  - 使用`genai.GenerativeModel().generate_content()`替代`client.chat.completions.create()`
  - Prompt格式略有不同（Gemini不需要单独的system message）

## 迁移步骤

### 步骤1: 更新依赖

```bash
pip install -r requirements.txt
```

### 步骤2: 更新环境变量

创建或更新`.env`文件：

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 步骤3: 重新索引（重要！）

**重要**: 由于embedding模型不同，之前使用OpenAI建立的向量索引无法直接使用。必须重新索引：

```bash
python main.py --pdf "your_pdf_path_or_url" --reindex
```

### 步骤4: 验证

运行测试问题验证系统正常工作：

```bash
python main.py --pdf "your_pdf_path_or_url" --question "测试问题"
```

## 注意事项

1. **向量维度不同**: OpenAI的`text-embedding-3-small`和Gemini的`text-embedding-004`向量维度不同，因此必须重新索引。

2. **API限制**: 
   - Gemini API有免费额度限制
   - 注意API调用频率限制
   - 检查token限制

3. **模型选择**: 
   - `gemini-2.5-flash`: 快速响应，适合大多数场景
   - `gemini-pro`: 更强的能力，但可能更慢
   - 可在`qa_engine.py`中修改`model_name`参数

4. **Prompt调整**: Gemini的prompt格式与OpenAI略有不同，如果发现答案质量下降，可能需要微调prompt。

5. **错误处理**: Gemini API的错误码和消息格式与OpenAI不同，如遇到问题请查看Google AI文档。

## 回退到OpenAI

如果需要回退到OpenAI，可以：

1. 恢复`requirements.txt`中的`openai`依赖
2. 恢复环境变量为`OPENAI_API_KEY`
3. 使用git恢复相关代码文件
4. 重新索引（因为embedding模型不同）

## 性能对比

| 指标 | OpenAI | Gemini |
|------|--------|--------|
| Embedding速度 | 中等 | 中等 |
| LLM响应速度 | 快 | 快（Flash版本） |
| 成本 | 按token计费 | 有免费额度 |
| 上下文窗口 | 大 | 大 |

## 支持

如有问题，请参考：
- [Google Gemini API文档](https://ai.google.dev/gemini-api/docs)
- [Google AI Studio](https://aistudio.google.com/)
