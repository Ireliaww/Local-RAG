# Hallucination Detection & Response Control

## 新功能概述

QA系统现在可以区分日常对话和文档相关问题，并智能处理响应：

- ✅ **日常对话**（如"你好"、"你是谁"）→ 自然回答
- ✅ **文档问题但找不到答案** → 返回"文档中暂未找到"
- ✅ **文档问题且有相关内容** → 基于文档回答并引用来源
- ✅ **可选的幻觉检测** → 验证答案是否基于文档

## 快速开始

### 1. 正常使用

无需任何配置更改，系统会自动：
1. 判断问题类型
2. 选择合适的响应策略
3. 显示调试信息（如果启用）

### 2. 查看问题分类

在Gradio界面中勾选"Show technical details"，你会看到：

```
Debug Info
- 💬 Question Type: CASUAL  (或 📊 DOCUMENT_QUERY)
- Similarity Threshold: 0.6
- Retrieved Chunks: 5
- Relevant Chunks: 0
- Hallucination Check: Disabled
```

### 3. 配置选项

编辑 `src/rag_qa/qa_engine.py`:

```python
# 相似度阈值 - 调高会更严格
SIMILARITY_THRESHOLD = 0.6

# 启用幻觉检测 - 更保守但更慢（需额外API调用）
ENABLE_HALLUCINATION_CHECK = False  # 改为 True 启用
```

## 测试

### 运行测试套件

```bash
cd /Users/ericwang/LLM-Practice/My-LLM-APP
python tests/test_hallucination.py
```

### 测试场景

**日常对话测试**:
```
Q: Hello
A: Hello! I'm your financial document assistant...
```

**有答案的文档查询**:
```
Q: What was NVIDIA's Q3 revenue?
A: Based on the financial statements, NVIDIA's revenue was $30.04 billion...
   Sources: NVIDIA_Q3_2024.pdf (Page 5)
```

**无答案的文档查询**:
```
Q: What is the CEO's favorite color?
A: I apologize, but I couldn't find relevant information about this 
   in the available documents...
```

## 技术细节

### 问题分类

使用 **gemini-2.5-flash** 进行快速分类：
- 速度快（比2.5-pro快）
- 成本低
- 无thinking mode开销

### 响应策略

```
CASUAL问题
  └→ 直接生成答案（无需文档）

DOCUMENT_QUERY
  ├→ 无相关文档（相似度 < 0.6）
  │   └→ 返回"未找到"消息
  └→ 有相关文档
      └→ 基于文档生成答案
          └→ [可选] 幻觉检测
```

### 文件更改

**核心修改**:
- `src/rag_qa/qa_engine.py`
  - `_classify_question()` - 问题分类
  - `_check_answer_relevance()` - 幻觉检测
  - `answer_question()` - 智能响应逻辑

**界面修改**:
- `app.py`
  - 更新调试输出，显示问题类型

**测试文件**:
- `tests/test_hallucination.py` - 完整测试套件
- `tests/test_classification_quick.py` - 快速分类测试

## 常见问题

### Q: 为什么有些日常问题被分类为DOCUMENT_QUERY？
A: 分类模型可能对模糊的问题保守处理。可以查看debug输出确认分类结果。

### Q: 如何调整相似度阈值？
A: 修改 `qa_engine.py` 中的 `SIMILARITY_THRESHOLD`：
   - 提高（如0.7）→ 更严格，更多"未找到"
   - 降低（如0.5）→ 更宽松，使用更多文档

### Q: 幻觉检测什么时候使用？
A: 默认禁用。如果你发现答案经常包含文档中没有的信息，可以启用 `ENABLE_HALLUCINATION_CHECK = True`

### Q: 会增加多少API成本？
A: 每个问题增加1次分类调用（gemini-2.5-flash，很便宜）。如果启用幻觉检测，再增加1次验证调用。

## 示例对话

```
User: Hi there!
Bot: Hello! I'm a financial document assistant...
[Debug: Question Type: CASUAL]

User: What was NVIDIA's revenue in Q3 2024?
Bot: Based on the financial statements, NVIDIA's revenue 
     for Q3 2024 was $30.04 billion...
     Sources: NVIDIA_Q3_2024.pdf (Page 5)
[Debug: Question Type: DOCUMENT_QUERY, Relevant: 3]

User: What's the CEO's phone number?
Bot: I apologize, but I couldn't find relevant information 
     about this in the available documents...
[Debug: Question Type: DOCUMENT_QUERY, Relevant: 0]
```

## 下一步

1. 启动Gradio测试: `python app.py`
2. 尝试不同类型的问题
3. 查看debug输出了解系统行为
4. 根据需要调整配置

有问题？查看 `walkthrough.md` 获取详细实现说明。
