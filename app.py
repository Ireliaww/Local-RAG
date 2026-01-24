"""
RAG问答系统 - Gradio Web界面

功能:
- 文档管理：上传PDF、查看已索引文档
- 智能问答：Knowledge-Enhanced RAG问答
- 清爽UI：主要显示自然回答，debug信息折叠
"""
import os
import shutil
import gradio as gr
from dotenv import load_dotenv

from src.indexer import DocumentIndexer
from src.rag_qa import RAGQAEngine

load_dotenv()

# 全局组件
indexer = None
qa_engine = None


def init_system():
    """初始化系统组件"""
    global indexer, qa_engine

    if indexer is None:
        indexer = DocumentIndexer()
        vector_store = indexer.vector_store
        qa_engine = RAGQAEngine(vector_store)

    return indexer, qa_engine


def get_index_status():
    """获取索引状态的格式化字符串"""
    init_system()
    status = indexer.get_status()

    status_text = f"""### Index Status
- **Total PDFs**: {status['total_pdfs']}
- **Indexed**: {status['indexed_pdfs']}
- **Pending**: {status['pending_pdfs']}
- **Total Chunks**: {status['total_chunks']}

### Indexed Files
"""
    if status['indexed_files']:
        for f in status['indexed_files']:
            status_text += f"- {f}\n"
    else:
        status_text += "_No files indexed yet_"

    return status_text


def upload_and_index(files):
    """上传并索引PDF文件"""
    init_system()

    if not files:
        return "No files uploaded", get_index_status()

    results = []
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join(indexer.pdf_dir, filename)
        shutil.copy(file.name, dest_path)
        result = indexer.index_single_file(dest_path)
        results.append(f"- **{filename}**: {result['status']}")

    return "\n".join(results), get_index_status()


def reindex_all():
    """重新索引所有文件"""
    init_system()
    results = indexer.index_all(force=True)

    if not results:
        return "No PDF files found in data/pdfs folder", get_index_status()

    summary = []
    for r in results:
        if r['status'] == 'success':
            summary.append(f"- **{r['file']}**: {r['chunks']} chunks")
        else:
            summary.append(f"- **{r['file']}**: {r['status']} - {r.get('message', '')}")

    return "\n".join(summary), get_index_status()


def index_new_files():
    """索引新文件（增量）"""
    init_system()
    results = indexer.index_all(force=False)

    if not results:
        return "All files already indexed", get_index_status()

    summary = []
    for r in results:
        if r['status'] == 'success':
            summary.append(f"- **{r['file']}**: {r['chunks']} chunks")
        elif r['status'] == 'skipped':
            summary.append(f"- **{r['file']}**: already indexed")
        else:
            summary.append(f"- **{r['file']}**: {r['status']} - {r.get('message', '')}")

    return "\n".join(summary), get_index_status()


def answer_question(question, history, show_debug):
    """处理用户问题并返回答案"""
    init_system()

    if not question.strip():
        return history, "", ""

    # 获取答案 - 始终调用LLM
    result = qa_engine.answer_question(question, k=5)

    # 主要回答 - 清爽自然
    answer = result['answer']

    # Debug信息
    debug_info = ""
    if show_debug:
        debug = result.get('debug', {})
        relevant_chunks = result.get('relevant_chunks', [])

        debug_info = f"""---
**Debug Info** (Threshold: {debug.get('threshold', 0.6)})
- Retrieved: {debug.get('total_retrieved', 0)} chunks
- Relevant (above threshold): {debug.get('relevant_count', 0)} chunks
- Avg Similarity: {debug.get('avg_similarity', 0):.3f}

"""
        if relevant_chunks:
            debug_info += "**Sources Used:**\n"
            for chunk in relevant_chunks[:3]:
                source = chunk.get('source_file', 'Unknown')
                page = chunk.get('page_number', '?')
                sim = chunk.get('similarity', 0)
                debug_info += f"- {source} (Page {page}) - Similarity: {sim:.3f}\n"
        else:
            debug_info += "_No relevant documents found - answered using general knowledge_"

        answer = answer + "\n\n" + debug_info

    # Use tuple format for Gradio 4.0 compatibility
    history.append((question, answer))
    return history, "", debug_info


def clear_chat():
    """清空对话历史"""
    return [], "", ""


# 创建Gradio界面
with gr.Blocks(title="RAG Document Q&A") as app:
    gr.Markdown("""
    # Document Q&A Assistant
    A knowledge-enhanced AI assistant with access to your documents.
    """)

    with gr.Tabs():
        # Q&A Tab
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask anything...",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                show_debug = gr.Checkbox(label="Show Debug Info", value=False)
                clear_btn = gr.Button("Clear Chat")

            # Debug输出区域（折叠）
            with gr.Accordion("Last Query Debug Details", open=False):
                debug_output = gr.Markdown("")

            # 事件绑定
            submit_btn.click(
                answer_question,
                inputs=[question_input, chatbot, show_debug],
                outputs=[chatbot, question_input, debug_output]
            )
            question_input.submit(
                answer_question,
                inputs=[question_input, chatbot, show_debug],
                outputs=[chatbot, question_input, debug_output]
            )
            clear_btn.click(clear_chat, outputs=[chatbot, question_input, debug_output])

        # Document Management Tab
        with gr.TabItem("Manage Documents"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload PDF Files")
                    file_upload = gr.File(
                        label="Select PDF files",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    upload_btn = gr.Button("Upload & Index", variant="primary")
                    upload_result = gr.Markdown()

                    gr.Markdown("### Quick Actions")
                    with gr.Row():
                        index_new_btn = gr.Button("Index New Files")
                        reindex_btn = gr.Button("Reindex All", variant="secondary")

                with gr.Column(scale=1):
                    status_display = gr.Markdown(get_index_status())
                    refresh_btn = gr.Button("Refresh Status")

            # 事件绑定
            upload_btn.click(
                upload_and_index,
                inputs=[file_upload],
                outputs=[upload_result, status_display]
            )
            index_new_btn.click(
                index_new_files,
                outputs=[upload_result, status_display]
            )
            reindex_btn.click(
                reindex_all,
                outputs=[upload_result, status_display]
            )
            refresh_btn.click(
                get_index_status,
                outputs=[status_display]
            )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Starting RAG Document Q&A Assistant...")
    print("Open in browser: http://127.0.0.1:7860")
    print("=" * 50 + "\n")

    # For Hugging Face Spaces compatibility
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
