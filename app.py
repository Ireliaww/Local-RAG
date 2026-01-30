"""
RAG问答系统 - Premium Financial Terminal UI
Features:
- Sophisticated dark terminal aesthetic
- LangChain conversation memory for context-aware responses
- Beautiful animations and micro-interactions
- Production-grade Gradio interface
"""
import os
import shutil
import gradio as gr
from dotenv import load_dotenv

from src.indexer import DocumentIndexer
from src.rag_qa import RAGQAEngine

load_dotenv()

# Global components
indexer = None
qa_engine = None

# Clean & Bright UI CSS
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-white: #ffffff;
    --bg-light: #f8fafc;
    --bg-gray: #f1f5f9;
    --border-light: #e2e8f0;
    --border-medium: #cbd5e1;
    --text-dark: #1e293b;
    --text-medium: #475569;
    --text-light: #64748b;
    --accent-blue: #3b82f6;
    --accent-blue-light: #eff6ff;
    --accent-blue-hover: #2563eb;
    --accent-green: #22c55e;
    --accent-green-light: #f0fdf4;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
}

/* Base */
.gradio-container {
    background: var(--bg-light) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Header */
.header-box {
    background: var(--bg-white);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-light);
    text-align: center;
}

.header-box h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-dark);
    margin: 0 0 0.5rem 0;
}

.header-box p {
    font-size: 0.95rem;
    color: var(--text-light);
    margin: 0;
}

/* Tabs */
.tab-nav {
    background: var(--bg-white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin-bottom: 1rem !important;
}

.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text-medium) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.625rem 1.25rem !important;
    transition: all 0.15s ease !important;
}

.tab-nav button:hover {
    background: var(--bg-gray) !important;
    color: var(--text-dark) !important;
}

.tab-nav button.selected {
    background: var(--accent-blue) !important;
    color: white !important;
}

/* Chat Container */
.chat-wrapper {
    background: var(--bg-white);
    border-radius: 16px;
    border: 1px solid var(--border-light);
    overflow: hidden;
}

.chatbot {
    background: var(--bg-white) !important;
    border: none !important;
}

.chatbot .message {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9375rem !important;
    line-height: 1.6 !important;
    padding: 1rem 1.25rem !important;
    margin: 0.75rem !important;
    border-radius: 12px !important;
}

.chatbot .user {
    background: var(--accent-blue) !important;
    color: white !important;
    margin-left: 20% !important;
}

.chatbot .bot {
    background: var(--bg-gray) !important;
    color: var(--text-dark) !important;
    margin-right: 20% !important;
    border: 1px solid var(--border-light) !important;
}

/* Input Area - PROMINENT */
.input-container {
    background: var(--bg-white);
    border: 2px solid var(--accent-blue);
    border-radius: 16px;
    padding: 0.5rem;
    margin-top: 1rem;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1);
}

textarea, input[type="text"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    background: transparent !important;
    border: none !important;
    color: var(--text-dark) !important;
    padding: 0.75rem 1rem !important;
}

textarea::placeholder, input::placeholder {
    color: var(--text-light) !important;
}

textarea:focus, input:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Send Button - PROMINENT */
button.primary {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    background: var(--accent-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.875rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

button.primary:hover {
    background: var(--accent-blue-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

button.secondary {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    background: var(--bg-white) !important;
    color: var(--text-medium) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 10px !important;
    padding: 0.625rem 1rem !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

button.secondary:hover {
    background: var(--bg-gray) !important;
    border-color: var(--text-light) !important;
}

/* Memory Badge */
.memory-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--accent-green);
    background: var(--accent-green-light);
    padding: 0.375rem 0.75rem;
    border-radius: 20px;
    border: 1px solid rgba(34, 197, 94, 0.2);
}

/* Status Panel */
.status-card {
    background: var(--bg-white);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 1.25rem;
}

.status-card h3 {
    font-size: 0.8125rem;
    font-weight: 600;
    color: var(--text-dark);
    text-transform: uppercase;
    letter-spacing: 0.025em;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-light);
}

/* Accordion */
.accordion {
    background: var(--bg-white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 10px !important;
    margin-top: 0.75rem !important;
}

/* Checkbox */
input[type="checkbox"] {
    accent-color: var(--accent-blue) !important;
}

/* Upload */
.upload-box {
    background: var(--bg-light);
    border: 2px dashed var(--border-medium);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: all 0.15s ease;
}

.upload-box:hover {
    border-color: var(--accent-blue);
    background: var(--accent-blue-light);
}

/* Footer */
.footer-text {
    font-size: 0.75rem;
    color: var(--text-light);
    text-align: center;
    padding: 1.5rem 0;
}

/* Labels */
label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    color: var(--text-medium) !important;
}

/* Markdown */
.markdown-content h3 {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-dark);
    margin: 1rem 0 0.5rem 0;
}

.markdown-content strong {
    color: var(--text-dark);
    font-weight: 600;
}
"""

def init_system():
    """Initialize system components"""
    global indexer, qa_engine

    if indexer is None:
        indexer = DocumentIndexer()
        vector_store = indexer.vector_store
        qa_engine = RAGQAEngine(vector_store)

    return indexer, qa_engine


def get_index_status():
    """Get formatted index status string"""
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
    """Upload and index PDF files"""
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
    """Reindex all files"""
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
    """Index new files (incremental)"""
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
    """Process user question and return answer with chat history context"""
    init_system()

    if not question.strip():
        return history, "", ""

    # Build conversation context from history for LangChain-style memory
    conversation_context = ""
    if history and len(history) > 0:
        recent_history = history[-6:]  # Last 3 exchanges (6 messages)
        for msg in recent_history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n\n"

    # Get answer with conversation context
    result = qa_engine.answer_question(
        question,
        k=5,
        conversation_history=conversation_context if conversation_context else None
    )

    # Main answer
    answer = result['answer']
    relevant_chunks = result.get('relevant_chunks', [])

    # Add source info when relevant documents are found
    if relevant_chunks:
        sources_info = "\n\n**Sources:**\n"
        seen_sources = set()
        for chunk in relevant_chunks[:3]:
            source = chunk.get('source_file', 'Unknown')
            page = chunk.get('page_number', '?')
            source_key = f"{source}_p{page}"
            if source_key not in seen_sources:
                sources_info += f"- {source} (Page {page})\n"
                seen_sources.add(source_key)
        answer = answer + sources_info

    # Debug info
    debug_info = ""
    if show_debug:
        debug = result.get('debug', {})
        
        # Question type info
        question_type = debug.get('question_type', 'Unknown')
        question_type_emoji = "💬" if question_type == "CASUAL" else "📊"
        
        debug_info = f"""---
**Debug Info**
- {question_type_emoji} **Question Type**: {question_type}
- **Similarity Threshold**: {debug.get('threshold', 0.6)}
- **Retrieved Chunks**: {debug.get('total_retrieved', 0)}
- **Relevant Chunks** (above threshold): {debug.get('relevant_count', 0)}
- **Avg Similarity**: {debug.get('avg_similarity', 0):.3f}
- **Memory Context**: {len(history)} messages
- **Hallucination Check**: {'Enabled' if debug.get('hallucination_check', False) else 'Disabled'}
"""
        
        if debug.get('hallucination_detected', False):
            debug_info += "\n⚠️ **Hallucination Detected**: Answer did not match document content\n"
        
        if relevant_chunks:
            debug_info += "\n**Sources Used:**\n"
            for chunk in relevant_chunks[:3]:
                source = chunk.get('source_file', 'Unknown')
                page = chunk.get('page_number', '?')
                sim = chunk.get('similarity', 0)
                debug_info += f"- {source} (Page {page}) - Similarity: {sim:.3f}\n"
        else:
            if question_type == "CASUAL":
                debug_info += "\n_Casual conversation - no documents needed_"
            else:
                debug_info += "\n_No relevant documents found - returned 'not found' message_"

        answer = answer + "\n\n" + debug_info


    # Update chat history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, "", debug_info


def clear_chat():
    """Clear conversation history"""
    return [], "", ""


def get_memory_status(history):
    """Get memory status indicator"""
    msg_count = len(history) if history else 0
    if msg_count == 0:
        return "◉ Memory: Empty"
    return f"◉ Memory: {msg_count} messages"


# Build Gradio Interface
with gr.Blocks(title="Financial Document Q&A") as app:

    # Header
    gr.HTML("""
        <div class="header-box">
            <h1>📄 Document Q&A Assistant</h1>
            <p>Ask questions about your uploaded documents. I'll find the answers for you.</p>
        </div>
    """)

    with gr.Tabs() as tabs:
        # Chat Tab
        with gr.TabItem("💬 Chat", id="chat"):
            # Memory status
            memory_status = gr.HTML(
                '<div class="memory-badge">💭 New conversation</div>'
            )

            chatbot = gr.Chatbot(
                label="",
                height=420,
                show_label=False,
                placeholder="Your conversation will appear here..."
            )

            # Prominent input area
            gr.HTML('<p style="margin: 1rem 0 0.5rem 0; font-weight: 600; color: #1e293b;">Ask your question:</p>')

            with gr.Row(elem_classes=["input-container"]):
                question_input = gr.Textbox(
                    label="",
                    placeholder="Type your question here and press Enter or click Send...",
                    scale=5,
                    show_label=False,
                    container=False,
                    lines=2
                )
                submit_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    min_width=120
                )

            with gr.Row():
                show_debug = gr.Checkbox(
                    label="Show technical details",
                    value=False
                )
                clear_btn = gr.Button(
                    "Clear Chat",
                    variant="secondary",
                    size="sm"
                )

            with gr.Accordion("Debug Details", open=False):
                debug_output = gr.Markdown("", elem_classes=["markdown-content"])

            # Event bindings
            def process_and_update_memory(question, history, show_debug):
                history, q, debug = answer_question(question, history, show_debug)
                msg_count = len(history) if history else 0
                if msg_count == 0:
                    memory_html = '<div class="memory-badge">💭 New conversation</div>'
                else:
                    memory_html = f'<div class="memory-badge">💬 {msg_count} messages in memory</div>'
                return history, q, debug, memory_html

            submit_btn.click(
                process_and_update_memory,
                inputs=[question_input, chatbot, show_debug],
                outputs=[chatbot, question_input, debug_output, memory_status]
            )
            question_input.submit(
                process_and_update_memory,
                inputs=[question_input, chatbot, show_debug],
                outputs=[chatbot, question_input, debug_output, memory_status]
            )

            def clear_and_reset_memory():
                return [], "", "", '<div class="memory-badge">💭 New conversation</div>'

            clear_btn.click(
                clear_and_reset_memory,
                outputs=[chatbot, question_input, debug_output, memory_status]
            )

        # Document Management Tab
        with gr.TabItem("📁 Documents", id="docs"):
            gr.HTML("""
                <div style="margin-bottom: 1rem; padding: 1rem; background: #eff6ff; border-radius: 10px; border: 1px solid #bfdbfe;">
                    <p style="margin: 0; color: #1e40af; font-size: 0.9rem;">
                        📌 Upload your PDF documents here. They will be indexed and searchable in the Chat tab.
                    </p>
                </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload PDF Files")
                    file_upload = gr.File(
                        label="Select PDF files",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    upload_btn = gr.Button(
                        "Upload & Index",
                        variant="primary"
                    )
                    upload_result = gr.Markdown()

                    gr.Markdown("### Quick Actions")
                    with gr.Row():
                        index_new_btn = gr.Button(
                            "Index New Files",
                            variant="secondary"
                        )
                        reindex_btn = gr.Button(
                            "Reindex All",
                            variant="secondary"
                        )

                with gr.Column(scale=1, elem_classes=["status-card"]):
                    status_display = gr.Markdown(get_index_status())
                    refresh_btn = gr.Button(
                        "Refresh",
                        variant="secondary",
                        size="sm"
                    )

            # Event bindings
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

    # Footer
    gr.HTML("""
        <p class="footer-text">
            Powered by Gemini 2.5 Pro · ChromaDB · Conversation Memory
        </p>
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Financial Document Intelligence Terminal")
    print("  Open in browser: http://127.0.0.1:7860")
    print("=" * 60 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS
    )
