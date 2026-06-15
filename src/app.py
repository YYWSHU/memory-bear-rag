"""Streamlit Web 界面 - 记忆承载 知识检索助手"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
from datetime import datetime

# 延迟导入，避免streamlit初始化时加载大模型
@st.cache_resource
def load_components():
    """加载所有组件（缓存，只加载一次）"""
    from src.embedder import EmbeddingStore, load_chunks
    from src.knowledge_graph import KnowledgeGraph
    from src.retriever import MultiStageRetriever
    from src.generator import ThinkingStyleGenerator

    with st.spinner("正在加载向量数据库..."):
        embedding_store = EmbeddingStore()

    with st.spinner("正在加载知识图谱..."):
        kg = KnowledgeGraph()
        has_kg = kg.load()

    retriever = MultiStageRetriever(embedding_store, kg)
    generator = ThinkingStyleGenerator()

    return embedding_store, kg, retriever, generator


st.set_page_config(
    page_title="记忆承载 - AI助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 标题
st.title("🧠 记忆承载 - 碧树西风 AI 助手")
st.caption("基于 RAG + 知识图谱 + 思维结构 三层架构 | 模拟作者的思维逻辑")

# 侧边栏
with st.sidebar:
    st.header("📊 系统状态")
    st.info("""
    **架构层级：**
    - Layer 1: 向量语义检索
    - Layer 2: 知识图谱关联
    - Layer 3: 思维结构匹配
    """)

    st.divider()

    st.header("💡 示例问题")
    examples = [
        "怎么看当前的房地产市场？",
        "普通人怎么跨越阶层？",
        "赚钱的本质是什么？",
        "如何做职业选择？",
        "经济周期的规律是什么？",
        "教育和内卷的关系？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.current_query = ex

    st.divider()

    st.header("⚙️ 设置")
    show_sources = st.checkbox("显示原文来源", value=True)
    show_thinking = st.checkbox("显示思维推理链", value=True)
    show_graph = st.checkbox("显示知识图谱关联", value=True)

    st.divider()
    st.caption(f"Powered by DeepSeek V4 Pro | {datetime.now().strftime('%Y-%m-%d')}")

# 初始化session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# 加载组件
try:
    embedding_store, kg, retriever, generator = load_components()
    components_loaded = True
except Exception as e:
    st.error(f"组件加载失败: {e}")
    st.info("请先运行数据处理流程：\n```bash\ncd {项目目录}\npython src/pdf_processor.py\npython src/text_chunker.py\npython src/embedder.py\npython src/knowledge_graph.py\npython src/thinking_structure.py\n```")
    components_loaded = False

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and show_sources:
            with st.expander("📚 参考来源"):
                for src in msg["sources"]:
                    st.caption(f"《{src['title']}》（{src['date']}）- 相关度 {src['relevance']}")
        if "thinking_chain" in msg and show_thinking:
            with st.expander("🔗 思维推理链"):
                st.markdown(msg["thinking_chain"])

# 处理输入
query = st.chat_input("向碧树西风提问...")

# 处理预设问题
if st.session_state.current_query:
    query = st.session_state.current_query
    st.session_state.current_query = ""

if query:
    if not components_loaded:
        st.error("组件未加载，无法处理提问")
        st.stop()

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 生成回答
    with st.chat_message("assistant"):
        # Step 1: 检索
        with st.spinner("🔍 正在检索相关文章..."):
            retrieval_result = retriever.retrieve(query)
            context = retriever.format_context_for_llm(retrieval_result)

        # Step 2: 生成
        with st.spinner("💭 正在模拟作者思维..."):
            response_placeholder = st.empty()
            full_response = ""
            for token in generator.generate_stream(query, context):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        # Step 3: 展示来源和思维链
        if show_sources:
            with st.expander("📚 参考来源"):
                for src in retrieval_result.get("cited_articles", [])[:5]:
                    st.caption(f"《{src['title']}》（{src['date']}）- 相关度 {src['relevance']}")

        if show_thinking:
            thinking = retrieval_result.get("thinking_results", {})
            if thinking.get("frameworks") or thinking.get("argument_patterns"):
                with st.expander("🔗 思维推理链"):
                    for fw in thinking.get("frameworks", [])[:2]:
                        st.markdown(f"**分析框架: {fw.get('name', '')}**")
                        st.caption(fw.get('description', ''))
                    for ap in thinking.get("argument_patterns", [])[:2]:
                        st.markdown(f"**论证模式: {ap['title']}**")
                        if ap.get("premises"):
                            st.caption(f"前提: {'; '.join(ap['premises'][:2])}")
                        if ap.get("conclusions"):
                            st.caption(f"结论: {'; '.join(ap['conclusions'][:2])}")

        if show_graph:
            graph_results = retrieval_result.get("graph_results", {})
            entities = graph_results.get("entities", [])
            if entities:
                with st.expander("🕸️ 知识图谱关联"):
                    cols = st.columns(min(len(entities), 4))
                    for i, e in enumerate(entities[:8]):
                        with cols[i % 4]:
                            st.metric(
                                label=e['entity'],
                                value=e.get('type', ''),
                                delta=f"{e.get('article_count', 0)}篇文章"
                            )

        # 保存消息
        saved_sources = retrieval_result.get("cited_articles", [])[:5]

        # 构建思维链文本
        thinking_text = ""
        thinking = retrieval_result.get("thinking_results", {})
        for fw in thinking.get("frameworks", [])[:2]:
            thinking_text += f"**框架**: {fw.get('name', '')} - {fw.get('description', '')}\n\n"

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": saved_sources,
            "thinking_chain": thinking_text,
        })
