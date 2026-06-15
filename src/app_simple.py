"""记忆承载 AI 助手 - Streamlit Web 界面（Ollama嵌入 + DeepSeek生成）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json, requests
import chromadb
from openai import OpenAI
from config import *

st.set_page_config(page_title="记忆承载 AI 助手", page_icon="🧠", layout="wide")

# ---- 初始化 ----
@st.cache_resource
def init():
    vdb = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    col = vdb.get_collection("memory_bear_articles")

    kg_path = KG_JSON_PATH
    if os.path.exists(kg_path):
        with open(kg_path) as f: kg = json.load(f)
    else: kg = {"nodes":[], "edges":[]}

    meta = {}
    mp = os.path.join(THINKING_DIR, "meta_patterns.json")
    if os.path.exists(mp):
        with open(mp) as f: meta = json.load(f)

    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return col, kg, meta, llm

col, kg, meta, llm = init()

# ---- 侧边栏 ----
with st.sidebar:
    st.header("📊 系统状态")
    st.metric("向量索引", f"{col.count()}条")
    st.metric("知识图谱", f"{len(kg['nodes'])}节点/{len(kg['edges'])}边")
    st.metric("思维模式", f"{len(meta.get('top_frameworks',[]))}个框架")
    st.divider()
    st.header("💡 示例问题")
    for q in ["怎么才能跨越阶层？", "怎么看房价？", "普通人怎么投资？",
              "什么是真正的认知？", "教育和赚钱的关系？", "如何做职业选择？"]:
        if st.button(q, use_container_width=True):
            st.session_state.query = q

# ---- 主界面 ----
st.title("🧠 记忆承载 - 碧树西风 AI 助手")
st.caption("基于 RAG + 知识图谱 + 思维结构 | 49篇文章, 275万字 | Ollama bge-m3 + DeepSeek V4 Pro")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "query" not in st.session_state:
    st.session_state.query = ""

# 聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理输入
query = st.chat_input("向碧树西风提问...") or st.session_state.query
if query:
    st.session_state.query = ""
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 检索相关文章..."):
            # 嵌入查询
            q_resp = requests.post("http://localhost:11434/api/embed",
                                   json={"model":"bge-m3","input":[query]}, timeout=30)
            q_emb = q_resp.json()["embeddings"][0]
            results = col.query(query_embeddings=[q_emb], n_results=8,
                               include=["documents","metadatas","distances"])

        # 构建上下文
        ctx = []
        sources = []
        for i in range(len(results["ids"][0])):
            m = results["metadatas"][0][i]
            s = 1.0 - results["distances"][0][i]
            sources.append({"title": m["article_title"], "date": m["article_date"], "score": s})
            ctx.append(f"### 《{m['article_title']}》({m['article_date']})")
            ctx.append(results["documents"][0][i][:600])

        # 思维模式
        if meta:
            ctx.append("\n## 作者思维特征")
            ctx.append(f"分析框架: {', '.join(fw[0] for fw in meta.get('top_frameworks',[])[:5])}")
            ctx.append(f"修辞策略: {', '.join(r[0] for r in meta.get('top_rhetoric',[])[:5])}")

        context = "\n\n".join(ctx)

        # 生成
        with st.spinner("💭 模拟碧树西风思考中..."):
            resp = llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是碧树西风，微信公众号'记忆承载'的作者。用利益分析法、人性底层逻辑剖析问题。口语化但有深度，像跟老朋友聊天。先破后立，善于用类比。"},
                    {"role": "user", "content": f"基于以下文章内容，用碧树西风的风格回答。\n\n{context[:5000]}\n\n---\n读者问题：{query}"}
                ],
                temperature=0.7, max_tokens=2000, stream=True
            )

            placeholder = st.empty()
            answer = ""
            for chunk in resp:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
                    placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)

        # 来源
        with st.expander("📚 参考来源"):
            seen = set()
            for src in sources:
                key = src["title"]
                if key not in seen:
                    seen.add(key)
                    st.caption(f"《{src['title'][:50]}》({src['date']}) - 相关度 {src['score']:.3f}")

        # 思维特征
        if meta:
            with st.expander("🧩 作者思维特征"):
                st.caption(f"**分析框架**: {', '.join(fw[0] for fw in meta.get('top_frameworks',[])[:5])}")
                st.caption(f"**高频词**: {', '.join(kw[0] for kw in meta.get('top_keywords',[])[:15])}")
                st.caption(f"**修辞策略**: {', '.join(r[0] for r in meta.get('top_rhetoric',[])[:5])}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
