"""记忆承载 AI 助手 v2 - 会话管理 + 思维链 + 分支回退"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json, requests, time, copy, uuid
import chromadb
from openai import OpenAI
from datetime import datetime
from config import *

st.set_page_config(page_title="记忆承载 AI v2", page_icon="🧠", layout="wide")

# ============ 数据管理 ============
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

def load_sessions():
    """加载所有会话列表"""
    sessions = []
    for f in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if f.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, f)
            try:
                with open(path) as fp:
                    s = json.load(fp)
                sessions.append(s)
            except: pass
    return sessions

def save_session(session):
    """保存会话"""
    path = os.path.join(SESSIONS_DIR, f"{session['id']}.json")
    with open(path, "w") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

def create_session(name=None):
    """创建新会话"""
    sid = uuid.uuid4().hex[:12]
    session = {
        "id": sid,
        "name": name or f"对话 {datetime.now().strftime('%m-%d %H:%M')}",
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "messages": [],  # [{id, parent_id, role, content, reasoning, timestamp, edited}]
    }
    save_session(session)
    return session

def delete_session(sid):
    """删除会话"""
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if os.path.exists(path):
        os.remove(path)

# ============ 组件初始化 ============
@st.cache_resource
def init():
    vdb = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    col = vdb.get_collection("memory_bear_articles")
    with open(KG_JSON_PATH) as f: kg = json.load(f)
    mp = os.path.join(THINKING_DIR, "meta_patterns.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return col, kg, meta, llm

col, kg, meta, llm = init()

# ============ 检索函数 ============
def retrieve_and_generate(query, session_messages):
    """检索 + 生成，返回结果和思维链"""
    # 嵌入查询
    q_resp = requests.post("http://localhost:11434/api/embed",
                           json={"model":"bge-m3","input":[query]}, timeout=30)
    q_emb = q_resp.json()["embeddings"][0]
    results = col.query(query_embeddings=[q_emb], n_results=8,
                       include=["documents","metadatas","distances"])

    # 构建上下文
    ctx_parts = []
    sources = []
    for i in range(len(results["ids"][0])):
        m = results["metadatas"][0][i]
        s = 1.0 - results["distances"][0][i]
        sources.append({"title": m["article_title"], "date": m["article_date"], "score": s})
        ctx_parts.append(f"### 《{m['article_title']}》({m['article_date']})")
        ctx_parts.append(results["documents"][0][i][:600])

    if meta:
        ctx_parts.append("\n## 作者思维特征")
        ctx_parts.append(f"分析框架: {', '.join(fw[0] for fw in meta.get('top_frameworks',[])[:5])}")
        ctx_parts.append(f"修辞: {', '.join(r[0] for r in meta.get('top_rhetoric',[])[:5])}")

    context = "\n\n".join(ctx_parts)

    # 构建消息历史
    msgs = [{"role": "system", "content": "你是碧树西风，微信公众号'记忆承载'的作者。用利益分析法、人性底层逻辑剖析问题。口语化但有深度，像跟老朋友聊天。先破后立，善于用类比。回答中可以引用之前对话中的内容。"}]

    # 添加最近几轮对话
    for msg in session_messages[-6:]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        msgs.append({"role": role, "content": msg["content"]})

    # 添加当前查询
    msgs.append({"role": "user", "content": f"基于以下文章内容和之前的对话，回答问题。\n\n{context[:5000]}\n\n---\n读者问题：{query}"})

    # 调用DeepSeek（获取reasoning）
    resp = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=msgs,
        temperature=0.7, max_tokens=2000
    )

    answer = resp.choices[0].message.content
    reasoning = getattr(resp.choices[0].message, 'reasoning_content', '') or ''

    return answer, reasoning, sources

# ============ UI ============
# 会话管理
with st.sidebar:
    st.header("💬 会话管理")

    # 新会话
    new_name = st.text_input("新建会话", placeholder="输入名称...", key="new_session_name")
    if st.button("➕ 创建", use_container_width=True):
        s = create_session(new_name if new_name else None)
        st.session_state.current_session = s["id"]
        st.rerun()

    st.divider()

    # 会话列表
    sessions = load_sessions()
    if "current_session" not in st.session_state:
        st.session_state.current_session = sessions[0]["id"] if sessions else create_session()["id"]

    for s in sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            label = f"{'🟢' if s['id'] == st.session_state.current_session else '💬'} {s['name']}"
            if st.button(label, key=f"sess_{s['id']}", use_container_width=True):
                st.session_state.current_session = s["id"]
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{s['id']}"):
                delete_session(s["id"])
                if st.session_state.current_session == s["id"]:
                    st.session_state.current_session = None
                st.rerun()

    st.divider()
    st.metric("文章", f"{col.count()}条向量")
    st.metric("思维", f"{len(meta.get('top_frameworks',[]))}框架")

# ============ 主界面 ============
st.title("🧠 记忆承载 - 碧树西风 AI 助手 v2")

# 加载当前会话
current = None
for s in sessions:
    if s["id"] == st.session_state.current_session:
        current = s
        break
if not current:
    current = create_session()
    st.session_state.current_session = current["id"]

st.caption(f"📝 {current['name']} | {len(current['messages'])} 条消息")

# 构建消息树（支持分支）
messages = current.get("messages", [])
# parent_id -> children 映射
children_map = {}
root_ids = []
for i, msg in enumerate(messages):
    pid = msg.get("parent_id", "root")
    if pid == "root":
        root_ids.append(i)
    else:
        children_map.setdefault(pid, []).append(i)

# 按时间线展示（简化：线性展示 + 分支提示）
def render_messages(msgs, depth=0):
    for i, msg in enumerate(msgs):
        role = msg["role"]
        msg_id = msg.get("id", str(i))
        parent = msg.get("parent_id", "root")
        edited = msg.get("edited", False)
        branch_count = len(children_map.get(msg_id, []))

        with st.chat_message(role):
            # 分支标识
            if parent != "root" and i > 0:
                st.caption(f"↳ 回复 #{parent[:8]}" + (" [已编辑]" if edited else ""))

            st.markdown(msg["content"])

            # 思维链（助手消息）
            reasoning = msg.get("reasoning", "")
            if reasoning and role == "assistant":
                with st.expander("🔗 思维链"):
                    st.markdown(reasoning)

            # 操作按钮
            col_a, col_b, col_c = st.columns([1, 1, 8])
            with col_a:
                if st.button("✏️", key=f"edit_{msg_id}", help="编辑此消息"):
                    st.session_state.editing_msg = msg_id
                    st.session_state.edit_content = msg["content"]
                    st.rerun()

            with col_b:
                if st.button("🔀", key=f"branch_{msg_id}", help="从此处分支"):
                    st.session_state.branch_from = msg_id
                    st.rerun()

            # 分支提示
            if branch_count > 0:
                st.info(f"🔀 此消息有 {branch_count} 个分支回复")

# 如果正在编辑
if "editing_msg" in st.session_state:
    eid = st.session_state.editing_msg
    with st.container(border=True):
        st.subheader("✏️ 编辑消息")
        new_content = st.text_area("修改内容", value=st.session_state.edit_content, height=100, key="edit_area")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 保存修改", use_container_width=True):
                for msg in current["messages"]:
                    if msg.get("id") == eid:
                        msg["content"] = new_content
                        msg["edited"] = True
                        break
                current["updated"] = datetime.now().isoformat()
                save_session(current)
                del st.session_state.editing_msg
                st.rerun()
        with c2:
            if st.button("❌ 取消", use_container_width=True):
                del st.session_state.editing_msg
                st.rerun()

# 分支提示
if "branch_from" in st.session_state:
    bid = st.session_state.branch_from
    with st.container(border=True):
        st.subheader(f"🔀 从消息 #{bid[:8]} 创建分支")
        st.caption("新消息将作为此消息的备选回复分支")
        if st.button("❌ 取消分支", use_container_width=True):
            del st.session_state.branch_from
            st.rerun()

# 简易线性显示（有分支时显示主线路）
main_line = []
visited = set()

# 找主线路：从root开始，走第一条子消息
def trace_main_line(msgs, pid="root"):
    children = [m for m in msgs if m.get("parent_id", "root") == pid]
    if not children:
        return
    first = children[0]
    main_line.append(first)
    trace_main_line(msgs, first.get("id", ""))

trace_main_line(messages)
render_messages(main_line)

# 输入区
st.divider()
query = st.chat_input("向碧树西风提问...", key="chat_input")

if query:
    bid = st.session_state.get("branch_from", "root")
    if bid != "root":
        st.info(f"🔀 从 #{bid[:8]} 分支回复")
        del st.session_state.branch_from

    # 添加用户消息
    msg_id = uuid.uuid4().hex[:12]
    current["messages"].append({
        "id": msg_id,
        "parent_id": bid,
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat(),
    })
    current["updated"] = datetime.now().isoformat()
    save_session(current)

    # 检索 + 生成
    with st.chat_message("assistant"):
        with st.spinner("🔍 检索 + 💭 思考中..."):
            session_msgs = current["messages"]
            answer, reasoning, sources = retrieve_and_generate(query, session_msgs)

        st.markdown(answer)

        # 显示思维链
        if reasoning:
            with st.expander("🔗 思维链（DeepSeek V4 推理过程）"):
                st.markdown(reasoning)

        # 显示来源
        with st.expander("📚 参考来源"):
            seen = set()
            for src in sources:
                key = src["title"]
                if key not in seen:
                    seen.add(key)
                    st.caption(f"《{src['title'][:50]}》({src['date']}) - {src['score']:.3f}")

    # 保存助手消息
    reply_id = uuid.uuid4().hex[:12]
    current["messages"].append({
        "id": reply_id,
        "parent_id": msg_id,
        "role": "assistant",
        "content": answer,
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat(),
    })
    current["updated"] = datetime.now().isoformat()
    save_session(current)
    st.rerun()
