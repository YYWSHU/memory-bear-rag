"""统一Pipeline - 一键运行全流程"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from config import (
    EXTRACTED_TEXT_DIR, CHUNKS_DIR, KG_JSON_PATH,
    THINKING_DIR, VECTOR_DB_PATH
)


def check_step(step_name: str, check_path: str, check_type: str = "file") -> bool:
    """检查某步骤是否已完成"""
    if check_type == "file":
        return os.path.exists(check_path)
    elif check_type == "dir_not_empty":
        return os.path.isdir(check_path) and len(os.listdir(check_path)) > 0
    elif check_type == "index":
        if os.path.exists(check_path):
            with open(check_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            return index.get("success_count", 0) > 0
    return False


def run_pdf_extraction(force: bool = False):
    """Step 1: PDF文字提取"""
    index_path = os.path.join(EXTRACTED_TEXT_DIR, "_index.json")
    if check_step("PDF文字提取", index_path, "index") and not force:
        with open(index_path, "r") as f:
            index = json.load(f)
        print(f"[跳过] PDF文字已提取: {index['success_count']} 篇文章")
        return

    print("=" * 50)
    print("Step 1/5: PDF文字提取")
    print("=" * 50)
    from src.pdf_processor import collect_all_pdfs, extract_all_pdfs
    pdfs = collect_all_pdfs()
    print(f"共 {len(pdfs)} 个PDF文件")
    extract_all_pdfs(pdfs, force=force)


def run_text_chunking(force: bool = False):
    """Step 2: 文本分块"""
    chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    if check_step("文本分块", chunks_path) and not force:
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
        print(f"[跳过] 文本已分块: {len(chunks)} 个块")
        return

    print("=" * 50)
    print("Step 2/5: 文本分块")
    print("=" * 50)
    from src.text_chunker import chunk_all_articles
    chunk_all_articles(force=force)


def run_embedding(force: bool = False):
    """Step 3: 向量嵌入"""
    if check_step("向量嵌入", VECTOR_DB_PATH, "dir_not_empty") and not force:
        print(f"[跳过] 向量索引已存在")
        return

    print("=" * 50)
    print("Step 3/5: 向量嵌入 + ChromaDB索引")
    print("=" * 50)
    from src.embedder import EmbeddingStore, load_chunks
    chunks = load_chunks()
    if not chunks:
        print("[错误] 无文本块，先运行文本分块")
        return
    store = EmbeddingStore()
    store.build_index(chunks, force=force)

    # 测试检索
    print("\n[测试] 检索 '跨域阶层的方法'...")
    results = store.search("跨域阶层的方法", top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['article_title'][:40]}...")


def run_knowledge_graph(force: bool = False):
    """Step 4: 知识图谱构建"""
    if check_step("知识图谱", KG_JSON_PATH) and not force:
        print(f"[跳过] 知识图谱已存在")
        return

    print("=" * 50)
    print("Step 4/5: 知识图谱构建")
    print("=" * 50)
    from src.knowledge_graph import KnowledgeGraph
    from openai import OpenAI
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

    # 加载所有文章
    index_path = os.path.join(EXTRACTED_TEXT_DIR, "_index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    articles = []
    for info in index.get("articles", []):
        article_path = os.path.join(EXTRACTED_TEXT_DIR,
                                     info["filename"].replace(".pdf", ".json"))
        if os.path.exists(article_path):
            with open(article_path, "r") as f:
                articles.append(json.load(f))

    llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    llm_client.model = DEEPSEEK_MODEL

    kg = KnowledgeGraph()
    kg.build_from_articles(articles, llm_client=llm_client)

    # 测试
    results = kg.search_entities("房地产")
    print(f"\n[测试] 搜索'房地产': {len(results)} 个关联实体")


def run_thinking_structure(force: bool = False, max_articles: int = None):
    """Step 5: 思维结构提取"""
    meta_path = os.path.join(THINKING_DIR, "meta_patterns.json")
    patterns_path = os.path.join(THINKING_DIR, "thinking_patterns.json")

    if check_step("思维结构", meta_path) and not force:
        print(f"[跳过] 思维结构已提取")
        return

    print("=" * 50)
    print("Step 5/5: 思维结构提取（最耗时，需要调用DeepSeek API）")
    print("=" * 50)
    from src.thinking_structure import ThinkingStructureExtractor

    extractor = ThinkingStructureExtractor()
    patterns = extractor.extract_all_articles(force=force, max_articles=max_articles)
    meta = extractor.build_meta_patterns(patterns)

    print(f"\n[元模式]")
    print(f"  高频框架: {meta['top_frameworks'][:5]}")
    print(f"  高频关键词: {[kw[0] for kw in meta['top_keywords'][:20]]}")
    print(f"  修辞策略: {meta['top_rhetoric'][:5]}")


def run_test_query(query: str = "怎么看当前的房地产市场？"):
    """端到端测试查询"""
    print("=" * 50)
    print(f"端到端测试: {query}")
    print("=" * 50)

    from src.embedder import EmbeddingStore
    from src.knowledge_graph import KnowledgeGraph
    from src.retriever import MultiStageRetriever
    from src.generator import ThinkingStyleGenerator

    # 加载组件
    print("[加载] 向量数据库...")
    embedding_store = EmbeddingStore()

    print("[加载] 知识图谱...")
    kg = KnowledgeGraph()
    kg.load()

    print("[加载] 检索器...")
    retriever = MultiStageRetriever(embedding_store, kg)

    print("[加载] 生成器...")
    generator = ThinkingStyleGenerator()

    # 检索
    print("\n[检索] 三阶段检索中...")
    result = retriever.retrieve(query)
    context = retriever.format_context_for_llm(result)

    print(f"  检索到 {len(result['vector_results'])} 个文本块")
    print(f"  引用 {len(result['cited_articles'])} 篇文章")
    print(f"  匹配 {result['thinking_results'].get('matched_article_count', 0)} 个思维模式")

    # 生成
    print("\n[生成] DeepSeek V4 Pro 生成中...\n")
    print("-" * 50)
    response = generator.generate(query, context)
    print(response)
    print("-" * 50)

    return response


def main():
    parser = argparse.ArgumentParser(description="记忆承载 RAG 知识库 Pipeline")
    parser.add_argument("--step", type=str, default="all",
                       choices=["all", "pdf", "chunk", "embed", "kg", "thinking", "test"],
                       help="运行指定步骤")
    parser.add_argument("--force", action="store_true", help="强制重新运行")
    parser.add_argument("--max-thinking", type=int, default=None, help="思维提取最大文章数")
    parser.add_argument("--query", type=str, default="怎么看当前的房地产市场？",
                       help="测试查询")

    args = parser.parse_args()

    steps = {
        "all": ["pdf", "chunk", "embed", "kg", "thinking"],
        "pdf": ["pdf"],
        "chunk": ["chunk"],
        "embed": ["embed"],
        "kg": ["kg"],
        "thinking": ["thinking"],
        "test": ["test"],
    }

    for step in steps[args.step]:
        if step == "pdf":
            run_pdf_extraction(force=args.force)
        elif step == "chunk":
            run_text_chunking(force=args.force)
        elif step == "embed":
            run_embedding(force=args.force)
        elif step == "kg":
            run_knowledge_graph(force=args.force)
        elif step == "thinking":
            run_thinking_structure(force=args.force, max_articles=args.max_thinking)
        elif step == "test":
            run_test_query(args.query)

    print("\n✅ Pipeline 完成！")
    print("运行 Web 界面: streamlit run src/app.py")


if __name__ == "__main__":
    main()
