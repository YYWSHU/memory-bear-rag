import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""多阶段检索Pipeline - 向量检索 + 图谱遍历 + 思维模式匹配"""
import os
import json
from typing import List, Dict, Optional
from config import TOP_K_CHUNKS, TOP_K_GRAPH, THINKING_DIR, KG_JSON_PATH


class MultiStageRetriever:
    """
    三阶段检索：
    Stage 1: 向量语义检索（找到语义最相似的文本块）
    Stage 2: 知识图谱扩展（从检索结果中的实体出发，找到关联实体和因果链）
    Stage 3: 思维模式匹配（匹配相似的论证结构和分析框架）
    """

    def __init__(self, embedding_store, knowledge_graph):
        self.embedding_store = embedding_store
        self.knowledge_graph = knowledge_graph
        self.meta_patterns = self._load_meta_patterns()
        self.thinking_patterns = self._load_thinking_patterns()

    def _load_meta_patterns(self) -> dict:
        path = os.path.join(THINKING_DIR, "meta_patterns.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_thinking_patterns(self) -> List[dict]:
        path = os.path.join(THINKING_DIR, "thinking_patterns.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def stage1_vector_search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict]:
        """Stage 1: 向量语义检索"""
        return self.embedding_store.search(query, top_k=top_k)

    def stage2_graph_expand(self, vector_results: List[Dict], query: str) -> Dict:
        """Stage 2: 知识图谱扩展"""
        if not vector_results or not self.knowledge_graph.graph.number_of_nodes():
            return {"entities": [], "paths": [], "subgraph_nodes": []}

        # 从检索结果中提取出现的实体
        found_entities = set()
        for r in vector_results:
            text = r.get("text", "")
            # 在知识图谱中搜索文本中提及的实体
            for node in list(self.knowledge_graph.graph.nodes)[:500]:  # 限制搜索范围
                if len(node) >= 2 and node in text:
                    found_entities.add(node)

        # 搜索与查询相关的实体
        query_entities = self.knowledge_graph.search_entities(query, top_k=TOP_K_GRAPH)

        all_entities = list(found_entities) + [e["entity"] for e in query_entities]
        all_entities = list(dict.fromkeys(all_entities))[:20]  # 去重+限制

        # 获取子图
        subgraph = None
        if all_entities:
            subgraph = self.knowledge_graph.get_subgraph(all_entities, depth=1)

        # 查找关键路径
        paths = []
        for e1 in all_entities[:5]:
            entity_results = self.knowledge_graph.search_entities(e1, top_k=3)
            for er in entity_results:
                for related in er.get("related_entities", [])[:3]:
                    p = self.knowledge_graph.find_paths(e1, related, max_len=3)
                    paths.extend(p)

        return {
            "entities": query_entities,
            "paths": paths[:10],
            "subgraph_nodes": list(subgraph.nodes) if subgraph else [],
            "subgraph_edges": list(subgraph.edges) if subgraph else [],
        }

    def stage3_thinking_match(self, query: str, vector_results: List[Dict]) -> Dict:
        """Stage 3: 思维模式匹配"""
        if not self.thinking_patterns:
            return {"frameworks": [], "argument_patterns": [], "keywords": []}

        # 收集检索结果中涉及的文章
        article_filenames = set()
        for r in vector_results:
            fname = r.get("article_filename", "")
            if fname:
                article_filenames.add(fname)

        # 找到对应的思维结构
        matched_patterns = []
        for tp in self.thinking_patterns:
            if tp.get("filename", "") in article_filenames or tp.get("title", "") in article_filenames:
                matched_patterns.append(tp)

        frameworks = []
        argument_patterns = []
        all_keywords = []

        for mp in matched_patterns[:5]:
            fw = mp.get("framework", {})
            if fw.get("name"):
                frameworks.append(fw)

            arg = mp.get("argument_chain", {})
            if arg:
                argument_patterns.append({
                    "title": mp.get("title", mp.get("filename", "")),
                    "premises": arg.get("premises", []),
                    "reasoning": arg.get("reasoning_steps", []),
                    "conclusions": arg.get("conclusions", []),
                })

            all_keywords.extend(mp.get("keywords", []))

        # 去重关键词
        keyword_freq = {}
        for kw in all_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "frameworks": frameworks,
            "argument_patterns": argument_patterns,
            "top_keywords": [kw[0] for kw in top_keywords],
            "matched_article_count": len(matched_patterns),
        }

    def retrieve(self, query: str) -> Dict:
        """
        完整三阶段检索，返回统一检索结果
        """
        # Stage 1
        vector_results = self.stage1_vector_search(query)

        # Stage 2
        graph_results = self.stage2_graph_expand(vector_results, query)

        # Stage 3
        thinking_results = self.stage3_thinking_match(query, vector_results)

        # 整理引用的文章
        cited_articles = []
        seen = set()
        for r in vector_results:
            title = r.get("article_title", "")
            date = r.get("article_date", "")
            if title not in seen:
                seen.add(title)
                cited_articles.append({"title": title, "date": date, "relevance": f"{r['score']:.4f}"})

        return {
            "query": query,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "thinking_results": thinking_results,
            "cited_articles": cited_articles[:10],
        }

    def format_context_for_llm(self, retrieval_result: Dict) -> str:
        """
        将检索结果格式化为LLM的上下文，
        重点突出思维结构以引导LLM模拟作者思维
        """
        parts = []

        # 0. 元模式（作者整体思维特征）
        if self.meta_patterns:
            parts.append("## 作者思维特征（元模式）")
            frameworks = self.meta_patterns.get("top_frameworks", [])
            if frameworks:
                parts.append("**常用分析框架：**")
                for fw_name, count in frameworks[:5]:
                    parts.append(f"- {fw_name}（{count}篇文章使用）")

            keywords = self.meta_patterns.get("top_keywords", [])
            if keywords:
                parts.append(f"\n**标志性词汇：**{', '.join(kw[0] for kw in keywords[:30])}")

            rhetoric = self.meta_patterns.get("top_rhetoric", [])
            if rhetoric:
                parts.append(f"\n**典型修辞策略：**{', '.join(r[0] for r in rhetoric[:10])}")

        # 1. 思维结构（检索匹配到的）
        thinking = retrieval_result.get("thinking_results", {})
        frameworks = thinking.get("frameworks", [])
        if frameworks:
            parts.append("\n## 相关思维框架")
            for fw in frameworks[:3]:
                parts.append(f"- **{fw.get('name', '')}**: {fw.get('description', '')}")
                parts.append(f"  运用模式: {fw.get('pattern', '')}")

        arg_patterns = thinking.get("argument_patterns", [])
        if arg_patterns:
            parts.append("\n## 相关论证模式")
            for ap in arg_patterns[:3]:
                parts.append(f"\n### {ap['title']}")
                if ap.get("premises"):
                    parts.append(f"**前提:** {'; '.join(ap['premises'][:3])}")
                if ap.get("reasoning"):
                    parts.append(f"**推理:** {'; '.join(ap['reasoning'][:3])}")
                if ap.get("conclusions"):
                    parts.append(f"**结论:** {'; '.join(ap['conclusions'][:3])}")

        # 2. 原文段落
        parts.append("\n## 相关原文段落")
        for i, r in enumerate(retrieval_result.get("vector_results", [])[:8]):
            parts.append(f"\n### 段落 {i+1}（来自《{r['article_title']}》，{r['article_date']}）")
            parts.append(r["text"][:800])
            parts.append("")

        # 3. 知识图谱关联
        graph = retrieval_result.get("graph_results", {})
        entities = graph.get("entities", [])
        if entities:
            parts.append("\n## 相关概念与关联")
            for e in entities[:5]:
                parts.append(f"- **{e['entity']}**（{e.get('type', '')}）: 出现在 {e.get('article_count', 0)} 篇文章中")
                related = e.get("related_entities", [])[:5]
                if related:
                    parts.append(f"  关联: {', '.join(related)}")

        return "\n".join(parts)
