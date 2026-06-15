import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""知识图谱模块 - 实体识别 + 关系抽取 + 图谱构建"""
import os
import json
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import networkx as nx
from config import KG_JSON_PATH, KG_DIR, CHUNKS_DIR


class KnowledgeGraph:
    """知识图谱：从文章中提取实体和关系，构建结构化知识网络"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_index = defaultdict(list)  # entity_name -> [article_refs]
        self.relation_index = defaultdict(list)  # (entity1, entity2) -> [relation_descriptions]

    def extract_entities_rule(self, text: str) -> List[Dict]:
        """
        基于规则的中文实体提取。
        提取：人名、地名、机构名、核心概念、经济术语等
        """
        entities = []

        # 常见中文命名实体模式
        patterns = {
            "人名": [
                r'(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',  # 英文名
                r'(?:[司马欧欧阳令狐上官慕容南宫独孤]+)',  # 复姓开头
            ],
            "概念术语": [
                r'(?:[A-Z]{2,})',  # 缩写如 AI, GDP
                r'(?:[的之])?(?:周期|杠杆|资产|负债|现金流|套利|对冲|边际|通胀|通缩|流动性|风险偏好|机会成本)',
            ],
            "经济名词": [
                r'(?:房地产|股市|基金|债券|期货|外汇|数字货币|量化|定投|分红|PE|PB|ROE)',
            ]
        }

        # 简化方案：用书名号、引号、括号提取概念
        quoted = re.findall(r'《([^》]+)》|"([^"]+)"|「([^」]+)」', text)
        for groups in quoted:
            concept = next((g for g in groups if g), "")
            if len(concept) > 1 and len(concept) < 30:
                entities.append({"name": concept, "type": "概念引用", "source": "引号提取"})

        # 提取冒号前的主题词
        topic_pattern = re.findall(r'(?:^|\n)([^\n：:]{2,20})(?:：|:)', text)
        for topic in topic_pattern:
            t = topic.strip()
            if 2 <= len(t) <= 20:
                entities.append({"name": t, "type": "主题", "source": "冒号提取"})

        return entities

    def extract_entities_llm(self, text: str, llm_client=None) -> List[Dict]:
        """
        使用LLM提取命名实体和关键概念。
        如果llm_client为None，回退到规则提取。
        """
        if llm_client is None:
            return self.extract_entities_rule(text)

        prompt = f"""请从以下文本中提取关键实体和概念，返回JSON格式。

文本：
{text[:3000]}

请提取：
1. 人名/群体（具体的人或人群）
2. 核心概念（文章讨论的核心经济/社会/哲学概念）
3. 事件/现象（文章提到的重要事件）
4. 书籍/理论（引用的书籍或理论）

返回严格JSON格式：
```json
{{
  "entities": [
    {{"name": "实体名称", "type": "人名/核心概念/事件/书籍理论", "重要性": "high/medium/low", "说明": "为什么这个实体重要"}}
  ]
}}
```"""

        try:
            response = llm_client.chat.completions.create(
                model=llm_client.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            content = response.choices[0].message.content

            # 提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get("entities", [])
            else:
                # 尝试直接解析
                data = json.loads(content)
                return data.get("entities", [])
        except Exception as e:
            print(f"[KG实体提取错误] {e}")
            return self.extract_entities_rule(text)

    def extract_relations_llm(self, text: str, entities: List[Dict], llm_client=None) -> List[Dict]:
        """使用LLM提取实体间的关系"""
        if llm_client is None or len(entities) < 2:
            return []

        entity_names = [e["name"] for e in entities[:10]]  # 最多10个实体
        entity_list = ", ".join(entity_names)

        prompt = f"""请分析以下文本中实体之间的关系，返回JSON格式。

文本：
{text[:3000]}

实体列表：{entity_list}

请识别实体之间的重要关系，包括：
- 因果关系（A导致B）
- 对立关系（A与B矛盾）
- 支撑关系（A是B的基础/前提）
- 类比关系（A类似于B）

返回严格JSON格式：
```json
{{
  "relations": [
    {{"source": "实体A", "target": "实体B", "relation": "因果关系/对立关系/支撑关系/类比关系", "description": "关系描述"}}
  ]
}}
```"""

        try:
            response = llm_client.chat.completions.create(
                model=llm_client.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            content = response.choices[0].message.content

            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get("relations", [])
            else:
                data = json.loads(content)
                return data.get("relations", [])
        except Exception as e:
            print(f"[KG关系提取错误] {e}")
            return []

    def build_from_articles(self, articles_texts: List[Dict], llm_client=None):
        """从多篇文章构建知识图谱"""
        print(f"[KG] 开始构建知识图谱，共 {len(articles_texts)} 篇文章...")

        all_entities = []
        all_relations = []

        for i, article in enumerate(articles_texts):
            text = article.get("text", "")
            if not text:
                continue

            entities = self.extract_entities_llm(text, llm_client)
            for e in entities:
                e["article"] = article.get("filename", "unknown")
            all_entities.extend(entities)

            relations = self.extract_relations_llm(text, entities, llm_client)
            for r in relations:
                r["article"] = article.get("filename", "unknown")
            all_relations.extend(relations)

            if i % 10 == 0:
                print(f"[KG] 已处理 {i+1}/{len(articles_texts)} 篇")

        # 构建NetworkX图
        for entity in all_entities:
            name = entity["name"]
            if not self.graph.has_node(name):
                self.graph.add_node(
                    name,
                    type=entity.get("type", "unknown"),
                    articles=[entity.get("article", "")]
                )
            else:
                existing = self.graph.nodes[name].get("articles", [])
                if entity.get("article") not in existing:
                    existing.append(entity.get("article"))
                    self.graph.nodes[name]["articles"] = existing
            self.entity_index[name].append(entity)

        for rel in all_relations:
            src, tgt = rel.get("source", ""), rel.get("target", "")
            if src and tgt:
                if not self.graph.has_edge(src, tgt):
                    self.graph.add_edge(
                        src, tgt,
                        relation=rel.get("relation", ""),
                        descriptions=[rel.get("description", "")]
                    )
                else:
                    existing_desc = self.graph.edges[src, tgt].get("descriptions", [])
                    existing_desc.append(rel.get("description", ""))
                    self.graph.edges[src, tgt]["descriptions"] = existing_desc
                self.relation_index[(src, tgt)].append(rel)

        print(f"[KG完成] {self.graph.number_of_nodes()} 个节点, "
              f"{self.graph.number_of_edges()} 条边")

        self._save()

    def _save(self):
        """保存知识图谱到JSON"""
        data = {
            "nodes": [
                {"name": n, **self.graph.nodes[n]}
                for n in self.graph.nodes
            ],
            "edges": [
                {"source": u, "target": v, **self.graph.edges[u, v]}
                for u, v in self.graph.edges
            ]
        }
        with open(KG_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """从JSON加载知识图谱"""
        if not os.path.exists(KG_JSON_PATH):
            return False
        with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.DiGraph()
        for node in data.get("nodes", []):
            name = node.pop("name")
            self.graph.add_node(name, **node)
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            self.graph.add_edge(src, tgt, **edge)
        print(f"[KG加载] {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
        return True

    def search_entities(self, query: str, top_k: int = 10) -> List[Dict]:
        """搜索相关实体（模糊匹配）"""
        results = []
        for node in self.graph.nodes:
            if query.lower() in node.lower():
                node_data = self.graph.nodes[node]
                neighbors = list(self.graph.neighbors(node))
                results.append({
                    "entity": node,
                    "type": node_data.get("type", "unknown"),
                    "articles": node_data.get("articles", []),
                    "related_entities": neighbors[:10],
                    "article_count": len(node_data.get("articles", [])),
                })
        results.sort(key=lambda r: r["article_count"], reverse=True)
        return results[:top_k]

    def get_subgraph(self, entity_names: List[str], depth: int = 1) -> nx.DiGraph:
        """获取指定实体周围的子图"""
        nodes = set()
        for name in entity_names:
            if self.graph.has_node(name):
                nodes.add(name)
                # BFS扩展depth层
                frontier = {name}
                for _ in range(depth):
                    new_frontier = set()
                    for n in frontier:
                        for neighbor in list(self.graph.predecessors(n)) + list(self.graph.successors(n)):
                            if neighbor not in nodes:
                                nodes.add(neighbor)
                                new_frontier.add(neighbor)
                    frontier = new_frontier
        return self.graph.subgraph(nodes)

    def find_paths(self, entity1: str, entity2: str, max_len: int = 4) -> List[List[str]]:
        """查找两个实体之间的关联路径"""
        if not self.graph.has_node(entity1) or not self.graph.has_node(entity2):
            return []
        try:
            paths = list(nx.all_simple_paths(
                self.graph.to_undirected(), entity1, entity2, cutoff=max_len
            ))
            return sorted(paths, key=len)[:10]
        except nx.NetworkXNoPath:
            return []


if __name__ == "__main__":
    kg = KnowledgeGraph()

    # 测试加载
    if kg.load():
        results = kg.search_entities("房地产")
        for r in results:
            print(f"\n实体: {r['entity']} ({r['type']})")
            print(f"  出现: {r['article_count']} 篇文章")
            print(f"  关联: {r['related_entities']}")
