import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""思维结构提取模块 - 论证链 + 分析框架 + 价值锚点"""
import os
import json
import re
from typing import List, Dict, Optional
from openai import OpenAI
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    THINKING_DIR, EXTRACTED_TEXT_DIR
)


class ThinkingStructureExtractor:
    """从文章中提取作者的思维结构"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )
        self.model = model or DEEPSEEK_MODEL

    def extract_thinking_pattern(self, article: dict) -> dict:
        """
        从单篇文章中深度提取：
        1. 核心论点（thesis）
        2. 论证链条（argument chain: 前提→推理→结论）
        3. 分析框架（analytical framework）
        4. 价值判断（value judgments）
        5. 思维跃迁（cross-topic bridges）
        6. 独特见解（unique insights）
        """
        title = article.get("filename", "").replace(".pdf", "").strip()
        text = article.get("text", "")

        if len(text) < 100:
            return {"error": "文本太短", "filename": title}

        # 截断过长文本（保留开头+结尾+中间采样）
        if len(text) > 8000:
            mid = len(text) // 2
            text = text[:3000] + "\n...[中段省略]...\n" + text[mid - 1000:mid + 1000] + "\n...[中段省略]...\n" + text[-3000:]

        prompt = f"""你是一位认知分析专家。请深度分析以下"记忆承载（碧树西风）"公众号文章，提取作者的思维结构。

文章标题：{title}

文章内容：
{text}

请严格按以下维度分析（返回JSON）：

1. **核心论点**（thesis）：作者想说服读者的核心观点（1-3个）
2. **论证链条**（argument_chain）：前提假设→推理步骤→最终结论，画出一条完整的逻辑线
3. **分析框架**（framework）：作者分析问题时采用的范式/框架（如：利益分析法、历史周期律、人性底层逻辑、博弈论视角等）
4. **隐含前提**（hidden_premises）：作者没说但依赖的不言自明的假设
5. **价值判断**（value_judgments）：作者认为什么是好的/坏的、对的/错的、重要的/不重要的
6. **思维跃迁**（bridges）：作者如何从一个具体话题跳到更深层/更普遍的规律（从X谈到Y）
7. **独特修辞策略**（rhetoric）：作者独特的说服技巧（如：先破后立、类比映射、极端场景推演、利益计算等）
8. **核心词汇**（keywords）：作者反复使用的标志性词汇和短语（20-30个）

返回严格JSON格式（不要markdown代码块之外的内容）：
```json
{{
  "title": "文章标题",
  "date_hint": "从文件名推测的日期",
  "thesis": ["核心论点1", "核心论点2"],
  "argument_chain": {{
    "premises": ["前提1", "前提2"],
    "reasoning_steps": ["推理步骤1", "推理步骤2"],
    "conclusions": ["结论1", "结论2"]
  }},
  "framework": {{
    "name": "主要分析框架名称",
    "description": "框架的简要说明",
    "pattern": "该框架的典型运用模式"
  }},
  "hidden_premises": ["隐含假设1", "隐含假设2"],
  "value_judgments": [
    {{"topic": "话题领域", "stance": "作者立场", "reasoning": "判断依据"}}
  ],
  "bridges": [
    {{"from": "具体话题", "to": "普遍规律", "mechanism": "跃迁机制"}}
  ],
  "rhetoric": ["修辞策略1", "修辞策略2"],
  "keywords": ["关键词1", "关键词2"]
}}
```"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的认知分析专家，擅长分析文本中的思维结构和论证模式。请严格按JSON格式输出分析结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            content = response.choices[0].message.content

            # 提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # 尝试直接解析
                result = json.loads(content)

            result["filename"] = title
            result["extraction_model"] = self.model
            return result

        except Exception as e:
            return {"error": str(e), "filename": title}

    def extract_all_articles(self, article_list: List[dict] = None,
                             force: bool = False, max_articles: int = None) -> List[dict]:
        """批量提取所有文章的思维结构"""
        if article_list is None:
            # 从提取的文字中加载
            article_list = self._load_all_articles()

        results = []
        filepath = os.path.join(THINKING_DIR, "thinking_patterns.json")

        # 加载已有结果
        existing = {}
        if os.path.exists(filepath) and not force:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    existing[item.get("filename", "")] = item
            print(f"[加载] 已有 {len(existing)} 条思维结构记录")

        to_process = article_list[:max_articles] if max_articles else article_list

        for i, article in enumerate(to_process):
            fname = article.get("filename", "")
            fname_key = fname.replace(".pdf", "").strip()
            if fname_key in existing and not force:
                results.append(existing[fname_key])
                continue

            print(f"[思维提取 {i+1}/{len(to_process)}] {fname[:60]}...")
            pattern = self.extract_thinking_pattern(article)

            if "error" in pattern:
                print(f"  [错误] {pattern['error']}")
                continue

            results.append(pattern)

            # 每5篇保存一次（防止中断丢失）
            if (i + 1) % 5 == 0:
                self._save_results(results, filepath)
                print(f"  [保存] 已保存 {len(results)} 条记录")

        # 最终保存
        self._save_results(results, filepath)
        print(f"\n[完成] 共提取 {len(results)} 篇文章的思维结构")
        return results

    def _load_all_articles(self) -> List[dict]:
        """加载所有提取过的文章"""
        index_path = os.path.join(EXTRACTED_TEXT_DIR, "_index.json")
        if not os.path.exists(index_path):
            return []

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        articles = []
        for info in index.get("articles", []):
            article_path = os.path.join(EXTRACTED_TEXT_DIR, info["filename"].replace(".pdf", ".json"))
            if os.path.exists(article_path):
                with open(article_path, "r", encoding="utf-8") as f:
                    articles.append(json.load(f))
        return articles

    def _save_results(self, results: List[dict], filepath: str):
        """保存思维结构提取结果"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def build_meta_patterns(self, patterns: List[dict]) -> dict:
        """
        从所有文章的思维结构中提炼元模式：
        - 高频分析框架
        - 常见论证模式
        - 核心价值体系
        - 标志性修辞策略
        """
        frameworks = {}
        all_keywords = {}
        all_thesis = []
        all_value_judgments = []
        all_rhetoric = {}

        for p in patterns:
            if "error" in p:
                continue

            # 分析框架统计
            fw = p.get("framework", {})
            fw_name = fw.get("name", "unknown")
            frameworks[fw_name] = frameworks.get(fw_name, 0) + 1

            # 关键词频率
            for kw in p.get("keywords", []):
                all_keywords[kw] = all_keywords.get(kw, 0) + 1

            # 论点收集
            all_thesis.extend(p.get("thesis", []))

            # 价值判断收集
            all_value_judgments.extend(p.get("value_judgments", []))

            # 修辞策略
            for r in p.get("rhetoric", []):
                all_rhetoric[r] = all_rhetoric.get(r, 0) + 1

        meta = {
            "total_articles_analyzed": len(patterns),
            "top_frameworks": sorted(frameworks.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_keywords": sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:50],
            "sample_thesis": all_thesis[:20],
            "value_judgments": all_value_judgments[:30],
            "top_rhetoric": sorted(all_rhetoric.items(), key=lambda x: x[1], reverse=True)[:10],
        }

        # 保存元模式
        meta_path = os.path.join(THINKING_DIR, "meta_patterns.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[元模式] 已生成：{len(meta['top_frameworks'])} 个框架, "
              f"{len(meta['top_keywords'])} 个高频词, "
              f"{len(meta['top_rhetoric'])} 种修辞策略")

        return meta


if __name__ == "__main__":
    extractor = ThinkingStructureExtractor()
    patterns = extractor.extract_all_articles(max_articles=5)  # 测试5篇
    meta = extractor.build_meta_patterns(patterns)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
