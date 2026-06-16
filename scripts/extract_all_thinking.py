"""批量提取所有文章思维结构"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.thinking_structure import ThinkingStructureExtractor

extractor = ThinkingStructureExtractor()
patterns = extractor.extract_all_articles(force=False)  # 跳过已有的19篇
meta = extractor.build_meta_patterns(patterns)
print(f"\n总计: {len(patterns)} 篇思维结构")
print(f"框架: {meta['top_frameworks'][:5]}")
