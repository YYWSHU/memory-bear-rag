import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""文本分块模块 - 语义分块 + 元数据标注"""
import os
import json
import hashlib
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_DIR, EXTRACTED_TEXT_DIR


def simple_semantic_chunk(text: str, chunk_size: int = CHUNK_SIZE,
                          overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    简单语义分块：以段落为边界，尽量保持语义完整性。
    当段落超过chunk_size时才在句子边界切割。
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            continue

        # 如果当前块 + 新段落 超限
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # 保留overlap：用当前块的后半部分作为新块的开始
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # 对于超长段落，进一步在句子边界切割
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size * 1.5:
            final_chunks.append(chunk)
        else:
            # 在句号、问号、感叹号处切割
            sentences = []
            for sep in ['。', '？', '！', '；']:
                if sep in chunk:
                    parts = chunk.split(sep)
                    sentences = [p.strip() + sep for p in parts[:-1]]
                    sentences.append(parts[-1].strip())
                    break
            if not sentences:
                sentences = [chunk]

            sub_chunk = ""
            for sent in sentences:
                if len(sub_chunk) + len(sent) > chunk_size and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = sent
                else:
                    sub_chunk += sent
            if sub_chunk.strip():
                final_chunks.append(sub_chunk.strip())

    return [c for c in final_chunks if len(c) > 50]


def chunk_article(article: dict) -> List[Dict]:
    """
    将一篇文章分块，每块附带元数据。
    尝试从文件名提取日期信息。
    """
    filename = article.get("filename", "")
    text = article.get("text", "")

    # 尝试提取日期（格式如：21-12-22 或 22-1-24）
    import re
    date_match = re.match(r'(\d{2})-(\d{1,2})-(\d{1,2})', filename)
    year = f"20{date_match.group(1)}" if date_match else "unknown"
    month = date_match.group(2) if date_match else "unknown"
    day = date_match.group(3) if date_match else "unknown"

    # 提取标题（去掉日期前缀和.pdf后缀）
    title = filename
    if date_match:
        title = filename[date_match.end():].lstrip('-_ ')
    title = title.replace('.pdf', '').strip()

    chunks = simple_semantic_chunk(text)

    results = []
    for i, chunk_text in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
        results.append({
            "chunk_id": f"{article.get('hash', 'unknown')}_{i}",
            "article_title": title,
            "article_filename": filename,
            "article_date": f"{year}-{month}-{day}",
            "chunk_index": i,
            "chunk_hash": chunk_hash,
            "char_count": len(chunk_text),
            "text": chunk_text,
        })

    return results


def chunk_all_articles(force: bool = False) -> List[Dict]:
    """对所有已提取的文章进行分块"""
    all_chunks = []

    index_path = os.path.join(EXTRACTED_TEXT_DIR, "_index.json")
    if not os.path.exists(index_path):
        print("[错误] 未找到提取的文字索引，先运行 pdf_processor.py")
        return []

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    for article_info in index.get("articles", []):
        article_path = os.path.join(EXTRACTED_TEXT_DIR, article_info["filename"].replace(".pdf", ".json"))

        if not os.path.exists(article_path):
            print(f"[跳过] 未找到 {article_info['filename']}")
            continue

        with open(article_path, "r", encoding="utf-8") as f:
            article = json.load(f)

        chunks = chunk_article(article)
        all_chunks.extend(chunks)

    # 保存分块结果
    output_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"[完成] 共产生 {len(all_chunks)} 个文本块，"
          f"平均每块 {sum(c['char_count'] for c in all_chunks) // max(len(all_chunks), 1)} 字符")

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_articles(force=True)
