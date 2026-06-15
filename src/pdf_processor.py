import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""PDF 文字提取模块 - 解压7z, 合并去重, 提取文字"""
import os
import re
import json
import hashlib
import subprocess
import fitz  # PyMuPDF
from tqdm import tqdm
from pathlib import Path
from config import PDF_SOURCE_DIR, EXTRACTED_TEXT_DIR, DATA_DIR


def extract_7z_if_needed():
    """解压7z文件到临时目录，返回解压后的PDF路径列表"""
    sz_file = os.path.join(PDF_SOURCE_DIR, "记忆承载（2025年不全）.7z")
    extract_dir = os.path.join(DATA_DIR, "raw_pdfs_from_7z")

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"[跳过] 7z已解压到 {extract_dir}")
        return list(Path(extract_dir).rglob("*.pdf"))

    if not os.path.exists(sz_file):
        print("[警告] 7z文件不存在，跳过")
        return []

    print(f"[解压] 正在解压 7z 文件...")
    os.makedirs(extract_dir, exist_ok=True)
    result = subprocess.run(
        ["7z", "x", sz_file, f"-o{extract_dir}", "-y"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[错误] 解压失败: {result.stderr}")
        return []

    pdfs = list(Path(extract_dir).rglob("*.pdf"))
    print(f"[完成] 从7z解压出 {len(pdfs)} 个PDF")
    return pdfs


def collect_all_pdfs():
    """收集所有PDF文件，去重（基于文件hash）"""
    seen_hashes = {}
    all_pdfs = []

    # 1. 收集原始目录下的PDF
    for root, dirs, files in os.walk(PDF_SOURCE_DIR):
        # 跳过7z解压目录（如果存在）
        if "raw_pdfs_from_7z" in root:
            continue
        for f in files:
            if f.endswith(".pdf"):
                all_pdfs.append(Path(root) / f)

    # 2. 收集7z解压的PDF
    all_pdfs.extend(extract_7z_if_needed())

    # 3. 基于文件名+大小去重
    unique_pdfs = []
    for pdf_path in all_pdfs:
        if not pdf_path.exists():
            continue
        file_stat = pdf_path.stat()
        key = f"{pdf_path.name}_{file_stat.st_size}"
        if key not in seen_hashes:
            seen_hashes[key] = True
            unique_pdfs.append(pdf_path)

    print(f"[收集] 总共 {len(all_pdfs)} 个PDF，去重后 {len(unique_pdfs)} 个")
    return sorted(unique_pdfs, key=lambda p: p.name)


def clean_text(text: str) -> str:
    """清洗提取出的文本"""
    # 移除多余空白行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 移除多余空格
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # 修复常见乱码
    text = text.replace('　', ' ')  # 全角空格
    # 去掉首尾空白
    text = text.strip()
    return text


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """从单个PDF提取文字"""
    try:
        doc = fitz.open(str(pdf_path))
        full_text = []
        page_count = len(doc)
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                full_text.append(text)
        doc.close()

        raw_text = "\n".join(full_text)
        cleaned_text = clean_text(raw_text)

        return {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "text": cleaned_text,
            "char_count": len(cleaned_text),
            "page_count": page_count,
            "hash": hashlib.md5(cleaned_text.encode()).hexdigest()
        }
    except Exception as e:
        print(f"[错误] 处理 {pdf_path.name} 失败: {e}")
        return {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "text": "",
            "char_count": 0,
            "page_count": 0,
            "error": str(e)
        }


def extract_all_pdfs(pdf_list=None, force=False):
    """批量提取所有PDF文字，返回提取结果列表"""
    if pdf_list is None:
        pdf_list = collect_all_pdfs()

    results = []
    errors = []

    for pdf_path in tqdm(pdf_list, desc="提取PDF文字"):
        # 检查是否已提取
        output_name = pdf_path.stem + ".json"
        output_path = os.path.join(EXTRACTED_TEXT_DIR, output_name)

        if os.path.exists(output_path) and not force:
            with open(output_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        result = extract_text_from_pdf(pdf_path)
        if result.get("error"):
            errors.append(result)
        elif result["char_count"] > 0:
            # 保存提取结果
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            results.append(result)

    # 保存全局索引
    index = {
        "total_pdfs": len(pdf_list),
        "success_count": len(results),
        "error_count": len(errors),
        "total_chars": sum(r["char_count"] for r in results),
        "articles": [
            {"filename": r["filename"], "char_count": r["char_count"], "hash": r["hash"]}
            for r in results
        ]
    }
    with open(os.path.join(EXTRACTED_TEXT_DIR, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 成功提取 {len(results)} 篇，失败 {len(errors)} 篇，"
          f"共 {index['total_chars']} 个字符")

    if errors:
        print(f"[失败列表]")
        for e in errors:
            print(f"  - {e['filename']}: {e.get('error', '未知错误')}")

    return results


if __name__ == "__main__":
    pdfs = collect_all_pdfs()
    print(f"准备处理 {len(pdfs)} 个PDF文件")
    results = extract_all_pdfs(pdfs, force=True)
    print(f"处理完成，共提取 {len(results)} 篇文章")
