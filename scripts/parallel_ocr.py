"""多进程并行OCR - 每个PDF一个进程"""
import os, sys, json, re, hashlib, subprocess
import fitz
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

PDF_DIR = "/home/yyw/memory-bear-rag/data/raw_pdfs"
OUT_DIR = "/home/yyw/memory-bear-rag/data/extracted_texts"
os.makedirs(OUT_DIR, exist_ok=True)
WORKERS = min(cpu_count(), 8)  # 8进程并行

def ocr_one_pdf(filename):
    """OCR单个PDF（在子进程中运行）"""
    pdf_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(pdf_path):
        return filename, {"error": "not found", "char_count": 0}

    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        full_text = []

        for page in doc:
            text = page.get_text("text").strip()
            if len(text) > 50:
                full_text.append(text)
                continue

            pix = page.get_pixmap(dpi=150)  # 降低DPI提速度
            img_path = f"/tmp/ocr_{os.getpid()}_{hashlib.md5(str(page.number).encode()).hexdigest()[:6]}.png"
            pix.save(img_path)

            result = subprocess.run(
                ["tesseract", img_path, "stdout", "-l", "chi_sim+eng", "--psm", "6"],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                full_text.append(result.stdout.strip())
            try:
                os.remove(img_path)
            except:
                pass

        doc.close()
        raw = "\n".join(full_text).strip()
        raw = re.sub(r'\n{3,}', '\n\n', raw)

        out = {
            "filename": filename,
            "text": raw,
            "char_count": len(raw),
            "page_count": page_count,
            "hash": hashlib.md5(raw.encode()).hexdigest(),
            "ocr": True
        }

        output_name = Path(filename).stem + ".json"
        with open(os.path.join(OUT_DIR, output_name), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        return filename, out

    except Exception as e:
        return filename, {"error": str(e), "char_count": 0}


if __name__ == "__main__":
    # 找0字符的PDF
    zero_files = []
    for f in os.listdir(OUT_DIR):
        if f.startswith("_") or not f.endswith(".json"): continue
        with open(os.path.join(OUT_DIR, f)) as fp:
            d = json.load(fp)
        if d.get("char_count", 0) == 0 and "error" not in d:
            zero_files.append(d["filename"])

    print(f"需OCR: {len(zero_files)} 个PDF, {WORKERS} 进程并行")

    # 并行OCR
    with Pool(WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(ocr_one_pdf, zero_files),
            total=len(zero_files),
            desc="OCR并行"
        ))

    ok = [(n, r) for n, r in results if r.get("char_count", 0) > 0]
    err = [(n, r) for n, r in results if "error" in r]
    total_chars = sum(r["char_count"] for _, r in ok)
    print(f"\n✅ 完成: {len(ok)}篇, 失败: {len(err)}篇, {total_chars:,}字符")

    # 更新索引
    all_articles = []
    for f in os.listdir(OUT_DIR):
        if f.startswith("_") or not f.endswith(".json"): continue
        with open(os.path.join(OUT_DIR, f)) as fp:
            d = json.load(fp)
        if d.get("char_count", 0) > 0:
            all_articles.append({"filename": d["filename"], "char_count": d["char_count"], "hash": d["hash"]})

    with open(os.path.join(OUT_DIR, "_index.json"), "w") as f:
        json.dump({
            "total_pdfs": len(all_articles),
            "success_count": len(all_articles),
            "total_chars": sum(a["char_count"] for a in all_articles),
            "articles": all_articles
        }, f, ensure_ascii=False, indent=2)

    print(f"索引更新: {len(all_articles)}篇, {sum(a['char_count'] for a in all_articles):,}字符")
