"""pdfplumber重新提取37篇图片PDF"""
import os, sys, json, re, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdfplumber
from pathlib import Path
from config import DATA_DIR

PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
OUT_DIR = os.path.join(DATA_DIR, "extracted_texts")

# 找0字符的
zero = []
for f in os.listdir(OUT_DIR):
    if f.startswith("_") or not f.endswith(".json"): continue
    with open(os.path.join(OUT_DIR, f)) as fp:
        d = json.load(fp)
    if d.get("char_count", 0) == 0:
        zero.append(d["filename"])

print(f"pdfplumber重试: {len(zero)} 篇")
ok = 0
for fname in zero:
    path = os.path.join(PDF_DIR, fname)
    if not os.path.exists(path):
        print(f"  ❌ 文件不存在: {fname}")
        continue
    try:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:50]:  # 限50页
                t = page.extract_text()
                if t: text_parts.append(t)
        raw = "\n".join(text_parts).strip()
        if len(raw) > 100:
            out = {
                "filename": fname, "text": raw, "char_count": len(raw),
                "page_count": len(text_parts),
                "hash": hashlib.md5(raw.encode()).hexdigest(), "ocr": "pdfplumber"
            }
            oname = Path(fname).stem + ".json"
            with open(os.path.join(OUT_DIR, oname), "w") as fp:
                json.dump(out, fp, ensure_ascii=False, indent=2)
            ok += 1
            print(f"  ✅ {fname[:50]}... ({len(raw):,}字符)")
        else:
            print(f"  ❌ {fname[:50]}... ({len(raw)}字符)")
    except Exception as e:
        print(f"  ⚠️ {fname[:40]}... {str(e)[:50]}")

print(f"\npdfplumber成功: {ok}/{len(zero)}")

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
        "total_pdfs": len(all_articles), "success_count": len(all_articles),
        "total_chars": sum(a["char_count"] for a in all_articles),
        "articles": all_articles
    }, f, ensure_ascii=False, indent=2)
print(f"索引: {len(all_articles)}篇, {sum(a['char_count'] for a in all_articles):,}字符")
