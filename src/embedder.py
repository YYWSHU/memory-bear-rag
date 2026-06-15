import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""向量嵌入模块 - BGE中文模型 + ChromaDB存储"""
import os
import json
import time
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import (
    EMBEDDING_MODEL, VECTOR_DB_PATH, CHUNKS_DIR, TOP_K_CHUNKS
)


class EmbeddingStore:
    """向量嵌入存储管理"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, persist_dir: str = VECTOR_DB_PATH):
        print(f"[加载] 嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.persist_dir = persist_dir

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = "memory_bear_articles"

    def _get_or_create_collection(self):
        """获取或创建ChromaDB集合"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> List[List[float]]:
        """批量向量化文本块"""
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def build_index(self, chunks: List[Dict], force: bool = False):
        """构建向量索引"""
        collection = self._get_or_create_collection()

        # 如果集合非空且不强制重建，跳过
        if collection.count() > 0 and not force:
            print(f"[跳过] 向量索引已存在 ({collection.count()} 条)，使用 force=True 重建")
            return

        # 如果强制重建，先清空
        if force and collection.count() > 0:
            self.client.delete_collection(self.collection_name)
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("[重建] 已清空旧索引")

        print(f"[向量化] 正在处理 {len(chunks)} 个文本块...")
        embeddings = self.embed_chunks(chunks)

        print(f"[存储] 正在写入 ChromaDB...")
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]

            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]
            metadatas = [
                {
                    "article_title": c.get("article_title", ""),
                    "article_filename": c.get("article_filename", ""),
                    "article_date": c.get("article_date", ""),
                    "chunk_index": c["chunk_index"],
                    "chunk_hash": c.get("chunk_hash", ""),
                    "char_count": c["char_count"],
                }
                for c in batch_chunks
            ]

            collection.add(
                ids=ids,
                embeddings=batch_embs,
                documents=documents,
                metadatas=metadatas,
            )

        print(f"[完成] 已存入 {collection.count()} 条向量记录")

    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict]:
        """语义检索"""
        collection = self._get_or_create_collection()
        if collection.count() == 0:
            print("[错误] 向量索引为空，请先运行 build_index()")
            return []

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                formatted.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "score": 1.0 - results["distances"][0][i],  # cosine distance -> similarity
                    "article_title": meta.get("article_title", ""),
                    "article_date": meta.get("article_date", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                })

        return formatted

    def get_chunks_by_article(self, article_filename: str) -> List[Dict]:
        """获取指定文章的所有块"""
        collection = self._get_or_create_collection()
        results = collection.get(
            where={"article_filename": article_filename},
            include=["documents", "metadatas"]
        )
        return [
            {"chunk_id": results["ids"][i], "text": results["documents"][i],
             "metadata": results["metadatas"][i]}
            for i in range(len(results["ids"]))
        ]


def load_chunks() -> List[Dict]:
    """加载分块数据"""
    chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    if not os.path.exists(chunks_path):
        print("[错误] 未找到分块文件，先运行 text_chunker.py")
        return []
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    chunks = load_chunks()
    if chunks:
        store = EmbeddingStore()
        store.build_index(chunks, force=True)

        # 测试检索
        results = store.search("跨域阶层的方法")
        for r in results:
            print(f"\n[{r['score']:.4f}] {r['article_title']} - {r['article_date']}")
            print(r['text'][:200])
