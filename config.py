"""配置管理模块"""
import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_SOURCE_DIR = "/mnt/d/yyw/download/记忆承载"
EXTRACTED_TEXT_DIR = os.path.join(DATA_DIR, "extracted_texts")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
KG_DIR = os.path.join(DATA_DIR, "knowledge_graph")
THINKING_DIR = os.path.join(DATA_DIR, "thinking_patterns")

# RAG配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
VECTOR_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

# 知识图谱配置
KG_JSON_PATH = os.path.join(KG_DIR, "knowledge_graph.json")

# 思维结构配置
THINKING_PATTERNS_PATH = os.path.join(THINKING_DIR, "thinking_patterns.json")

# 检索配置
TOP_K_CHUNKS = 10
TOP_K_GRAPH = 5

# 确保目录存在
for d in [DATA_DIR, EXTRACTED_TEXT_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, KG_DIR, THINKING_DIR]:
    os.makedirs(d, exist_ok=True)
