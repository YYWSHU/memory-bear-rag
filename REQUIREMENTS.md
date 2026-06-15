# REQUIREMENTS.md - 记忆承载 RAG 知识库项目

## 项目概述
将"记忆承载"（碧树西风）公众号 90+ 篇 PDF 文章构建为三层智能知识检索系统，能够模拟作者的思维逻辑回答问题。

---

## 迭代记录

### 第1轮迭代 (2026-06-15) - ✅ 已完成

**用户需求：**
1. 提取 `D:\yyw\download\记忆承载` 下所有 PDF 的文字
2. 构建 RAG + 知识图谱 + 思维结构 三层架构
3. 能够模拟"记忆承载"作者的思维逻辑和推理链条
4. 使用 DeepSeek V4 Pro API 做最终生成
5. Web 聊天界面（Streamlit），展示检索来源和思维推理链
6. 关联 GitHub 仓库: https://github.com/YYWSHU/memory-bear-rag

**完成情况：**
- [x] 解压 7z 文件并合并去重 PDF（91个PDF）
- [x] 提取所有 PDF 文字（49篇有文字，275万字符；42篇为图片型需OCR）
- [x] 构建 RAG 层（5120个文本块 + Ollama bge-m3 嵌入 + ChromaDB）
- [x] 构建知识图谱层（56个核心概念节点，1540条关联边）
- [x] 构建思维结构层（19篇文章深度提取，10个分析框架，50个高频词）
- [x] 实现检索+生成 pipeline（向量检索 + 图谱 + 思维模式 → DeepSeek V4 Pro）
- [x] 构建 Streamlit Web 界面（聊天 + 来源展示 + 思维特征展示）
- [x] GitHub 仓库关联

**技术架构：**
```
Layer 3: 思维结构层 → DeepSeek V4 Pro 深度提取论证链/分析框架/价值锚点
Layer 2: 知识图谱层 → NetworkX 共现图谱（56节点/1540边）
Layer 1: RAG 语义检索层 → Ollama bge-m3 嵌入 + ChromaDB（5120条）
     ↑
  49篇 PDF 原文（275万字符）
```

**关键技术选型：**
- PDF 提取: PyMuPDF (fitz)，多进程并行，9秒完成91个PDF
- 嵌入模型: Ollama bge-m3 (1024维)，本地CPU运行
- 向量数据库: ChromaDB
- 知识图谱: NetworkX + 规则抽取
- 思维提取: DeepSeek V4 Pro API
- 生成模型: DeepSeek V4 Pro API
- Web 界面: Streamlit

**已知限制：**
- 42个图片型PDF（2024-2025年的大部分文章）需要OCR才能提取文字
- 思维结构仅提取了19篇（前20篇中有1篇失败），其余49篇待提取
- 知识图谱使用规则抽取，后续可升级为LLM抽取

**Web 访问:** http://localhost:8501
