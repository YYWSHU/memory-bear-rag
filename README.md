# 记忆承载 RAG 知识库

基于"记忆承载"（碧树西风）公众号文章构建的三层智能知识检索系统。

## 架构

```
Layer 3: 思维结构层 → 论证链 + 分析框架 + 价值锚点
Layer 2: 知识图谱层 → 实体关系 + 因果链
Layer 1: RAG 语义检索层 → 向量嵌入 + 语义搜索
```

## 技术栈

- **PDF 提取**: PyMuPDF
- **向量模型**: BGE-large-zh-v1.5
- **向量数据库**: ChromaDB
- **知识图谱**: NetworkX + JSON
- **生成模型**: DeepSeek API
- **Web 界面**: Streamlit

## 使用

```bash
pip install -r requirements.txt
streamlit run src/app.py
```
