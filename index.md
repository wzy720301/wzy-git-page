---
title: Welcome to 菩提大仙的盘丝洞 - my page
---

# AI-RAG搭建方法及流程



## 一、参考文章

| 序号  | 资料名称及链接                                                                                                                        | 内容概要          | 参考指数                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | ------------- | ------------------------------------------------------ |
| 1   | [N8N+数据爬取工作流，收集数据集本地搭建RAG知识库！](https://www.toutiao.com/video/7493093048172216884/?log_from=11dd408b22442_1745076546106 "网络视频") | 非常好的搭建流程      | [★★★★★](video/N8N+数据爬取工作流，收集数据集本地搭建RAG知识库！.mp4 "本地视频") |
| 2   | [示例资料2](https://example2.com)                                                                                                  | 这是示例资料2的概要内容。 | ★★★★☆                                                  |
| 3   | [示例资料3](https://example3.com)                                                                                                  | 这是示例资料3的概要内容。 | ★★☆☆☆                                                  |
| 4   | [示例资料4](https://example4.com)                                                                                                  | 这是示例资料4的概要内容。 | ★★★★★                                                  |
| 5   | [示例资料5](https://example5.com)                                                                                                  | 这是示例资料5的概要内容。 | ★★★☆☆                                                  |

---

## 二、 重点文章学习

### 2.1.RAG 是什么？为何如此重要？

[参考资料](https://www.toutiao.com/article/7478598179768861199/?log_from=6f5a4e8527956_1744272712176#:~:text=%E5%9C%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0,%E6%8A%80%E6%9C%AF%E7%9A%84%E8%BA%AB%E5%BD%B1%E3%80%82 "RAG 是什么？为何如此重要？")

> AI RAG，简单来说就是一种让人工智能更聪明、更准确回答问题的技术。
> 
> **“RAG”是“Retrieval-Augmented Generation”的缩写，翻译过来就是“检索增强生成”。**
> 
> 通常我们问人工智能一个问题，它是根据之前学习到的大量知识来回答。但有时候这些知识可能不够全面> 或者不够新，回答就可能不太准确或完善。
> 
> 而AI RAG呢，就像是给人工智能找了一个“小助手”。当你提出问题后，它不仅会依靠自己原有的知识储> 备，还会立刻去相关的数据库、文档或者网页等地方快速检索，把最新、最准确、最相关的信息找出来，> 然后结合这些信息来生成答案。这样一来，人工智能给出的回答就会更加准确、详细，也更能符合你的需> 求啦。
> 
> 比如说，你问人工智能关于最新的科技发明，它通过RAG技术就可以马上从最新的科技资讯网站上找到相> 关内容，然后告诉你准确的信息，而不是只凭它之前学习到的一些旧知识来回答。

![参考图](https://p3-sign.toutiaoimg.com/tos-cn-i-6w9my0ksvp/c4f5045faf21438cad5bd4a7f70dddb4~tplv-tt-origin-web:gif.jpeg?_iz=58558&from=article.pc_detail&lk3s=953192f4&x-expires=1744877512&x-signature=r1rPBJVvIJNjM3wdGsEXTpAO9ZA%3D "图片")

### 2.2. LLM 存在的问题与 RAG 的解决方案

从您提供的文档内容来看，LLM（大语言模型）存在的主要问题包括：

#### 2.2.1 LLM存在的问题

2.2.1.1. **幻觉问题（Hallucination）**

- 这是文档中明确提到的问题
- 表现为：模型可能生成错误信息或与事实不符的内容
- 原因：
  * 训练数据中缺乏足够的上下文信息
  * 处理长文本时记忆能力有限

2.2.1.2. **知识局限性**（文档开头部分提到）

- 模型依赖训练时的静态知识库
- 无法自动获取最新信息（如最新的科技发明）
- 知识覆盖可能不全面

2.2.1.3. **准确性不足**（隐含在RAG解决方案的描述中）

- 仅依靠预训练知识时回答可能不够准确
- 缺乏实时验证机制

#### 2.2.2 RAG解决方案

RAG技术通过以下方式解决LLM存在的问题：

2.2.2.1 **实时检索机制**

- 动态获取最新数据（解决知识陈旧问题）
- 支持多数据源接入：数据库/文档/API/网页等
- 示例：查询最新科技发明时直接从科技资讯站获取

2.2.2.2 **增强生成质量**

- 结合检索结果与预训练知识生成回答
- 通过相关性排序筛选最优参考内容
- 减少幻觉现象（提供事实依据）

2.2.2.3 **模块化架构**

- 检索器(Retriever)：高效向量搜索技术
- 生成器(Generator)：保持原有语言生成能力
- 协调模块：智能融合检索与生成结果

2.2.2.4 **持续更新能力**

- 无需重新训练即可更新知识库
- 支持增量式数据添加
- 可配置的刷新频率策略

需要我补充其他LLM常见问题（如计算成本高、推理速度慢等）吗？这些在您的文档中尚未提及。

### 2.3. RAG标准搭建流程

#### 2.3.1 核心组件

| 组件    | 功能说明   | 常用技术方案                         |
| ----- | ------ | ------------------------------ |
| 文档加载器 | 原始数据导入 | PyPDF/Docx/HTML解析器             |
| 文本分割器 | 文档分块处理 | RecursiveCharacterTextSplitter |
| 向量数据库 | 存储嵌入向量 | Pinecone/Weaviate/FAISS        |
| 嵌入模型  | 文本向量化  | OpenAI/text-embedding-ada-002  |
| 检索器   | 相似度搜索  | Cosine/ANN算法                   |
| 大语言模型 | 答案生成   | GPT-4/Llama-2                  |

### 3.2 标准流程

1. **数据准备阶段**
   
   - 文档采集（PDF/HTML/数据库等）
   - 文本清洗与标准化
   - 分块处理（通常512-1024 tokens/块）

2. **向量化阶段**
   
   - 选择嵌入模型
   - 生成文本向量
   - 存入向量数据库

3. **查询处理阶段**
   
   - 用户问题向量化
   - 检索Top-K相关文档块
   - 生成增强提示词

4. **生成阶段**
   
   - 将检索结果输入LLM
   - 生成最终回答
   - 可选的后处理步骤

### 3.3 流程图

<div style="clear: both;">
    <img src="Images\RAG001.png" style="float: left; width: 40%; margin-right: 20px;">

    ```mermaid
    graph TD
        A[原始数据] --> B[文档加载]
        B --> C[文本分割]
        C --> D[向量嵌入]
        D --> E[向量数据库]
        F[用户问题] --> G[问题向量化]
        G --> H[向量检索]
        E --> H
        H --> I[构建提示词]
        I --> J[LLM生成]
        J --> K[返回答案]
    ```

</div>

<div style="clear: both;">

### 3.4 重点步骤解释

#### 3.4.1. 检索

当用户输入查询时，首先会经过嵌入模型的处理。嵌入模型将用户查询转换为向量表示，这个向量包含了查询的语义信息。然后，这个查询向量会被送入向量数据库中进行相似性搜索。向量数据库会计算查询向量与库中所有向量的相似度，并根据相似度得分对结果进行排序，最终返回与查询向量最相似的若干个向量及其对应的文档或数据片段。例如，在一个企业知识问答系统中，用户询问 "如何申请报销？"，嵌入模型将这个问题转换为向量后，向量数据库会在存储的企业报销制度文档、报销流程指南等相关向量中进行搜索，找到最相关的文档片段，如 "报销申请需要填写报销单，附上相关发票，并提交给部门领导审批……" 等内容

#### 3.4.2. 增强

检索到相关上下文后，接下来就是增强环节。在这个环节中，检索到的上下文信息会与用户查询整合在一起，填充到预先设计好的 prompt 模板中。prompt 模板就像是一个 “问题引导框架”，它规定了输入信息的组织方式和提问方式，以引导生成模块生成更好的回答。例如，一个简单的 prompt 模板可能是 “用户问题：{用户查询}，相关信息：{检索到的上下文}，请根据以上信息回答用户问题。” 通过将用户查询和检索到的上下文按照这样的模板进行组合，生成模块能够更好地理解问题的背景和要求，从而生成更准确、更有针对性的回答。

#### 3.4.3. 生成

将增强后的 prompt 输入到生成模块，也就是大语言模型中。大语言模型会对输入的 prompt 进行理解和分析，利用其内部的语言知识和语义理解能力，结合检索到的上下文信息，生成最终的回答。在生成过程中，大语言模型会根据预训练学到的语言模式和逻辑，逐步生成文本。例如，当生成模块接收到关于 “如何申请报销？” 的增强 prompt 后，它会根据其中的信息，生成详细的报销申请步骤和注意事项，如 “首先，您需要填写报销单，确保填写的信息准确无误。然后，将相关发票整齐粘贴在报销单背面。完成后，将报销单提交给您的部门领导进行审批。审批通过后，再提交到财务部门进行后续处理……”，最终将这个回答返回给用户。

</div>
以下是关于RAG系统组件和流程的详细补充说明，包括向量数据库选型和Python实现示例：

## 4. 组件深度解析

### 4.1 向量数据库选型比较

| 数据库          | 优点             | 缺点          | 适用场景     |
| ------------ | -------------- | ----------- | -------- |
| **FAISS**    | 高性能，Facebook开源 | 无持久化功能      | 研究/小规模生产 |
| **Pinecone** | 全托管服务，自动扩展     | 收费，有学习曲线    | 企业级应用    |
| **Weaviate** | 支持混合搜索，内置ML模型  | 需要自托管或购买云服务 | 复杂检索需求   |
| **Milvus**   | 分布式架构，高可用      | 运维复杂        | 超大规模向量数据 |

### 4.2 完整Python实现示例

```python:e:\01-Trae-Dev\rag_demo.py
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 文档加载
loader = WebBaseLoader(["https://example.com/ai-news"])
docs = loader.load()

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

# 3. 向量化存储
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. 检索增强生成
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever
)

result = qa_chain.run("AI领域最新突破是什么？")
print(result)
```

## 5. 流程优化技巧

### 5.1 分块策略优化

```python:e:\01-Trae-Dev\chunk_optimization.py
# 基于内容类型的分块策略
def dynamic_chunking(text, content_type):
    if content_type == "legal":
        return RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100
        ).split_text(text)
    elif content_type == "news":
        return TokenTextSplitter(
            chunk_size=1024,
            chunk_overlap=256
        ).split_text(text)
```

### 5.2 混合检索实现

```python:e:\01-Trae-Dev\hybrid_search.py
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer

# BM25关键词检索
corpus = [doc.page_content for doc in chunks]
bm25 = BM25Okapi(corpus)

# 混合检索函数
def hybrid_search(query, k=5):
    # 关键词检索
    bm25_scores = bm25.get_scores(query)
    # 向量检索
    vector_results = vector_db.similarity_search(query, k=k*3)
    # 结果融合...
```

## 四、生产环境建议

1. **监控指标**：
   
   - 检索召回率@K
   - 响应延迟P99
   - 生成结果相关性评分

2. **缓存策略**：
   
   - 高频查询结果缓存
   - 向量相似度预计算

3. **更新机制**：
   
   - 增量索引更新
   - 定时全量重建

4. **安全与隐私**：
   
   - 数据脱敏
   - 访问控制
   - 日志监控

5. **性能优化**：
   
   - 分布式部署
   - 硬件加速
   - 缓存优化

## 五、Python实现准备

以下是Python实现RAG系统的详细准备工作清单，我将从技术栈、数据准备、环境配置和最佳实践四个方面进行说明：

### 7.1 技术栈准备

#### 7.1.1 核心依赖库

```bash
# 基础框架
pip install langchain==0.1.0
# 向量计算
pip install faiss-cpu sentence-transformers
# 向量数据库
pip install chromadb
# 文档处理
pip install pypdf python-docx bs4 unstructured
```

#### 7.1.2 可选组件

```bash
# 混合检索增强
pip install rank_bm25 sklearn
# 本地模型支持
pip install transformers torch
# API服务
pip install fastapi uvicorn
```

### 7.2 数据准备规范

#### 7.2.1. **数据采集要求**：

- 格式支持：PDF/Word/Excel/Markdown/HTML
- 编码标准：UTF-8优先
- 元数据要求：包含source/title/update_time字段

#### 7.2.2. **预处理流程**：

```python:e:\01-Trae-Dev\07-RAG\preprocess.py
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import Language

# 专业领域特殊处理
def legal_doc_processing(text):
    # 处理法律条文特殊格式
    return clean_text

# 多语言支持
SUPPORTED_LANGUAGES = ['zh', 'en', 'ja']
```

### 7.3 开发环境配置

#### 推荐IDE配置

1. VSCode插件：
   
   - Python Extension Pack
   - Jupyter Notebook支持
   - Docker扩展

2. 调试配置：
   
   ```json:e:\01-Trae-Dev\07-RAG\.vscode\launch.json
   {
    "configurations": [
        {
            "name": "调试RAG流程",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/pipeline.py"
        }
    ]
   }
   ```

### 5.1 生产级规范

#### 5.1.1 性能优化

1. 索引优化：
   
   - 使用HNSW算法加速检索
   - 量化压缩向量维度
   - 定期重建索引

2. 缓存策略：
   
   ```python:e:\01-Trae-Dev\07-RAG\cache.py
   from functools import lru_cache
   
   ```

@lru_cache(maxsize=1000)
def cached_embedding(text):
    return embed_model.encode(text)

```

#### 7.4.2 安全规范
1. 数据脱敏流程：
   - 自动识别身份证/手机号等敏感信息
   - 使用正则表达式替换敏感字段

2. 访问控制：
```python:e:\01-Trae-Dev\07-RAG\auth.py
from fastapi import Depends, HTTPException

async def verify_token(token: str):
    if not valid_token(token):
        raise HTTPException(status_code=403)
```

### 7.5 测试验证方案

#### 7.5.1 自动化测试套件

```python:e:\01-Trae-Dev\07-RAG\tests/test_retrieval.py
import pytest

@pytest.mark.parametrize("query,expected", [
    ("报销流程", "填写报销单"),
    ("请假政策", "3天以上需审批")
])
def test_retrieval(query, expected):
    result = retriever(query)
    assert expected in result
```

#### 7.5.2 性能基准

```python:e:\01-Trae-Dev\07-RAG\benchmark.py
def run_benchmark():
    # 测试100次查询平均耗时
    return timeit.timeit(query_func, number=100)
```

### 7.6 文档规范

#### 7.6.1. API文档生成：

```python:e:\01-Trae-Dev\07-RAG\api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
```

#### 7.6.2. 项目文档结构：

```
/docs
  ├── API-Reference.md
  ├── Deployment-Guide.md
  └── Data-Specification.md
```

### 7.7 持续集成

```yaml:e:\01-Trae-Dev\07-RAG\.github\workflows/ci.yml
name: RAG CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

## 六、多模态扩展

以下是针对多模态(图片+文本)RAG系统的补充准备细节：

### 8.1 多模态处理技术栈

```bash
# 图像处理基础库
pip install pillow opencv-python
# 多模态模型
pip install transformers[torch] sentencepiece
# 向量化工具
pip install clip-anytorch
```

### 8.2 专用模型选型

| 模型名称     | 类型   | 特点     | 适用场景   |
| -------- | ---- | ------ | ------ |
| CLIP     | 图文匹配 | 开源/轻量级 | 通用图文检索 |
| BLIP-2   | 生成式  | 多任务支持  | 图片描述生成 |
| Florence | 微软出品 | 大规模预训练 | 专业领域   |

### 6.1 数据处理流程

```python:e:\01-Trae-Dev\07-RAG\multimodal_processor.py
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 初始化多模态处理器
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

def process_image(img_path):
    # 提取视觉特征
    image = Image.open(img_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

    # 生成图片描述
    generated_ids = model.generate(**inputs)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {
        "image_embedding": inputs.pixel_values,
        "text_description": description
    }
```

### 8.4 存储方案优化

1. **混合索引结构**：
   
   ```python:e:\01-Trae-Dev\07-RAG\multimodal_db.py
   from chromadb.utils.embedding_functions import MultiModalEmbeddingFunction
   
   ```

# 创建多模态集合

multimodal_ef = MultiModalEmbeddingFunction(
    text_embedding_model="BAAI/bge-small",
    image_embedding_model="clip-vit-base-patch32"
)

collection = client.create_collection(
    name="multimodal",
    embedding_function=multimodal_ef
)

```

### 8.5 查询处理增强
```python:e:\01-Trae-Dev\07-RAG\multimodal_query.py
def hybrid_retrieval(query, image=None, top_k=5):
    # 文本查询分支
    if isinstance(query, str):
        text_results = text_db.query(query_texts=[query], n_results=top_k)

    # 图像查询分支 
    if image is not None:
        img_vec = image_model.encode(image)
        image_results = image_db.query(query_embeddings=[img_vec], n_results=top_k)

    # 结果融合策略
    return fuse_results(text_results, image_results)
```

### 8.6 领域适配建议

1. **医疗影像**：
   
   - 使用专业领域模型如CheXzero
   - DICOM格式特殊处理
   - 添加病变部位标记元数据

2. **电商场景**：
   
   - 商品属性结构化存储
   - 支持以图搜图
   - 多角度图片索引

3. **教育资料**：
   
   - 公式截图转LaTeX
   
   - 讲义图文关联
   
   - 手写笔记识别
     
     ```
     
     ```

需要补充具体某个垂直领域的实现示例吗？例如可以演示：

1. 医疗报告中的影像+诊断文本联合检索
2. 商品图片与评论的关联分析
3. 学术论文图表与正文的交叉引用
