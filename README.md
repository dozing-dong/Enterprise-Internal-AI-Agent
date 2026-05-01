# Evaluable RAG Demo

一个面向学习与验证的最小化 RAG 项目。当前版本只保留项目内置的本地评测数据集，不再依赖外部公开数据集下载。

## 当前定位

- 离线建库，在线检索问答
- 向量检索、BM25、混合检索
- 商用 embedding，使用 DashScope
- 基于显式 `context_id` 的检索评测与 RAGAS 评测

这个项目的目标是验证 RAG 工程链路是否跑通，不是追求生产级效果，也不是多智能体框架。

## 数据集

当前只保留一套项目内置自构造数据集：

- `backend/data/data/local_eval/documents.json`
- `backend/data/data/local_eval/eval_cases.json`

数据规模：

- 18 条知识库文档
- 72 条问答样本
- 每条问答包含问题、标准答案和 `reference_context_ids`

题型覆盖定义、比较、原因、流程、失败处理、索引重建、评测设计、负样本、chunk 边界和本地 embedding 局限等主题。

## 主要流程

1. 用 `build_index.py` 离线读取本地知识库并构建 Chroma 索引。
2. 在线服务加载已有索引，执行向量检索、BM25 或混合检索。
3. 评测脚本基于显式标注的 `context_id` 计算检索指标。

默认配置下：

- embedding model: `text-embedding-v4`
- rerank: 关闭
- query rewrite: 关闭

这样做是为了让向量检索只使用商用 embedding。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

配置环境变量：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

构建索引：

```bash
python build_index.py
```

启动 CLI：

```bash
python TEST01.py
```

启动 API：

```bash
python run_api.py
```

运行检索评测：

```bash
python evals/rag_eval.py
```

运行 RAGAS 检索评测：

```bash
python evals/ragas_retrieval_eval.py
```

运行 chunk 策略对比：

```bash
python evals/chunk_eval.py
```

## 目录

```text
zjh/
├─ backend/
│  ├─ api/
│  ├─ data/
│  ├─ rag/
│  ├─ storage/
│  ├─ cli.py
│  ├─ config.py
│  └─ runtime.py
├─ evals/
├─ frontend/
├─ build_index.py
├─ run_api.py
├─ TEST01.py
└─ README.md
```

## 说明

- DashScope embedding 默认需要可用的 API Key 和正常计费状态。
- 这套数据集是教学型、可控型数据，不是领域真实生产语料。
- 如果你要做真实效果评估，应换成更大、更真实、带噪声和负样本分布的数据集。
