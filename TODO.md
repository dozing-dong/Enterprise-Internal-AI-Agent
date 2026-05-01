# TEST01 RAG Demo 学习路线图

这个文件服务于一个明确目标：

把当前项目从“能跑的 RAG demo”逐步提升为“能体现 RAG 与 Agent 核心能力的学习型项目”。

前端在这个项目里只是辅助观察工具，不是主学习方向。

---

## 学习目标

你当前真正要掌握的主线，不是前端，而是这两条：

1. RAG
2. Agent

更具体一点，这个项目应该帮助你逐步掌握：

- 文档加载与清洗
- 文本切分策略
- 向量检索
- BM25 与混合检索
- 检索结果可解释性
- RAG 评测与对比
- 查询改写
- 重排序
- 检索失败时的回退策略
- 工具调用
- Agent 最小路由
- 多工具协作

---

## 文档维护规则

以后每次有实质性代码变更，都必须同步更新 `TODO.md`。

以后每次完成一轮代码修改后，都必须额外给出一条中文版的“更改总结消息”，方便你直接整理成本地 git commit 说明。

这里的“实质性代码变更”包括：

- 新增或完成一个功能
- 调整项目目录结构
- 修改接口返回结构
- 修改主学习路线
- 修改当前推荐的下一步

每次更新 `TODO.md` 时，至少检查这 4 项：

- 当前状态有没有变化
- 文件路径有没有变化
- 当前最推荐的下一步是否还合理
- 是否有内容偏离“RAG / Agent 主线”
- 是否已经补上本轮修改的更改总结消息

---

## 当前项目快照

### 当前架构

```text
zjh/
├── backend/
│   ├── api/         # FastAPI 接口层
│   ├── data/        # 数据读取、切分、格式化
│   ├── rag/         # 模型、检索、RAG 链
│   ├── storage/     # 历史记录存储
│   ├── cli.py       # CLI 逻辑
│   ├── config.py    # 配置
│   └── runtime.py   # 统一初始化
├── frontend/        # 最小可视化页面
├── evals/           # 评测脚本
├── build_index.py   # 手动重建索引入口
├── TEST01.py
├── run_api.py
├── README.md
└── TODO.md
```

### 当前已有能力

- `DONE` 向量检索
- `DONE` BM25 检索
- `DONE` 混合检索
- `DONE` 本地历史记录
- `DONE` FastAPI 接口
- `DONE` 返回 `sources`
- `DONE` 最小前端页面
- `DONE` 索引构建与在线服务分离
- `DONE` 独立索引重建入口
- `DONE` 最小 RAG 检索评测脚本
- `DONE` Chunk 策略对比脚本
- `DONE` RAGAS 检索评测脚本

### 当前真正的短板

这些才是现在最值得补的内容：

- `TODO` 检索质量仍然很基础
- `TODO` 目前只有最小评测基线，还缺更系统的评测集
- `TODO` 缺少查询改写
- `TODO` 缺少重排序
- `TODO` 缺少明确的失败回退策略
- `TODO` 没有真正的 Agent 能力
- `TODO` 还没有工具抽象和路由能力

---

## 已完成里程碑

### 阶段 0：把 demo 跑起来

- `DONE` 把单文件 demo 拆成多个模块
- `DONE` 保留向量检索 + BM25 混合检索
- `DONE` 保留会话历史

### 阶段 1：把 demo 变成最小服务

- `DONE` 增加 FastAPI 服务
- `DONE` 增加 `/chat`
- `DONE` 增加 `/history/{session_id}`
- `DONE` 增加 `/health`
- `DONE` 保留 CLI 入口

### 阶段 2：让 RAG 可观察

- `DONE` `/chat` 返回 `sources`
- `DONE` CLI 可以打印参考片段
- `DONE` 检索结果和答案使用同一批文档

### 阶段 3：做轻量重组和最小页面

- `DONE` 完成轻量目录重组
- `DONE` 增加最小前端页面

注意：

这一步已经够了，前端暂时不再作为主任务推进。

### 阶段 4：拆开离线索引和在线服务

- `DONE` 服务启动时默认复用已有索引
- `DONE` 新增 `build_index.py` 作为独立建索引入口
- `DONE` CLI 和 API 不再承担默认重建索引的职责

### 阶段 5：建立最小 RAG 评测基线

- `DONE` 新增固定问题集
- `DONE` 能比较 `vector`、`bm25`、`hybrid` 三种检索方式
- `DONE` 能输出标准检索指标和分类汇总
- `DONE` 已引入 RAGAS 作为工具化检索评测补充

### 阶段 6：开始基于评测优化切分策略

- `DONE` 把 chunk 配置集中到 `backend/config.py`
- `DONE` 默认在线服务继续使用固定策略，避免实验代码干扰主链
- `DONE` 新增 `evals/chunk_eval.py` 对比不同 chunk 策略

---

## 主学习路线

下面才是接下来真正要做的内容。

顺序是按学习收益排的，不建议跳着做。

---

## 第一阶段：把 RAG 做扎实

### 任务 1：优化历史接口，但只做最小必要改造

- 状态：`TODO`
- 优先级：`中`
- 为什么现在仍然保留这个任务：
  它不是为了学前端，而是为了避免前端调试时被原始 JSON 干扰。
- 你会学到：
  - 内部存储格式和对外接口格式的区别
- 建议改动文件：
  - `backend/storage/history.py`
  - `backend/api/app.py`
- 完成标准：
  - `GET /history/{session_id}` 返回更清晰的数据结构
  - 不再直接把底层原始结构暴露给页面

### 任务 2：把索引构建和在线服务分开

- 状态：`DONE`
- 优先级：`最高`
- 为什么很重要：
  这是 RAG 工程的基础。真实系统不会每次启动服务都重建索引。
- 你会学到：
  - 离线索引与在线检索的边界
  - 为什么构建链路和服务链路要分离
- 建议改动文件：
  - `backend/runtime.py`
  - `backend/rag/retrievers.py`
  - `run_api.py`
- 完成标准：
  - 服务启动时优先复用已有向量库
  - 手动执行时才重建索引

已完成说明：

- `backend/runtime.py` 现在把“重建索引”和“加载运行时”拆成了两个函数
- `backend/rag/retrievers.py` 现在区分了 `rebuild_vectorstore()` 和 `load_vectorstore()`
- 在线服务如果没有现成索引，会明确提示先运行 `python build_index.py`

### 任务 2.1：增加最小索引管理入口

- 状态：`DONE`
- 优先级：`高`
- 为什么要加这个子任务：
  只说“分开”还不够，必须有一个明确入口来重建或刷新索引，否则后面使用会混乱。
- 你会学到：
  - 离线构建入口如何设计
  - 为什么 RAG 项目需要单独的索引管理命令
- 建议改动文件：
  - `backend/runtime.py`
  - `TEST01.py` 或新增脚本文件
- 完成标准：
  - 至少有一个明确入口用于重建索引
  - 不影响当前 API 服务启动

已完成说明：

- 根目录已经新增 `build_index.py`
- 现在可以先运行 `python build_index.py`，再运行 `python run_api.py` 或 `python TEST01.py`

### 任务 3：增强数据输入能力

- 状态：`TODO`
- 优先级：`高`
- 为什么很重要：
  现在的数据源虽然比最初好，但仍然偏固定。RAG 的核心前提是“能处理真实文档”。
- 你会学到：
  - 多文件加载
  - 文档来源追踪
  - 数据清洗的最小流程
- 建议改动文件：
  - `backend/data/knowledge_base.py`
  - `backend/config.py`
- 完成标准：
  - 能从指定目录加载多个 TXT 或 Markdown 文件
  - 每个文档都带来源信息

### 任务 4：对比不同 chunk 策略

- 状态：`DONE`
- 优先级：`高`
- 为什么很重要：
  很多人会用 RAG，但不知道 chunk 为什么这样切。这个任务能帮你真正理解“切分策略影响检索质量”。
- 你会学到：
  - `chunk_size`
  - `chunk_overlap`
  - 按段切分和按字符切分的差异
- 建议改动文件：
  - `backend/data/processing.py`
  - `backend/config.py`
- 完成标准：
  - 至少支持两种切分配置
  - 你能比较它们对检索结果的影响

已完成说明：

- `backend/config.py` 现在集中维护 `CHUNK_PROFILES`
- `backend/data/processing.py` 现在支持按策略名称切分文档
- 已新增 `evals/chunk_eval.py`，固定检索器只比较 chunk 策略
- 当前在线服务仍然使用默认切分策略，实验脚本不会污染主链
- chunk 对比脚本已经同步使用标准检索指标输出结果

### 任务 5：增加 RAG 评测脚本

- 状态：`DONE`
- 优先级：`最高`
- 为什么这是核心任务：
  没有评测，你就不知道“改进是否真的变好”。这比继续做页面重要得多。
- 你会学到：
  - 如何构建小型问答集
  - 如何比较不同检索策略
  - 为什么 RAG 优化必须依赖评测
- 建议改动文件：
  - 新增 `evals/` 或 `tests/`
  - `backend/rag/retrievers.py`
  - `backend/data/knowledge_base.py`
- 完成标准：
  - 至少有一组固定测试问题
  - 能比较 `vector`、`bm25`、`hybrid` 的效果

已完成说明：

- 已新增 `evals/rag_eval.py`
- 当前评测脚本会加载固定问题集，并比较 `vector`、`bm25`、`hybrid`
- 当前输出使用标准指标 `Recall@1`、`Recall@3`、`Precision@1`、`Precision@3`、`MRR`
- 当前脚本兼容 `python evals/rag_eval.py` 的直接运行方式
- 当前问题集已经加入口语化、同义改写、解释型和场景型问题，更接近真实项目验证
- 已新增 `evals/ragas_retrieval_eval.py`，可用 RAGAS 的 ID-based 指标辅助评测

### 任务 6：加入查询改写

- 状态：`TODO`
- 优先级：`高`
- 为什么很重要：
  真实用户的问题往往不适合直接检索。查询改写是从“基础 RAG”走向“更强 RAG”的关键一步。
- 你会学到：
  - query rewrite 的作用
  - 原始 query 和 rewrite query 的差别
  - 为什么 rewrite 往往先于复杂 Agent
- 建议改动文件：
  - `backend/rag/chain.py`
  - `backend/rag/models.py`
  - `backend/api/app.py`
- 完成标准：
  - 至少支持“原始问题 -> 改写问题 -> 检索”这一条链路
  - 可以观察改写前后的差异

### 任务 7：加入重排序

- 状态：`TODO`
- 优先级：`高`
- 为什么很重要：
  只有混合检索还不够。很多高质量 RAG 系统都会在召回后再做 rerank。
- 你会学到：
  - 召回和排序是两件不同的事
  - 为什么 top-k 结果不一定已经最优
- 建议改动文件：
  - `backend/rag/retrievers.py`
  - 可能新增 `backend/rag/rerank.py`
- 完成标准：
  - 至少支持一个简单的 rerank 方案
  - 可以比较 rerank 前后的结果变化

### 任务 8：加入检索失败回退策略

- 状态：`TODO`
- 优先级：`中`
- 为什么要做：
  RAG 不是永远检索成功。你需要明确当“检索结果弱”时系统该怎么处理。
- 你会学到：
  - 空结果处理
  - 低置信度处理
  - 何时拒答，何时降级
- 建议改动文件：
  - `backend/rag/chain.py`
  - `backend/api/app.py`
- 完成标准：
  - 当来源不足时，系统能明确提示“信息不足”
  - 不再只是无条件生成答案

---

## 第二阶段：从 RAG 过渡到最小 Agent

注意：

Agent 不是一开始就上多智能体。

更合理的顺序是：

先把 RAG 做扎实，再做一个“最小工具调用 Agent”。

### 任务 9：抽象出工具接口

- 状态：`TODO`
- 优先级：`中`
- 为什么要做：
  Agent 的基础不是花哨路由，而是“系统能调用不同工具”。
- 你会学到：
  - 什么叫 tool abstraction
  - 为什么把检索、历史、搜索封装成工具很重要
- 建议改动文件：
  - 新增 `backend/agent/`
  - `backend/rag/chain.py`
  - `backend/api/app.py`
- 完成标准：
  - 至少抽象出 2 个工具
  - 工具职责清楚，不和 API 层混在一起

### 任务 10：做一个最小单 Agent 路由

- 状态：`TODO`
- 优先级：`高`
- 为什么要做：
  这是从“RAG 问答”进入“Agent 应用”的第一步。
- 你会学到：
  - 意图分类
  - 什么场景该走 RAG，什么场景该走普通聊天或其他工具
- 建议改动文件：
  - 新增 `backend/agent/router.py`
  - `backend/api/app.py`
- 完成标准：
  - 至少能区分 2 到 3 类请求
  - 路由逻辑可以被解释

### 任务 11：让 Agent 调用检索工具和历史工具

- 状态：`TODO`
- 优先级：`高`
- 为什么要做：
  这是最接近“大模型应用开发工程师”岗位的地方之一。
- 你会学到：
  - Agent 如何与 RAG 融合
  - 工具调用并不等于多智能体
- 建议改动文件：
  - `backend/agent/`
  - `backend/storage/history.py`
  - `backend/rag/chain.py`
- 完成标准：
  - Agent 至少能根据请求决定是否调用检索
  - Agent 至少能读取或利用会话历史

### 任务 12：增加 Agent 评测案例

- 状态：`TODO`
- 优先级：`中`
- 为什么要做：
  Agent 如果没有评测，后面会非常难调。
- 你会学到：
  - 路由正确率
  - 工具调用正确率
- 建议改动文件：
  - 新增 `evals/agent_*.py`
- 完成标准：
  - 至少有一组固定问题用于测试路由和工具调用

---

## 第三阶段：补最小工程能力

### 任务 13：增加基础测试

- 状态：`TODO`
- 优先级：`高`
- 为什么要做：
  没有测试的项目，越往后越难维护。
- 建议改动文件：
  - 新增 `tests/`
  - `backend/api/app.py`
  - `backend/rag/`
  - `backend/storage/history.py`
- 完成标准：
  - 至少覆盖 `health`、`chat`、`history`

### 任务 14：增加 `.env` 配置

- 状态：`TODO`
- 优先级：`中`
- 为什么要做：
  模型名、路径、检索参数不能一直写死。
- 建议改动文件：
  - `backend/config.py`
  - `requirements.txt`
- 完成标准：
  - 至少能从环境变量读取模型名、路径和检索参数

---

## 当前最推荐的下一步

如果只做一个，我建议现在先做：

`任务 6：加入查询改写`

原因是：

- 任务 4 已经完成，你已经开始具备用评测驱动优化的能力
- 下一步最值得提升的是改写类、口语化问题的检索表现
- 你当前的评测集里正好已经包含 paraphrase 和 colloquial 类问题，适合直接验证 rewrite 的收益

如果你想先补一个更偏工程基础的任务，我建议做：

`任务 3：增强数据输入能力`

---

## 暂时不要做的事

现阶段先不要把精力花在这些方向：

- React / Vue 工程化前端
- 复杂页面样式优化
- Docker 部署
- 多智能体
- Neo4j
- MySQL
- Redis
- LangGraph 大工作流

原因不是这些不重要，而是它们会稀释你当前最关键的学习主线：

`RAG -> RAG 优化 -> 最小 Agent -> Agent 评测`

---

## 2026-03-19 最新进展

- `DONE` `evals/chunk_eval.py` 已删除自定义标准指标输出，改为与 `evals/ragas_retrieval_eval.py` 对齐，统一使用 `RAGAS Context Precision` 和 `RAGAS Context Recall`
- `DONE` 已新增 `backend/rag/rerank.py`，把 rerank 做成独立模块，而不是把重排逻辑直接塞进原检索器实现
- `DONE` 在线主链路现在默认采用“两阶段检索”：
  先用 `hybrid` 召回更多候选文档，再用 `DashScopeRerank` 做二次排序
- `DONE` `evals/rag_eval.py` 的检索器构建逻辑已加入 `hybrid_rerank`，后续可以直接比较 `hybrid` 和 `hybrid_rerank`

## 当前推荐的下一步

如果只做一个，我建议现在先做：

`任务 6：加入查询改写`

原因是：

- 当前项目已经先后完成了基础召回、chunk 对比、RAGAS 工具化评测、以及最小 rerank
- 下一步最值得补的是“用户问题本身”的优化，而不是继续堆更多底层检索组件
- 这样你就能形成一个更完整的 RAG 优化链路：
  `query rewrite -> retrieval -> rerank -> generation`

## 本轮更改总结消息

```text
feat: align chunk evaluation with RAGAS and add minimum rerank pipeline

- replace evals/chunk_eval.py custom metrics with RAGAS Context Precision/Recall
- add backend/rag/rerank.py to wrap a base retriever with DashScope rerank
- enable two-stage retrieval in runtime: hybrid recall first, rerank second
- extend eval retriever builder with hybrid_rerank for before/after comparison
- update TODO.md with latest learning progress and next recommended task
```

---

## 2026-03-19 查询改写进展

- `DONE` 已新增 `backend/rag/rewrite.py`，把“查询改写”做成独立模块，而不是把 prompt 直接写死在检索器或 API 里
- `DONE` 在线主链路现在已经支持：
  `original question -> retrieval question -> retrieval -> rerank -> generation`
- `DONE` `/chat` 现在会同时返回 `original_question` 和 `retrieval_question`
- `DONE` CLI 现在也会打印“原问题”和“检索问题”，便于你直接观察改写效果
- `DONE` `evals/rag_eval.py` 现在已加入 `hybrid_rerank_rewrite`
- `DONE` 现在可以直接用 `evals/ragas_retrieval_eval.py` 比较：
  `hybrid`
  `hybrid_rerank`
  `hybrid_rerank_rewrite`

## 当前推荐的下一步

如果只做一个，我建议现在先做：

`任务 3：增强数据输入能力`

原因是：

- 当前检索优化链路已经基本成型：`rewrite -> retrieval -> rerank`
- 现在最明显的短板已经不是链路结构，而是数据源仍然偏固定、偏单一
- 如果不尽快支持多文件和真实文档输入，后面的评测和优化很容易停留在“小样本 demo”层面

## 本轮更改总结消息

```text
feat: implement minimum query rewrite pipeline for retrieval

- add backend/rag/rewrite.py as a standalone query rewrite module
- support original question -> retrieval question flow in runtime, CLI, and /chat
- expose retrieval_question in API response for learning and debugging
- extend eval retrievers with hybrid_rerank_rewrite for before/after comparison
- update TODO.md with task 6 progress and set the next recommended task to stronger data ingestion
```

---

## 2026-03-19 公开数据集替换进展

- `DONE` 默认知识库已从原来的本地 demo 文本切换为公开数据集 `explodinggradients/fiqa`
- `DONE` 当前知识库默认改为直接使用 fiqa 全量 `corpus` 语料建库，
  同时保留 `ragas_eval_v3` 参考上下文用于公开评测映射
- `DONE` 当前评测题集已替换为 fiqa `ragas_eval_v3` baseline，共 30 题
- `DONE` 当前公开评测使用稳定的 `context_id` 作为标签，不再依赖手工段落编号
- `DONE` `build_index.py`、`runtime`、`ragas_retrieval_eval.py`、`chunk_eval.py` 现在都默认围绕 fiqa 公共数据集运行

## 当前推荐的下一步

如果只做一个，我建议现在先做：

`任务 3：增强数据输入能力`

原因是：

- 你已经完成了“最小 demo 数据 -> 公开数据集”的升级
- 下一步更有价值的是把“固定公共数据集”继续升级成“可配置的数据源输入”
- 也就是说，后面应该学会在：
  本地文件
  公开数据集
  自己整理的数据
  之间切换，而不是把数据源继续写死在代码里

## 本轮更改总结消息

```text
feat: replace demo dataset with public FIQA evaluation dataset

- switch knowledge base loading from local demo text to explodinggradients/fiqa
- build a manageable public corpus from fiqa eval reference contexts plus corpus distractors
- replace handwritten eval cases with 30 public fiqa ragas_eval_v3 baseline questions
- keep retrieval labels stable by assigning context_id values to public reference contexts
- update rag_eval.py and ragas_retrieval_eval.py to evaluate against the new fiqa-based public benchmark
```

---

## 2026-03-25 FIQA 建库规模调整
- `DONE` 当前知识库默认改为使用 `ragas_eval_v3` 参考上下文 + 前 5000 条 `fiqa corpus` 文档
- `DONE` 同时继续保留 `ragas_eval_v3` 参考上下文，保证公开评测里的 `context_id` 映射稳定
- `DONE` 把 corpus 取样规模改成配置项，避免再次回到全量建库导致成本失控
- `DONE` `evals/rag_eval.py` 和项目文档已同步更新为 5000 条 corpus 文档

## 本轮变更总结消息

```text
feat: limit FIQA knowledge base build to 5000 corpus documents

- keep ragas_eval_v3 reference contexts for stable public evaluation mapping
- cap fiqa corpus ingestion at 5000 documents to control indexing cost
- move the corpus limit into config instead of hardcoding it in the build flow
- update evaluation and project docs to reflect the new default corpus size
```

---

## 2026-03-25 低成本验证链路调整
- `DONE` 默认关闭 rerank 和 query rewrite，避免评测链路继续依赖外部计费模型
- `DONE` `evals/rag_eval.py` 现在只在对应开关启用时才构建 `hybrid_rerank` 和 `hybrid_rerank_rewrite`
- `DONE` 项目默认目标调整为优先支持低成本 RAGAS 验证

## 本轮变更总结消息

```text
feat: switch validation pipeline to local zero-cost embeddings

- disable rerank and query rewrite by default to avoid external model dependencies
- only build rerank-based eval retrievers when the related feature flags are enabled
```

---

## 2026-03-25 自构造数据集落地
- `DONE` 新增项目内置自构造知识库 `backend/data/data/local_eval/documents.json`
- `DONE` 新增配套评测集 `backend/data/data/local_eval/eval_cases.json`
- `DONE` 默认数据源切换为 `local_eval`，不再强依赖公开数据集下载
- `DONE` 当前自构造数据集包含 12 条知识文档和 28 条问答样本
- `DONE` 已验证 `build_index.py` 可直接基于该数据集完成本地建库

## 本轮变更总结消息

```text
feat: add a built-in synthetic dataset for local RAG validation

- add local knowledge documents and paired eval cases under backend/data/data/local_eval
- switch the default dataset source to local_eval for low-cost offline validation

## 2026-03-25 Dataset Consolidation

- `DONE` 扩充本地自构造评测集到 18 条知识文档、72 条问答样本
- `DONE` 删除代码中的 FIQA 和 fallback 数据分支，知识库加载只保留 `local_eval`
- `DONE` 项目数据目录只保留 `backend/data/data/local_eval`
- `DONE` 删除本地 `local_hash` embedding 相关代码，项目只保留商用 DashScope embedding
- route build_documents and eval case loading through the selected dataset source
- verify that build_index.py can rebuild the vector index from the synthetic dataset
```
