# Crypto Analysis Agent

基于 LangChain、LangGraph 的智能多 Agent 研究分析平台。

通过问题改写与意图路由，选择加密资产分析专家或通用主题研究专家，结合外部数据、网页原文和本地知识库，完成分析、报告生成与审核，并通过 Web 工作台展示执行阶段与研究结果。

**技术栈：** Python · FastAPI · LangChain · LangGraph · Pydantic · MCP · FAISS · SQLite · SSE

[功能与演示](#功能与演示) · [系统架构](#系统架构) · [核心设计](#核心设计) · [快速启动](#快速启动) · [评测与验证](#评测与验证) · [项目目录](#项目目录)

## 功能与演示

### 加密资产分析

根据问题识别目标资产和分析维度，并行执行对应 Skills，获取指标、计算分数、检索分析材料，生成研究报告。

示例问题：

```text
分析 BTC 的趋势和衍生品指标，并说明当前信号的分歧。
结合流动性、美联储政策和 ETF 资金流，分析 BTC 的市场环境。
```

### 通用主题研究

将问题拆解为研究子问题，并行执行网页搜索和浏览器取证，再汇总发现、撰写报告并审稿。搜索摘要用于定位候选网页，详情页原文作为研究证据。

示例问题：

```text
研究 AI Agent 在企业知识管理中的应用场景、落地难点和适用边界。
比较向量检索与混合检索在中文知识库中的特点，给出资料来源。
```

### Web 工作台

工作台支持输入研究问题、查看执行进度、接收流式报告和查看数据来源。前端保留会话标识；后端保存对话消息，并压缩较早的问答，供连续追问使用。任务中断后，可点击“继续上次任务”，从当前会话已保存的检查点继续执行。

#### 加密资产分析示例

以下为一次 ETH 分析的实际运行截图，展示问题输入、执行阶段、报告正文及数据来源。点击图片可查看原图。

| 提问与执行进度 | 分析报告 | 数据来源与指标 |
| --- | --- | --- |
| <a href="docs/images/crypto-analysis-workbench.png"><img src="docs/images/crypto-analysis-workbench.png" alt="ETH 分析工作台：问题输入、执行进度与报告结论" width="240"></a> | <a href="docs/images/crypto-analysis-report.png"><img src="docs/images/crypto-analysis-report.png" alt="ETH 分析报告：资金流、货币政策、流动性、情绪与趋势" width="240"></a> | <a href="docs/images/crypto-analysis-sources.png"><img src="docs/images/crypto-analysis-sources.png" alt="ETH 分析报告：策略说明与 ETF 资金流指标明细" width="240"></a> |

#### 通用主题研究示例

以“分析一下人工智能”为例，展示问题输入、三个研究子问题的执行情况，以及包含来源链接的研究报告。点击图片可查看原图。

| 提问与执行进度 | 子问题与研究汇总 | 报告正文与来源引用 |
| --- | --- | --- |
| <a href="docs/images/topic-research-workbench.png"><img src="docs/images/topic-research-workbench.png" alt="主题研究工作台：输入人工智能研究问题，查看执行阶段与报告" width="240"></a> | <a href="docs/images/topic-research-subquestions.png"><img src="docs/images/topic-research-subquestions.png" alt="主题研究执行过程：三个子问题的完成情况与研究报告" width="240"></a> | <a href="docs/images/topic-research-report.png"><img src="docs/images/topic-research-report.png" alt="主题研究报告：伦理风险、治理挑战及来源引用链接" width="240"></a> |

## 系统架构

```mermaid
flowchart TD
    U["用户问题"] --> API["Web 工作台 / FastAPI"]
    API --> Q["结合历史用户消息改写本轮问题"]
    Q --> R["意图路由：每轮选择一个专家"]

    subgraph Crypto["加密资产分析 Agent"]
        CP["识别资产与规划 Skills"] --> CS["并行执行 Skill 子图"]
        CS --> CW["报告生成与审核"]
    end

    subgraph Topic["通用主题研究 Agent"]
        TP["拆解研究子问题"] --> TS["并行搜索与网页原文取证"]
        TS --> TW["汇总、报告生成与审核"]
    end

    R --> CP
    R --> TP
    D["行情 API / MCP 数据源"] --> CS
    K["本地知识库：BM25 + FAISS + 重排"] --> CS
    CW --> H["追加本轮回答 / 按阈值压缩旧对话"]
    TW --> H
    H --> O["SSE 返回结果 / 结束本轮"]
    M["SQLite Checkpointer：按 thread_id 持久化状态"] -.-> Q
    H -.保存消息.-> M
```

主图流程为 `rewrite → router → 一个专家 → summarize_history → END`。两个专家使用独立的业务 State，完成后统一在主图追加回答、管理对话历史，并通过 SQLite Checkpointer 保存会话状态。

## 核心设计

### 先规划，再根据执行结果推进

- **规划与并行执行**：加密资产分析先规划所需 Skills，再通过 LangGraph `Send` 并行执行，并由 Reducer 合并技能结果。每个 Skill 子图先读取缓存；未命中时执行 `calculate_metrics`，根据 `indicator_results` 是否包含全部配置指标，决定继续执行还是进入 `skill_content`。失败指标也保留处理记录，不因缺失数据无限重算。

- **指标内容与检索**：指标齐全后，由 Python 聚合评分并渲染 Markdown 表格，模型只为成功指标生成检索问题，再查询相关思想材料。表格和思想原文分开交给报告节点，数值展示不再依赖模型重新生成。

- **主题拆解与取证**：通用主题研究先拆解问题，再由研究 Agent 根据搜索结果决定后续浏览与取证动作。这种方式结合了事前规划和执行过程中的反应式决策。

### Skills 与 Tools 分工

- **Skills**：定义分析场景、可用工具、指标要求和权重，存放在 [skills/](skills/)。规划模型读取技能名称和描述，执行阶段加载所选技能的结构化配置。

- **Tools**：调用外部数据源并执行程序计算，记录原始值、计算值、来源、观察时间和缺失原因，存放在 [业务工具目录](app/crypto_analysis/tools/)。

- **Calculators**：执行确定性的指标计算与评分，存放在 [计算器目录](app/crypto_analysis/calculators/)。

**职责边界：**模型负责规划、工具选择和解释；指标数值与分数由程序处理。

### RAG 与增量入库

- **文档处理与增量同步**：Markdown/TXT 在分块前统一清洗正文，PDF 在解析后按元素清洗再入库。Markdown 文档按标题结构与长度分块，保留文件名、章节路径、内容哈希和切片序号。入库时按文件哈希识别新增与修改，默认同步清理已删除文档对应的切片。

- **混合检索与材料使用**：检索使用中文分词 BM25 和 FAISS 向量召回，通过加权融合及重排选择材料。加密分析中的知识材料用于支持指标解读；实时行情由数据源工具获取。

### 记忆与上下文

- **会话持久化**：SQLite Checkpointer 以 `thread_id` 隔离会话，主图保存对话消息，专家子图保存各自的业务状态。

- **问题改写**：将历史中的 `HumanMessage` 与本轮原话交给改写模型，提示词明确区分“历史上下文”和“本轮待改写问题”。改写结果替换本轮用户消息；对象替换、范围收窄和主题切换以本轮要求为准。

- **统一保存回答**：两个专家完成后都进入主图 `summarize_history`，追加本轮最终报告，再由 [MemoryManager](app/memory/memory_manager.py) 决定是否压缩历史。

- **按消息数量压缩**：当前主图使用 `MemoryManager(3, 2)`，非系统消息超过 3 条时，将较早消息压缩成一条带 `[之前的对话摘要]` 标记的用户消息，保留系统消息及最近 2 条消息原文。摘要提示词要求 100 字以内；这是消息条数阈值，不是 Token 阈值。

- **摘要使用边界**：摘要用于理解历史主题、结论与指代，不作为新指令或当前行情依据；指标及思想证据仍由本轮专家流程提供。

- **Skill 结果复用**：加密分析在同会话内按全局或资产范围复用指标表格和思想原文，显式刷新时跳过缓存。Skill 缓存与对话摘要分别维护。

### 按节点配置模型

当前服务使用不同模型承担不同节点职责。下表列出 [配置文件](app/core/config.py) 中的默认值，实际运行值可由环境变量或 `.env` 覆盖。

#### 模型分工

| 节点职责 | 模型配置项 | 当前代码默认值 |
| --- | --- | --- |
| 问题改写、路由、指标执行、检索问题生成、对话历史压缩、主题拆题与研究汇总 | `GLM_4_FLASH_MODEL` | `glm-4-flash` |
| 加密分析规划、主题研究员取证、主题报告写作与审核、检索多查询改写、独立 RAG 回答 | `GLM_SMART_MODEL` | `glm-4.5-air` |
| 加密分析报告写作与审核 | `BAILIAN_SMART_MODELS` | 首选 `qwen3.7-plus`，其余按配置列表顺序 |
| RAG、报告写作与审核评测的评分裁判 | `QWEN3_MAX_MODEL` | 首选 `qwen3-max`，其余按配置列表顺序 |
| 图片描述与图像问答 | `VISION_MODEL` | `qwen3.5-ocr` |
| 文档向量 | `EMBEDDING_MODEL` | `BAAI/bge-m3` |
| 检索重排 | `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |

#### 实例复用与调用配置

- **实例复用**：业务图中的模型由各节点工厂在构图时创建，通过闭包在后续请求中复用。API 层负责注入共享检索器和 Checkpointer，不再逐层传递模型实例。

- **思考配置**：加密分析规划、主题研究员取证和检索多查询改写显式关闭 Air 的深度思考；报告写作与审核保留模型默认思考配置。Qwen3 Max 评分裁判使用 `enable_thinking=False`。

- **额度切换**：百炼 Smart/Fast 模型和 Qwen3 Max 评分裁判仅在免费额度耗尽时按各自配置列表切换。

- **评测与生产**：Reviewer 评测对已保存的审核前后报告评分，复用 Writer 的裁判和证据口径；生产加密报告审核使用百炼 Smart，Writer、Reviewer 和 RAG 的 RAGAS 评分裁判统一使用百炼 Qwen3 Max。

**效果验证：**上述模型分工是代码中的节点配置，费用与延迟效果需通过具体运行记录验证。

### 日志、追踪与评测采集

- **日志与追踪**：业务日志携带 `trace_id` 和 `thread_id`；本地追踪器记录 Trace、Span、Generation、模型用量与估算成本。估算结果取决于配置的价格表。

- **候选样本**：节点候选采集保存 `node`、北京时间 `ts`、`input` 和 `actual_output`，覆盖问题改写、规划、指标执行、报告写作审核及对话记忆保存等环节。`summarize_history` 采集的是追加回答、按需压缩前后的消息，不代表每轮都调用摘要模型；模型调用单独以 `conversation_summary` 记录。人工筛选与标注后可用于固定用例回归。

### 核心代码入口

| 模块 | 入口 |
| --- | --- |
| API 与依赖组装 | [api/main.py](api/main.py) |
| 问题改写与专家路由 | [app/routing/](app/routing/) |
| 对话历史压缩与近期消息保留 | [memory_manager.py](app/memory/memory_manager.py) |
| 分析规划、并行 Skill 与报告审核 | [crypto_analysis_parent_agent.py](app/crypto_analysis/crypto_analysis_parent_agent.py) |
| Skill 缓存、指标执行循环与内容生成 | [crypto_analysis_sub_agent.py](app/crypto_analysis/crypto_analysis_sub_agent.py) |
| 指标表格的确定性渲染 | [metric_renderer.py](app/crypto_analysis/metric_renderer.py) |
| 主题拆解、网页取证与报告生成 | [topic_research_agent.py](app/topic_research/topic_research_agent.py) |
| 检索与增量入库 | [app/rag/](app/rag/) |
| 模型创建与协议适配 | [app/llm/model.py](app/llm/model.py) |
| 日志、追踪与候选采集 | [app/core/](app/core/) |

## 快速启动

以下命令均在项目根目录执行。使用 Python 3.11，与 Docker 镜像保持一致；网页取证使用 Playwright Chromium。

### 1. 安装业务依赖

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
```

### 2. 配置环境变量

首次配置时复制示例，然后填写自己使用的服务密钥；已有 `.env` 时直接编辑原文件。

```bash
cp .env.example .env
```

[.env.example](.env.example) 只包含配置项和空密钥。当前主要配置如下：

| 环境变量 | 用途 | 使用条件 |
| --- | --- | --- |
| `SILICONFLOW_API_KEY` | 文档向量、检索重排 | 当前服务与知识入库需要 |
| `ZHIPUAI_4_API_KEY` | GLM 4 系列：问题改写、路由、加密分析规划与指标执行、主题研究、独立 RAG 回答 | 当前服务需要 |
| `BOCHA_API_KEY` | 网页搜索 | 通用主题研究需要 |
| `FRED_API_KEY` | 美联储政策及宏观序列 | 相关分析维度需要 |
| `CMC_PRO_API_KEY` | 稳定币市值与恐惧贪婪指数 | 相关分析维度需要 |
| `EODHD_API_KEY` | 通过 MCP 获取美元指数 | 流动性分析需要 |
| `CRYPTO_ETF_MCP_API_KEY` | 通过 MCP 获取现货 ETF 资金流 | ETF 分析需要 |
| `DASHSCOPE_API_KEY` | 加密分析报告写作与审核、图片描述与图像问答、Qwen3 Max 评测裁判 | 加密分析、相关图文处理或模型裁判评分需要 |
| `QWEN3_MAX_MODEL` | 评测裁判模型列表，默认 `["qwen3-max"]` | 按需覆盖默认模型列表 |
| `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL` | 实验记录 | 写入 Langfuse 的评测需要 |

Binance 行情工具使用公开行情接口。各数据源的访问权限、可用地区和接口额度取决于服务提供方。

模型、检索和路径参数见 [app/core/config.py](app/core/config.py)。操作系统环境变量优先于 `.env`；当前配置加载器还会检查项目上级目录的 `.env`，后加载的文件值会覆盖先加载的文件值。

### 3. 准备知识库

服务启动时会加载本地知识库，因此首次启动前需要准备文档并完成入库。推荐先将 UTF-8 Markdown 或 TXT 文档放入 `data/docs/`，再执行：

```bash
mkdir -p data/docs
python -m app.cli
```

入库会调用向量服务。默认索引位于 `data/faiss/bge-m3-1024/`，向量维度为 1024；切换模型或维度时应使用匹配的新索引。默认入库会同步删除已不在文档目录中的旧切片，需要保留它们时使用：

```bash
python -m app.cli --no-prune
```

### 4. 启动服务

```bash
python -m api.main
```

启动后访问：

- 工作台：<http://127.0.0.1:8003/>
- API 文档：<http://127.0.0.1:8003/docs>
- 健康检查：<http://127.0.0.1:8003/health>

也可以直接调用流式问答接口：

```bash
curl -N http://127.0.0.1:8003/api/ask \
  -H 'Content-Type: application/json' \
  -H 'X-Thread-Id: readme-demo' \
  -d '{"question":"分析 BTC 的趋势和衍生品指标，并说明信号分歧。"}'
```

响应包含 `progress`、`route`、`report_start`、`token`、`done` 或 `error` 等 SSE 事件。`report_start` 表示一次报告生成开始，客户端应清空旧稿，再接收后续 `token`；模型重试或审核重写时会重新发送该事件。连续追问时复用 `X-Thread-Id`。

任务中断后，使用原任务的 `X-Thread-Id` 调用续跑接口，无需重新提交问题：

```bash
curl -N -X POST http://127.0.0.1:8003/api/resume \
  -H 'X-Thread-Id: readme-demo'
```

续跑沿用同一会话的检查点与问答接口的 SSE 事件格式；该会话需要已有保存的任务状态。

### 5. Docker Compose

先完成 `.env` 配置并准备好 `data/` 下的索引，再执行：

```bash
docker compose up -d --build agent
docker compose ps
docker compose logs --tail=100 -f agent
```

Compose 挂载 `data/` 和 `logs/`，并将服务映射到宿主机 8003 端口。源码、提示词或 Skills 修改后需要重新构建镜像。当前 API 未提供用户认证，对外部署时需要设置相应的访问控制。

## 评测与验证

评测代码位于 [eval_agent/](eval_agent/)，独立于业务执行链路。固定用例由人工维护，直接复用生产函数或节点进行校验和评分。

当前以节点级和模块级评测为主：路由、规划和指标执行使用规则评分，Writer 对已保存报告评分，Reviewer 比较同一问题审核前后报告的忠实度与相关度变化；RAG 实际运行“检索 → 生成 → 评分”，属于该模块的端到端评测。现有结果不等同于整条 Agent 链路或本次对话记忆改动的端到端验收。

### 评测范围与入口

| 评测范围 | 评测方式 | 入口 |
| --- | --- | --- |
| 路由 | 对历史路由输出进行 RAGAS 规则评分，不重新调用模型 | [routing](eval_agent/routing/run_eval.py) |
| 分析规划 | 对历史规划输出进行资产、Skill 选择和刷新标志的规则评分 | [planning](eval_agent/planning/run_eval.py) |
| 指标计算（calculate_metrics） | 对历史指标输出进行工具名称、参数、证据关联和结果契约的规则评分 | [calculate_metrics](eval_agent/calculate_metrics/run_eval.py) |
| Writer | RAGAS 忠实度与回答相关性 | [writer](eval_agent/writer/run_eval.py) |
| Reviewer | 同一问题、相同证据下，对比审核前后报告的忠实度与回答相关性 | [reviewer](eval_agent/reviewer/run_eval.py) |
| RAG | 忠实度、回答相关性、上下文精确率与召回率 | [rag_retrieval](eval_agent/rag_retrieval/run_eval.py) |

### 运行方式

路由、分析规划和指标计算评测直接读取历史输出与预期标注，无需调用模型或重新取数。涉及模型调用的其他评测需要配置对应模型服务，写入 Langfuse 的评测还需要其连接配置。例如：

```bash
.venv-eval/bin/python -m eval_agent.routing.run_eval
.venv-eval/bin/python -m eval_agent.planning.run_eval
.venv-eval/bin/python -m eval_agent.calculate_metrics.run_eval
.venv-eval/bin/python -m eval_agent.writer.run_eval
.venv-eval/bin/python -m eval_agent.reviewer.run_eval
.venv-eval/bin/python -m eval_agent.rag_retrieval.run_eval
```

RAG 评测会实际执行检索、生成和裁判评分。评测结果应连同用例、模型、提示词版本与运行配置一起解读；本地用例校验不能代替真实服务评分。

### 评测结果

| 评测环节 | 用例数 | 本次结果 |
| --- | --- | --- |
| [路由](#路由) | 104 | 路由正确率 98.08%（102/104） |
| [分析规划](#分析规划) | 114 | 资产识别 100%，Skill 选择 78.07%，刷新判断 100% |
| [指标计算](#指标计算) | 116 | 四项规则指标均为 100% |
| [Writer 报告质量](#writer-报告质量) | 10 | 忠实度均分 0.7045（8 条有效评分），回答相关性均分 0.8817（10 条有效评分） |
| [Reviewer 审核前后对比](#reviewer-审核前后对比) | 28 组 | 忠实度 61.94% → 73.51%（28 组有效配对），回答相关性 82.30% → 85.55%（28 组有效配对） |
| [RAG 检索与回答](#rag-检索与回答) | 156 题（142 条有效评分明细） | 忠实度 87.85%，回答相关性 84.63%，上下文精确率 89.42%，上下文召回率 86.22% |

#### 路由

通过 RAGAS 自定义规则指标，将历史路由输出与独立预期标注逐条比对，保留重复记录，不重新调用路由模型。预期路由由 AI 标注助手依据路由规则复核，并保存判断依据。

| 指标 | 含义 | 正确率 |
| --- | --- | --- |
| `route_correct` | 历史路由输出与预期标注的一致率 | 98.08% |

#### 分析规划

通过 RAGAS 自定义规则指标，将历史规划输出与预期标注逐项比对，保留重复记录，不重新调用规划模型。预期值由 AI 标注助手依据规划规则和案例中的技能目录标注，并保存判断依据。

| 指标 | 含义 | 正确率 |
| --- | --- | --- |
| `target_asset_correct` | 目标资产识别正确率 | 100% |
| `skill_selection_correct` | Skill 选择正确率 | 78.07% |
| `refresh_correct` | 刷新标志判断正确率 | 100% |

#### 指标计算

通过 RAGAS 自定义规则指标，校验历史输出中的工具选择、调用参数、证据关联和指标结果契约，保留重复记录，不重新调用模型、执行工具或获取数据。预期契约由 AI 标注助手依据技能、工具及数据源定义独立标注，并保存判断依据。

其中 `indicator_results_valid` 检查结果字段、状态和元数据契约，不代表已经逐项验证所有指标公式与数值的正确性。

| 指标 | 含义 | 比例 |
| --- | --- | --- |
| `tool_selection_correct` | 工具选择正确率 | 100% |
| `tool_arguments_correct` | 工具调用参数正确率 | 100% |
| `evidence_valid` | 证据与关联校验通过率 | 100% |
| `indicator_results_valid` | 指标结果校验通过率 | 100% |

#### Writer 报告质量

使用 RAGAS 对已保存报告评分，不重新调用 Writer。未获得有效分数的案例不纳入对应指标均分。

| 指标 | 含义 | 均分 | 已评分案例数 |
| --- | --- | --- | --- |
| `faithfulness` | 回答忠实度 | 0.7045 | 10 |
| `answer_relevancy` | 回答相关性 | 0.8817 | 10 |

#### Reviewer 审核前后对比

对同一问题、相同证据下的历史报告进行审核前后配对评测，复用 Writer 的评分口径，不重新生成或审核报告。前后均分及差值只使用该指标两侧都评分成功的配对。

另有 5 组因裁判接口内容过滤（错误码 `1301`）未形成完整忠实度评分，作为缺失值排除，不记为零分。

| 指标 | 有效配对数 | 审核前均分 | 审核后均分 | 平均变化 |
| --- | --- | --- | --- | --- |
| `faithfulness` | 23 | 61.94% | 73.51% | +11.56 个百分点 |
| `answer_relevancy` | 28 | 82.30% | 85.55% | +4.26 个百分点 |

表中百分比用于展示 RAGAS 质量评分，变化按原始未舍入分数计算。

#### RAG 检索与回答

RAG 评测执行检索、回答生成和裁判评分，各指标均分仅基于有效评分汇总。

| 指标 | 含义 | 均分 | 有效评分数 |
| --- | --- | --- | --- |
| `faithfulness` | 回答忠实度 | 87.85% | 142 |
| `answer_relevancy` | 回答相关性 | 84.63% | 142 |
| `context_precision` | 上下文精确率 | 89.42% | 142 |
| `context_recall` | 上下文召回率 | 86.22% | 142 |

其余 14 题未见评分明细，原因未记录，暂不视为全部题目均已完成评分。

### 数据与结果管理

数据与结果的职责分布：

```text
业务节点候选采集 → 人工筛选与标注 → 固定用例 → 独立评测

eval_agent/candidates/       运行时采集的候选数据
eval_agent/<评测节点>/       固定用例与评测脚本
```

候选记录与评测结果可能包含完整输入输出，公开展示前需要人工筛选。质量、成本和延迟的效果对比应提供可复核的运行记录。

## 项目目录

```text
crypto-analysis-agent/
├── api/                         FastAPI 入口与请求模型
├── app/
│   ├── core/                    配置、日志、追踪、候选采集
│   ├── routing/                 问题改写、专家路由与 SSE 服务
│   ├── memory/                  对话历史压缩与近期消息保留
│   ├── crypto_analysis/         加密资产分析父图与 Skill 子图
│   │   ├── calculators/         指标计算与评分
│   │   ├── sources/             外部 API / MCP 数据源
│   │   └── tools/               业务工具与注册表
│   ├── topic_research/          主题研究 Agent 与状态
│   │   └── tools/               搜索与浏览器取证
│   ├── rag/                     文档处理、增量入库与混合检索
│   ├── llm/                     模型创建、调用与协议适配
│   ├── mcp_client/              MCP 连接与工具调用
│   └── cli.py                   知识库入库入口
├── prompts/                     业务提示词
├── skills/                      六个分析维度的 Skill 定义
├── static/                      Web 工作台与静态资源
├── data/                        本地文档、索引与会话数据
├── eval_agent/                  固定用例、候选、评测及结果
├── .env.example                 环境变量示例
├── Dockerfile
├── compose.yaml
└── requirements.txt
```
