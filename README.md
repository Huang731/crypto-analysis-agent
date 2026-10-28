# Crypto Analysis Agent

基于 LangChain、LangGraph 的智能多 Agent 研究分析平台，将实时数据、网页原文与本地知识库组织为分析证据，生成可查看来源的研究报告。

支持两类任务：**加密资产分析**覆盖趋势、衍生品、ETF、情绪、流动性与美联储政策；**通用主题研究**支持问题拆解、并行搜索、网页取证与报告审核。Web 工作台提供执行进度、流式输出和中断续跑。

**技术栈：** Python · FastAPI · LangChain · LangGraph · Pydantic · MCP · FAISS · SQLite · SSE

[工程亮点](#工程亮点) · [功能演示](#功能演示) · [系统架构](#系统架构) · [模型配置](#模型配置) · [评测结果](#评测结果) · [项目目录](#项目目录)

## 工程亮点

| 设计 | 实现与作用 |
| --- | --- |
| **规划与并行执行** | 根据问题选择 Skills，通过 LangGraph `Send` 并行执行子图，使用显式 State 与 Reducer 汇总结果，再进入报告生成和审核。[代码](app/crypto_analysis/crypto_analysis_parent_agent.py) |
| **模型与计算分工** | Skills 定义场景、工具和指标清单，Tools 取数，Python 完成指标计算与表格渲染，模型负责规划、工具选择与解释。[Skill 定义](skills/) · [指标渲染](app/crypto_analysis/metric_renderer.py) |
| **混合检索与增量入库** | 中文 BM25 + FAISS 向量召回，经加权 RRF 融合与重排；按文件哈希同步新增、修改和删除。知识库支持分析解读，实时行情由工具获取。[代码](app/rag/) |
| **会话记忆与结果复用** | SQLite Checkpointer 按 `thread_id` 保存状态；历史摘要与近期消息用于问题改写，Skill 结果按会话和资产范围复用，显式刷新时跳过缓存。[记忆管理](app/memory/) · [Skill 缓存](app/crypto_analysis/skill_memory.py) |
| **流式交互与中断续跑** | SSE 展示执行阶段与报告正文；通过 `POST /api/resume` 恢复同一会话的检查点。重写报告时发送 `report_start`，前端清空旧稿后接收新内容。[代码](app/routing/service.py) |
| **追踪与评测闭环** | 按 Trace / Span 记录节点、模型用量与估算成本；采集节点输入输出，经筛选标注形成固定用例，结合规则、RAGAS 与浏览器端到端评测定位问题。[追踪](app/core/tracing.py) · [评测](eval_agent/) |

## 功能演示

### 加密资产分析示例

以下为一次 ETH 分析的实际运行截图，展示问题输入、执行阶段、报告正文及数据来源。点击图片可查看原图。

| 提问与执行进度 | 分析报告 | 数据来源与指标 |
| --- | --- | --- |
| <a href="docs/images/crypto-analysis-workbench.png"><img src="docs/images/crypto-analysis-workbench.png" alt="ETH 分析工作台：问题输入、执行进度与报告结论" width="240"></a> | <a href="docs/images/crypto-analysis-report.png"><img src="docs/images/crypto-analysis-report.png" alt="ETH 分析报告：资金流、货币政策、流动性、情绪与趋势" width="240"></a> | <a href="docs/images/crypto-analysis-sources.png"><img src="docs/images/crypto-analysis-sources.png" alt="ETH 分析报告：策略说明与 ETF 资金流指标明细" width="240"></a> |

### 通用主题研究示例

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

## 模型配置

业务节点与评测裁判通过百炼调用千问系列，按职责配置模型列表：

| 分类 | 模型系列 | 使用场景 | 环境变量 |
| --- | --- | --- | --- |
| `quick_think_llm` | Flash | 路由、问题改写、历史摘要、分析规划、主题拆解与汇总、指标工具执行、技能检索问题生成 | `QUICK_THINK_MODELS` |
| `deep_think_llm` | Plus | 主题研究、报告写作与审核、RAG 回答与多查询改写 | `DEEP_THINK_MODELS` |
| 独立评测模型 | Max | Writer / Reviewer 报告评分、RAG 质量评分 | `QWEN3_MAX_MODEL` |

三类入口分别为 `make_quick_think_llm()`、`make_deep_think_llm()` 和 `make_qwen3_max_llm()`，实现见 [模型工厂](app/llm/model.py)。快思考模型默认关闭思考；深思考模型保留模型默认配置，研究工具调用和多查询改写节点显式关闭思考。

复制 [.env.example](.env.example) 为 `.env` 后填写密钥：`DASHSCOPE_API_KEY` 用于百炼聊天、评测与视觉模型，`SILICONFLOW_API_KEY` 用于向量和重排服务。模型列表使用 JSON 数组，按顺序调用，仅在免费额度耗尽时切换下一项；默认列表及其他运行参数见 [配置文件](app/core/config.py)。

## 评测结果

以下为历史运行结果；业务模型统一为千问后尚未重新评测。

### 节点与模块评测

| 环节 | 样本范围 | 结果 | 入口 |
| --- | --- | --- | --- |
| 路由 | 104 条 | 正确率 **98.08%**（102/104） | [routing](eval_agent/routing/run_eval.py) |
| 分析规划 | 114 条 | 资产识别 100%，Skill 选择 78.07%，刷新判断 100% | [planning](eval_agent/planning/run_eval.py) |
| 指标执行契约 | 116 条 | 工具名称、参数、证据关联、结果契约四项均为 100% | [calculate_metrics](eval_agent/calculate_metrics/run_eval.py) |
| Writer | 10 题 | 忠实度 0.7045，回答相关性 0.8817 | [writer](eval_agent/writer/run_eval.py) |
| Reviewer | 28 组历史报告 | 审核前后忠实度 **61.94% → 73.51%**，回答相关性 82.30% → 85.55% | [reviewer](eval_agent/reviewer/run_eval.py) |
| RAG | 156 题，142 条有效评分明细 | 忠实度 **87.85%**，回答相关性 84.63%，上下文精确率 89.42%，召回率 86.22% | [rag_retrieval](eval_agent/rag_retrieval/run_eval.py) |

规则评测检查历史输出，RAGAS 评估报告质量；指标契约通过率不代表公式与数值正确率，忠实度不代表事实准确率。

<details>
<summary>评测方式与有效样本数</summary>

- 路由、规划与指标执行读取历史输出，不重新调用模型或取数。
- Writer / Reviewer 对已保存报告评分；Writer 忠实度为 8 条有效评分、相关性为 10 条，Reviewer 忠实度为 23 组有效配对、相关性为 28 组。缺失评分不记为零分。
- RAG 实际执行检索、生成与评分；156 题中有 142 条有效评分明细，其余 14 题未见评分明细。

</details>

### 端到端评测

通过真实浏览器，在同一会话中连续提交 10 题，包含 4 组重复提问，检查任务完成、响应耗时及不同问法下的一致性。

**四轮均 10/10 完成、零重试；第四轮平均耗时 101.1s，报告综合评分 45.5/50。** 第四轮报告列出的 8 项数值或方向核对通过，BTC 技术面、ETF 资金流与宏观三组对照基本一致；ETH 衍生品仍出现同组数据下多头拥挤判断相反的问题。

**评分维度：** 每项满分 10 分，综合得分为五项之和，满分 50 分。以下为各轮报告对页面回答质量与响应表现的评分，完成率不等同于事实准确率。

| **维度** | **第四轮** | **第三轮** | **第二轮** | **第一轮** |
| :--- | :--- | :--- | :--- | :--- |
| 任务完成度 | 9.5 | 9.0 | 8.5 | 8.5 |
| 上下文一致性 | 9.0 | 8.0 | 7.5 | 7.0 |
| 事实依据 | 9.0 | 8.0 | 7.5 | 7.5 |
| 响应稳定性 | 9.5 | 9.5 | 7.5 | 9.0 |
| 多轮交互质量 | 8.5 | 8.0 | 8.0 | 8.0 |
| **综合** | **45.5** | 42.5 | 39.0 | 40.0 |

报告：[第一轮（10/21）](eval_agent/end_to_end/crypto_10q_evaluation_report_run1.md) · [第二轮（10/21）](eval_agent/end_to_end/crypto_10q_evaluation_report_run2.md) · [第三轮（10/28 10:44）](eval_agent/end_to_end/crypto_10q_evaluation_report_run3.md) · [第四轮（10/28 13:19）](eval_agent/end_to_end/crypto_10q_evaluation_report_run4.md)。评分依据页面可见回答与响应时长；市场数据与服务版本在四轮间有变化，不据此归因某项改动的收益，也未据此验证多轮指代能力。

<details>
<summary>展开四轮完整对比与结论</summary>

| 指标 | 第一轮 10/21 17:07 | 第二轮 10/21 19:16 | 第三轮 10/28 10:44 | 第四轮 10/28 13:19 |
|---|---|---|---|---|
| 完成率 | 10/10 | 10/10 | 10/10 | **10/10** |
| 重试次数 | 0 | 0 | 0 | **0** |
| 平均耗时 | 118.7s | 143.0s | 103.0s | **101.1s** |
| 耗时极差 | 125.4s | 207.0s | 77.8s | **约 49s** |
| 总耗时 | 19 分 49 秒 | 23 分 49 秒 | 17 分 10 秒 | **16 分 51 秒** |
| A 组（BTC 技术） | 🟢 | 🟢 | 🟢 | 数值与方向一致 |
| B 组（ETF 资金流） | 🟡 漂移 | 🟡 漂移 | 🟢 对齐 | 数值与方向一致 |
| C 组（ETH 衍生品） | 🔴 矛盾 | 🟢 对齐 | 🟡 轻度措辞差 | **Q3 / Q7 数据一致，拥挤判断相反** |
| D 组（宏观） | 🔴 相反 | 🔴 相反 | 🟢 数据统一，仅剩加权分歧 | 数值与方向一致 |
| 单条回答的数值或口径差异 | 无 | Q7：2.2% vs 2.0% | Q3：2355.36 vs 2350.36，未说明价格时点差异 | 未见此前 ETH 价格时点差异 |
| 缺少解释的内部评分 | 无 | 无 | 有（Q4、Q10） | 报告未记录此问题 |
| 报告综合评分 | 40.0 | 39.0 | 42.5 | **45.5** |

第四轮耗时范围为 75.6–124.8s；报告记载极差 49.3s，按逐题显示值复算为 49.2s，表中统一记为约 49s。

**第四轮结论：** 本轮记录中的平均耗时更低，A / B / D 三组对照的数值与结论方向基本一致，ETH 价格也统一为 2350.36 USDT。主要遗留问题是 Q3 / Q7 对同一组资金费率和持仓量给出不同的拥挤判断，BTC 持仓量部分缺少起止绝对值，且这组题目尚未验证多轮指代能力；8 项核对通过仅说明报告所列数值或方向可复核，不代表全部回答均已验证正确。

</details>

## 项目目录

业务编排位于 `app/`，提示词与 Skill 定义分别位于 `prompts/`、`skills/`，独立评测位于 `eval_agent/`。模型与运行参数见 [配置文件](app/core/config.py)。

<details>
<summary>展开目录结构</summary>

```text
crypto-analysis-agent/
├── api/                         FastAPI 入口与请求模型
├── app/
│   ├── core/                    配置、日志、追踪、执行预算与候选采集
│   ├── routing/                 问题改写、专家路由与 SSE 服务
│   ├── memory/                  对话历史压缩与近期消息保留
│   ├── crypto_analysis/         加密资产分析父图与 Skill 子图
│   │   ├── calculators/         指标计算
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
│   ├── routing/                 路由规则评测
│   ├── planning/                分析规划规则评测
│   ├── calculate_metrics/       指标执行契约评测
│   ├── writer/                  报告质量评分
│   ├── reviewer/                审核前后配对评分
│   ├── rag_retrieval/           检索与回答评测
│   ├── end_to_end/              四轮 10 题浏览器端到端评测报告
│   ├── candidates/              运行时采集的候选样本
│   ├── outputs/                 评测输出与结果
│   └── requirements.txt         独立评测环境依赖
├── tests/                       节点与数据源验证脚本
├── docs/                        项目文档资源
│   └── images/                  README 演示图片
├── .env.example                 环境变量示例
├── pyproject.toml               Ruff 检查与格式化配置
├── Dockerfile
├── compose.yaml
└── requirements.txt
```

</details>
