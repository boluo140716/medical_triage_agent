# 🏥 医疗分诊决策 Agent

基于 LangGraph ReAct 循环的智能医疗分诊助手，支持症状评估、科室推荐、药物查询、就医指引。提供豆包风格 Web UI 和 FastAPI REST 两种交互入口。

## ✨ 功能特性

- **多轮追问式分诊** — 不直接给结论，先追问关键信息（持续时间、疼痛程度、伴随症状）再综合判断
- **三层紧急分级** — 🔴紧急(立即急诊) / 🟡建议尽快就诊 / 🟢可居家观察
- **危险信号自动排查** — 逐条排查脑卒中FAST、心梗、脑膜炎、过敏性休克等致命信号
- **HyDE 查询改写** — 将用户口语症状描述自动改写为医学风格检索文本，提升召回率
- **双路检索 + Rerank 精排** — FAISS 粗筛 → 精细分片 → Chroma+BM25 → DashScope 重排序
- **药物安全查询** — 检索药品适应症、禁忌、注意事项、相互作用
- **PDF 内嵌图片 OCR** — EasyOCR 识别 PDF 内嵌图片中的文字
- **自动增量索引** — 检测 kb_docs 文件变更，自动重建 FAISS 索引
- **联网搜索** — Tavily + DuckDuckGo 双引擎，搜索医院科室、药物信息
- **多轮对话上下文记忆** — MemorySaver 检查点 + thread_id 会话隔离

## 🧱 架构概览

```
┌───────────────────────────────────────────────────────────┐
│  入口层                                                    │
│  豆包风格 Web UI (:7863)            │  FastAPI REST (:7863) │
├───────────────────────────────────────────────────────────┤
│  Agent 层 (LangGraph ReAct)                               │
│  agent_think_node ⇄ tool_execute_node                     │
│  MemorySaver 检查点 · MAX_TOOL_ROUNDS=5                   │
├───────────────────────────────────────────────────────────┤
│  工具层                                                    │
│  search_knowledge_base │ search_online │ save_summary     │
│  assess_symptom_urgency │ check_drug_safety               │
├───────────────────────────────────────────────────────────┤
│  检索链路                                                  │
│  HyDE 改写 → FAISS 粗筛 → 精细分片(500字) → Chroma+BM25   │
│  → Rerank 精排 → 双路合并（改写+原始）                      │
├───────────────────────────────────────────────────────────┤
│  文档处理                                                  │
│  多格式加载 → 文字提取 + EasyOCR 图片识别 → 智能分片        │
├───────────────────────────────────────────────────────────┤
│  模型服务                                                  │
│  LLM: DeepSeek-chat │ Embedding: 阿里百炼                  │
│  Rerank: DashScope gte-rerank │ OCR: EasyOCR 中英双语      │
│  联网: Tavily / DuckDuckGo                                │
└───────────────────────────────────────────────────────────┘
```

## 📁 目录结构

```
enterprise_kb_agent/
├── main.py                  # 控制台入口（asyncio REPL 流式输出）
├── run_api.py               # FastAPI 启动器 + Web UI 托管 (:7863)
│
├── static/                  # 纯 HTML 豆包风格前端
│   ├── index.html           # Vue3 CDN 单页面（左侧历史会话 + 右侧对话区）
│   └── css/
│       └── style.css        # 医疗蓝绿配色样式
│
├── core/                    # 核心配置与工具
│   ├── settings.py          # 全局配置（模型、检索、分片参数）
│   ├── prompts.py           # 医疗分诊系统提示词
│   ├── log_config.py        # 日志配置
│   ├── session_store.py     # ContextVar 会话存储
│   ├── session_utils.py     # 会话工具
│   └── utils.py             # 通用工具函数
│
├── agent/                   # LangGraph Agent
│   ├── state.py             # AgentState
│   ├── graph_builder.py     # 图构建（MemorySaver 检查点）
│   ├── nodes.py             # ReAct 思考/执行节点
│   ├── retriever.py         # 双层检索 + HyDE 改写 + LRU 缓存
│   └── routes.py            # 条件路由
│
├── tools/                   # LangChain 工具
│   ├── agent_tools.py       # 通用工具 + 工具列表注册
│   └── medical_tools.py     # 医疗分诊专用工具
│
├── document/                # 文档处理管道
│   ├── loader.py            # 多格式加载 + PDF OCR
│   ├── splitter.py          # 文本分片
│   ├── vector_store.py      # FAISS 索引构建/自动增量重建
│   └── reranker.py          # gte-rerank 重排序
│
├── api/                     # FastAPI REST
│   ├── __init__.py          # 应用入口 + StaticFiles 托管前端
│   ├── models.py            # Pydantic 模型
│   ├── dependency.py        # 会话注入与清理
│   └── routers/
│       ├── chat.py          # SSE 流式对话
│       └── upload.py        # 文件上传
│
├── kb_docs/                 # 知识库文档目录
│   └── medical/             # 医疗知识库
│       ├── 症状对照库.txt
│       ├── 科室导航.txt
│       ├── 药品常识.txt
│       └── 急救指南.txt
├── temp_summary/            # 摘要临时存储
├── logs/                    # 轮转日志
├── requirements.txt         # Python 依赖
└── .env                     # 环境变量
```

## 🚀 快速开始

### 环境要求

- Python 3.12+（本地运行）/ Docker（无需 Python 环境）
- 知识库文档放入 `kb_docs/` 目录（首次启动自动构建 FAISS 索引）

### 环境变量

创建 `.env` 文件：

```bash
# 必填：LLM
DEEPSEEK_API_KEY=sk-your-deepseek-key

# 必填：Embedding 向量化
BAILIAN_API_KEY=sk-your-bailian-key

# 可选：联网搜索（缺失时自动降级为 DuckDuckGo）
TAVILY_API_KEY=tvly-your-tavily-key

# 可选：自定义 API 地址（默认值如下）
# DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动方式

```bash
# 唯一启动入口：FastAPI + Web UI → 浏览器访问 http://localhost:7863
python run_api.py

# 控制台交互（asyncio REPL 流式输出）
python main.py
```

## 📡 REST API 接口

Base URL: `http://localhost:7863`

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 → `{"status":"ok"}` |
| `POST` | `/api/chat/stream` | SSE 流式对话 → `data: {"type":"token","content":"..."}` |
| `POST` | `/api/chat` | 非流式对话 → `{"answer":"...","session_id":"..."}` |
| `GET` | `/api/download/{session_id}` | 下载会话摘要文件 |
| `POST` | `/api/upload` | 上传临时文档（multipart） |
| `DELETE` | `/api/upload` | 清空会话上传文件 |

### 请求示例

```bash
# 流式对话
curl -X POST http://localhost:7863/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"主动离职需提前多少天提交申请？","session_id":null}'

# 非流式对话
curl -X POST http://localhost:7863/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"年假怎么算？","session_id":null}'
```

## 🐳 Docker 部署

```bash
docker compose up -d
```

- Gradio Web UI → `http://localhost:7862`
- FastAPI REST → `http://localhost:7863`
- `kb_docs/` 挂载为只读，`temp_summary/` 持久化
- Supervisor 管理 Gradio + FastAPI 双进程，异常自动重启

## 🔧 核心配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL_NAME` | `deepseek-chat` | 大语言模型 |
| `LLM_TEMPERATURE` | `0.3` | 生成温度 |
| `EMBED_MODEL_NAME` | `text-embedding-v3` | 向量嵌入模型（阿里百炼） |
| `TOP_K_FIRST_FAISS` | `3` | FAISS 粗筛返回数 |
| `TOP_K_SUB_RETRIEVE` | `3` | 混合检索返回数 |
| `TOP_K_RERANK` | `3` | Rerank 后保留数 |
| `MAX_TOOL_ROUNDS` | `5` | ReAct 最大工具调用轮次 |
| `MAX_CONTEXT_CHARS` | `20000` | 上下文截断上限 |
| `DETAIL_CHUNK_SIZE` | `500` | 正文分片大小（字符） |
| `UPLOAD_MAX_FILE_SIZE_MB` | `10` | 上传单文件上限 |
| `UPLOAD_MAX_FILE_COUNT` | `5` | 单次上传数量上限 |

## 🔍 检索流程

```
用户提问
  │
  ▼
HyDE 改写：LLM 生成与文档风格一致的检索文本
  │
  ├─→ 改写文本 FAISS 摘要级向量检索
  └─→ 原始提问 FAISS 摘要级向量检索
  │
  ▼
双路结果去重合并（Top-3 文档粗筛）
  │
  ▼
命中文档 → 精细分片（500 字，100 字重叠）
  │
  ▼
混合检索：Chroma 向量 + BM25 关键词 → Ensemble 加权（0.6/0.4）
  │
  ▼
DashScope gte-rerank 重排序（Top-3）
  │
  ▼
上下文扩展：附加相邻分片，避免文档碎片化
  │
  ▼
结果注入 LLM 上下文 → 生成回答
```

## 🛠 技术栈

| 层级 | 技术选型 |
|------|----------|
| Agent 框架 | LangGraph 1.x（ReAct 循环 + MemorySaver 检查点）|
| LLM | DeepSeek-chat |
| 向量嵌入 | 阿里百炼 DashScope text-embedding-v3 |
| 重排序 | DashScope gte-rerank |
| 查询改写 | HyDE（Hypothetical Document Embeddings）|
| OCR | EasyOCR 中英双语 |
| 联网搜索 | Tavily + DuckDuckGo 双引擎回退 |
| Web UI | 纯 HTML + Vue3 CDN + CSS（豆包风格）|
| REST API | FastAPI + Uvicorn（SSE 流式 + StaticFiles）|

## ⚠️ 免责声明

本系统仅提供医学知识参考和就医指引，**不能替代专业医生诊断**。如有身体不适，请及时就医。

## 📄 License

MIT