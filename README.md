# 🏢 企业知识库 RAG 智能问答 Agent

企业内部知识库智能问答系统，支持 PDF/DOCX/TXT/MD/XLSX 文档检索、联网搜索、文档摘要导出。基于 LangGraph ReAct 循环，提供 Gradio Web、FastAPI REST、控制台三种交互入口。

## ✨ 功能特性

- **双层检索 + Rerank 精排** — FAISS 粗筛 → 精细分片 → 集合检索（Chroma 向量 + BM25 关键词）→ DashScope 重排序
- **临时文档即时索引** — 用户上传文件自动分片、嵌入，会话隔离，关闭即销毁
- **联网搜索双引擎** — Tavily 优先，DuckDuckGo 自动回退，确保网络检索可用
- **对话摘要导出** — 保存/导出/另存为关键词门禁，导出为文本文件并嵌入回答卡片
- **多轮对话上下文记忆** — MemorySaver 检查点 + thread_id 会话隔离，支持指代消解
- **多标签页隔离** — ContextVar 管理 Chroma 实例、摘要目录、保存门禁，互不干扰
- **Docker 一键部署** — 单容器双进程（Gradio 7862 + FastAPI 7863），`docker compose up -d`

## 🧱 架构概览

```
┌───────────────────────────────────────────────────────────┐
│  入口层                                                    │
│  Gradio Web UI (:7862)  │  FastAPI REST (:7863)  │  控制台  │
├───────────────────────────────────────────────────────────┤
│  Agent 层 (LangGraph ReAct)                               │
│  agent_think_node ⇄ tool_execute_node                     │
│  MemorySaver 检查点 · MAX_TOOL_ROUNDS=5                   │
├───────────────────────────────────────────────────────────┤
│  工具层                                                    │
│  search_knowledge_base │ search_online │ save_summary_to_txt│
├───────────────────────────────────────────────────────────┤
│  检索链路                                                  │
│  FAISS 粗筛 → 精细分片(500字) → Chroma+BM25 → Rerank 精排  │
├───────────────────────────────────────────────────────────┤
│  模型服务                                                  │
│  LLM: DeepSeek-chat (api.deepseek.com)                    │
│  Embedding: 阿里百炼 text-embedding-v3                     │
│  Rerank: DashScope gte-rerank                             │
│  联网: Tavily / DuckDuckGo                                │
└───────────────────────────────────────────────────────────┘
```

## 📁 目录结构

```
enterprise_kb_agent/
├── main.py                  # 控制台入口（asyncio REPL 流式输出）
├── web_start.py             # Gradio Web 启动器 (:7862)
├── run_api.py               # FastAPI 启动器 (:7863)
├── docker-compose.yml       # Docker 编排（双进程容器）
│
├── core/                    # 核心配置与工具
│   ├── settings.py          # 全局配置（模型、检索、分片参数）
│   ├── prompts.py           # 系统提示词 + 保存关键词白名单
│   ├── log_config.py        # 日志配置（轮转：10MB × 5）
│   ├── session_store.py     # ContextVar 会话存储
│   └── utils.py             # 通用工具函数
│
├── agent/                   # LangGraph Agent
│   ├── state.py             # AgentState（消息列表累加）
│   ├── graph_builder.py     # 图构建（MemorySaver 检查点）
│   ├── nodes.py             # ReAct 思考/执行节点
│   ├── retriever.py         # 双层检索聚合 + LRU 缓存 + 上下文扩展
│   └── routes.py            # 条件路由（工具调用 → 执行 → 结束）
│
├── tools/                   # LangChain 工具
│   └── agent_tools.py       # search_knowledge_base / search_online / save_summary_to_txt
│
├── document/                # 文档处理管道
│   ├── loader.py            # 多格式加载（txt/pdf/docx/md/xlsx）+ 编码回退
│   ├── splitter.py          # 文本分片（摘要 80 字 / 正文 500 字）
│   ├── vector_store.py      # FAISS 索引构建/检索 + DashScope 嵌入
│   └── reranker.py          # gte-rerank 重排序（指数退避重试）
│
├── api/                     # FastAPI REST
│   ├── models.py            # Pydantic 请求/响应模型
│   ├── dependency.py        # 会话注入与清理
│   └── routers/
│       ├── chat.py          # SSE 流式对话 + 非流式回退 + 文件下载
│       └── upload.py        # 文件上传 + 会话清理
│
├── web/                     # Gradio Web UI
│   ├── layout.py            # UI 布局 + 事件绑定（Gradio Blocks）
│   ├── chat.py              # 流式对话生成器（含摘要卡片）
│   ├── upload.py            # 文件上传处理
│   └── session_utils.py     # 会话工具（UUID、摘要目录、答案提取）
│
├── docker/
│   ├── Dockerfile           # python:3.13-slim 多阶段镜像
│   └── supervisord.conf     # Supervisor 双进程管理
│
├── kb_docs/                 # 持久化知识库文档目录
├── temp_summary/            # 摘要临时存储（按 session_id 隔离）
├── logs/                    # 轮转日志
├── requirements.txt         # Python 依赖
└── .env                     # 环境变量（API 密钥等）
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

### 启动方式（4 选 1）

```bash
# 方式 1：Gradio Web 界面 → 浏览器访问 http://localhost:7862
python web_start.py

# 方式 2：FastAPI REST API → Swagger 文档 http://localhost:7863/docs
python run_api.py

# 方式 3：控制台交互（asyncio REPL 流式输出）
python main.py

#方式 4：docker命令启动（docker-compose up -d --build）
拉取成功后，浏览器访问页面本地地址即可

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
FAISS 摘要级向量检索（Top-3 文档粗筛）
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
| Agent 框架 | LangGraph 1.x（ReAct 循环 + MemorySaver 检查点） |
| LLM | DeepSeek-chat（OpenAI 兼容 API） |
| 向量嵌入 | 阿里百炼 DashScope text-embedding-v3 |
| 重排序 | DashScope gte-rerank |
| 向量索引 | FAISS（粗筛）+ Chroma（精细临时索引） |
| 关键词检索 | BM25（rank-bm25） |
| 联网搜索 | Tavily + DuckDuckGo 双引擎回退 |
| Web UI | Gradio 6.x（异步流式） |
| REST API | FastAPI + Uvicorn（SSE 流式） |
| 依赖管理 | pip + Docker |
| 容器化 | Docker Compose + Supervisor 双进程 |

## 📄 License

MIT
