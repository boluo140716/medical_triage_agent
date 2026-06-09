# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Enterprise knowledge base Q&A agent built on LangGraph 1.x + LangChain 1.x. Supports local document retrieval (txt/pdf/docx/md/xlsx), online web search fallback via Tavily, and summary export. Two entry points: interactive console (`main.py`) and Gradio web UI (`web_start.py`). Uses Ollama for local LLM inference and embeddings.

## Prerequisites

- **Ollama** must be running locally with models pulled:
  ```bash
  ollama pull qwen2:7b
  ollama pull all-minilm
  ```
- **`.env` file** with `TAVILY_API_KEY=<your-key>` (required for online search fallback). Get a free key at [tavily.com](https://tavily.com).

## Commands

```bash
# Install dependencies (use python -m pip, NOT bare pip — Python/pip paths differ on this machine)
python -m pip install -r requirements.txt

# Run the agent (interactive console)
python main.py

# Run the Gradio web UI (http://0.0.0.0:7860)
python web_start.py

# Run any single module
python -m agent.graph_builder
```

## Architecture

```
main.py                  # Console entry point — imports global agent_app, runs interactive loop
web_start.py             # Gradio web UI entry point — creates its own agent instance, serves on :7860
settings.py              # All config: paths, model params, chunk sizes, retrieval params. Also calls set_llm_cache(InMemoryCache()) at import time.
prompts.py               # System prompt (SYS_PROMPT) + judgment prompt template (JUDGE_PROMPT_TPL) — decoupled from code for easy tuning
log_config.py            # Global logger ("KB-Agent"), writes to ./logs/run.log + stdout
utils.py                 # format_retrieve_docs helper

document/
  loader.py              # Multi-format loader: .txt (TextLoader), .pdf (PyPDFLoader), .docx (Docx2txtLoader), .md (UnstructuredMarkdownLoader), .xlsx/.xls (UnstructuredExcelLoader)
  splitter.py            # Two splitters: abstract_splitter (80-char, for FAISS coarse) + detail_splitter (500-char, for fine retrieval)
  vector_store.py        # FAISS IndexFlatL2 + index↔fulltext mappings. Auto-builds on first run, caches to disk (first_faiss.index + index_mapping.json). init_faiss_store() runs at module import time — importing this module triggers index build/load.

retriever.py             # Two-tier hybrid RAG: (1) FAISS abstract search → (2) Chroma vector + BM25 keyword ensemble on detail chunks. Includes time.sleep(0.15) rate limiting between tiers.

tools/agent_tools.py     # Three @tool functions: search_knowledge_base, search_online (Tavily), save_summary_to_txt

agent/
  state.py               # AgentState TypedDict with Annotated[Sequence[BaseMessage], operator.add] for message accumulation
  nodes.py               # LangGraph nodes: agent_think_node (LLM + tool binding), tool_execute_node (runs tool), judge_check_node (validates result), final_answer_node (synthesizes natural-language answer from retrieved content)
  routes.py              # Conditional edges: tool_route_func (has tool_calls? → execute : END), judge_route_func (ng + retries < 3? → retry think : final_answer)
  graph_builder.py       # StateGraph assembly: think → (tool? → execute → judge → (ng? → retry think : final_answer) : END). Exports global agent_app singleton.
```

## Key design decisions

- **LangChain version**: LangChain **1.x** (not 0.x). Core APIs live in `langchain_core`, not `langchain` directly — use `from langchain_core.globals import set_llm_cache`, `from langchain_core.caches import InMemoryCache`, etc. Some compat imports come from `langchain_classic` (e.g. `EnsembleRetriever`).
- **Two-tier RAG**: FAISS searches document abstracts first (cheap coarse filter), then Chroma+BM25 EnsembleRetriever searches detail chunks within matched documents. `retriever.py` is the only module the agent tools call — they never touch `document/` directly.
- **LLM + Embeddings**: Both run locally via Ollama. LLM = `qwen2:7b`, embeddings = `all-minilm`. Configured in `settings.py`.
- **Judgment loop with retry cap**: After every tool execution, `judge_check_node` validates result relevance. If it returns "ng" and fewer than 3 retries have occurred, the graph loops back to `agent_think_node`. On "ok" or after 3 retries, flows to `final_answer_node` which synthesizes a natural-language answer from the retrieved content. This prevents both hallucinated answers and infinite loops.
- **FAISS cache**: `document/vector_store.py` auto-scans the project root for supported files on first run and builds the FAISS index. Subsequent runs load from disk (`first_faiss.index` + `index_mapping.json`). Delete both files to force a rebuild when documents change. Note: `init_faiss_store()` runs at module import time — there's no explicit init call needed.
- **Embedding cache**: `settings.py` calls `set_llm_cache(InMemoryCache())` at import time, deduplicating identical embedding requests and reducing Ollama API calls ~90%.
- **Entry point difference**: `main.py` imports the global `agent_app` singleton from `graph_builder.py`. `web_start.py` calls `build_agent_graph()` directly, creating a separate instance — both are valid patterns.
- **`.env` for secrets**: `TAVILY_API_KEY` is loaded from `.env` via `python-dotenv`. The file is `.gitignore`'d.
- **Rate limiting**: `retriever.py` sleeps 150ms between FAISS and Chroma/BM25 tiers to avoid overwhelming the local Ollama embedding service.
