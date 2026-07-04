"""
FAISS向量库管理模块：索引构建、持久化、向量检索
"""
import os
import json
import faiss
import numpy as np
from langchain_community.embeddings import DashScopeEmbeddings
from core.settings import (
    FAISS_INDEX_PATH,
    MAPPING_JSON_PATH,
    KB_DOCS_DIR,
    TEMP_SUMMARY_DIR,
    EMBED_MODEL_NAME,
    TOP_K_FIRST_FAISS,
    BAILIAN_API_KEY,
)
from core.log_config import logger

# 初始化向量化模型（阿里百炼 DashScope）
embeddings = DashScopeEmbeddings(
    model=EMBED_MODEL_NAME,
    dashscope_api_key=BAILIAN_API_KEY,
)

# 全局FAISS缓存变量
faiss_index = None
index2abs = {}
index2full = {}

def _get_kb_file_list():
    """获取 kb_docs 目录下所有支持格式的文件名列表（用于比较缓存是否过期）"""
    os.makedirs(KB_DOCS_DIR, exist_ok=True)
    supported_ext = (".txt", ".pdf", ".docx")
    files = []
    for filename in os.listdir(KB_DOCS_DIR):
        full_path = os.path.join(KB_DOCS_DIR, filename)
        if os.path.isdir(full_path):
            continue
        if filename.lower().endswith(supported_ext):
            files.append(filename)
    return sorted(files)


def _build_faiss_index(file_paths):
    """全量构建 FAISS 索引并持久化"""
    global faiss_index, index2abs, index2full

    from document.loader import load_documents
    from document.splitter import abstract_splitter

    all_docs = load_documents(file_paths)
    abstract_list = []
    doc_full_map = {}

    for doc in all_docs:
        full_text = doc.page_content
        abs_chunks = abstract_splitter.split_text(full_text)
        if not abs_chunks:
            continue
        abs_text = abs_chunks[0]
        abstract_list.append(abs_text)
        doc_full_map[abs_text] = full_text

    embed_arr = embeddings.embed_documents(abstract_list)
    embed_np = np.array(embed_arr, dtype=np.float32)
    dim = embed_np.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embed_np)

    for idx, abs_text in enumerate(abstract_list):
        index2abs[idx] = abs_text
        index2full[idx] = doc_full_map[abs_text]

    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(MAPPING_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "index2abs": index2abs,
            "index2full": index2full,
            "source_files": [os.path.basename(p) for p in file_paths]
        }, f, ensure_ascii=False)
    logger.info("FAISS索引与映射文件保存完成")


def init_faiss_store():
    """初始化/加载FAISS索引，启动时自动检测新文件并重建"""
    global faiss_index, index2abs, index2full

    current_files = _get_kb_file_list()

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(MAPPING_JSON_PATH):
        with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        cached_files = sorted(map_data.get("source_files", []))

        if current_files == cached_files:
            logger.info("FAISS缓存有效，直接加载本地索引")
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            index2abs = {int(k): v for k, v in map_data["index2abs"].items()}
            index2full = {int(k): v for k, v in map_data["index2full"].items()}
            return
        else:
            new_files = set(current_files) - set(cached_files)
            removed_files = set(cached_files) - set(current_files)
            logger.info(f"kb_docs 文件变更，自动重建索引。新增: {new_files}, 移除: {removed_files}")

    logger.info("开始全量构建FAISS索引")
    file_paths = [os.path.join(KB_DOCS_DIR, f) for f in current_files]

    if not file_paths:
        logger.warning(f"知识库目录 {KB_DOCS_DIR} 中无文档，FAISS 索引为空")
        return

    _build_faiss_index(file_paths)

def faiss_search(query: str) -> list[int]:
    """FAISS一级向量检索，返回有效文档下标"""
    if faiss_index is None:
        logger.error("FAISS 索引未初始化，无法检索")
        return []
    q_vec = np.array([embeddings.embed_query(query)], dtype=np.float32)
    _, idxs = faiss_index.search(q_vec, TOP_K_FIRST_FAISS)
    hit_list = idxs[0].tolist()
    valid_index = [i for i in hit_list if i != -1]
    return valid_index

# 模块加载时自动初始化向量库
init_faiss_store()