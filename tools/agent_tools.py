"""
Agent 自定义工具集合：知识库检索、联网搜索、文档保存
"""
from langchain.tools import tool
from tavily import TavilyClient
from settings import TAVILY_API_KEY, SAVE_SUMMARY_PATH, UPLOAD_TOP_K_TEMP
from retriever import multi_hybrid_retrieve
from utils import format_retrieve_docs
from log_config import logger
import session_store

# 初始化联网搜索客户端
tavily = TavilyClient(api_key=TAVILY_API_KEY)


def _deduplicate_docs(docs: list) -> list:
    """基于 page_content 去重，保留首次出现的文档"""
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


@tool
def search_knowledge_base(query: str) -> str:
    """
    企业本地知识库检索工具，读取PDF/DOCX/TXT内部文档
    :param query: 用户检索关键词/问题
    """
    # 1. 全局 FAISS 持久知识库检索
    faiss_docs = multi_hybrid_retrieve(query)

    # 2. 当前会话临时 Chroma 检索（用户上传文档）
    temp_docs = []
    try:
        temp_chroma = session_store.get_current_chroma()
        if temp_chroma is not None:
            temp_docs = temp_chroma.similarity_search(query, k=UPLOAD_TOP_K_TEMP)
    except Exception as e:
        logger.error(f"临时文档 Chroma 检索异常（不影响全局检索）: {e}")

    # 3. 合并去重：临时文档排前（用户刚上传更相关），全局 FAISS 排后
    all_docs = temp_docs + (faiss_docs if faiss_docs else [])
    if not all_docs:
        return ""
    all_docs = _deduplicate_docs(all_docs)
    return format_retrieve_docs(all_docs)

@tool
def search_online(query: str) -> str:
    """
    全网实时资讯检索，用于本地无数据的实时政策、日期、赛事
    :param query: 联网搜索关键词
    """
    resp = tavily.search(query=query)
    res_text = ""
    for item in resp["results"]:
        res_text += f"【标题】{item['title']}\n【内容】{item['content']}\n\n"
    return res_text

@tool
def save_summary_to_txt(summary_text: str) -> str:
    """
    将总结内容导出到本地txt文件
    :param summary_text: 待保存的总结文本
    """
    with open(SAVE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return f"执行完成：总结已保存至 {SAVE_SUMMARY_PATH}"

# 对外导出工具列表
tool_list = [search_knowledge_base, search_online, save_summary_to_txt]