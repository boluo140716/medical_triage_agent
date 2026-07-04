'''
医疗分诊专用工具：症状评估，医院搜索，药物查询
'''
from langchain.tools import tool
from agent.retriever import multi_hybrid_retrieve
from core.utils import format_retrieve_docs
from core.log_config import logger

@tool
def assess_symptom_urgency(symptom_query: str) -> str:
    """
    根据症状描述，从知识库中检索相关疾病信息、危险信号和紧急程度判断依据。
    调用此工具后再由 LLM 综合判断紧急程度。
    :param symptom_query: 症状关键词，如"头痛 持续3天 恶心"
    """
    try:
        docs = multi_hybrid_retrieve(symptom_query)
        if not docs:
            return "未在知识库中检索到相关症状信息，请基于通用医学知识谨慎判断。"
        return format_retrieve_docs(docs)
    except Exception as e:
        logger.error(f"症状评估工具调用失败：{e}")
        return f"症状评估检索异常: {e}"
    

@tool
def check_drug_safety(drug_query: str) -> str:
    """
    查询药品的适应症、禁忌、注意事项和相互作用。
    :param drug_query: 药品名称或关键词，如"布洛芬 胃溃疡"
    """
    try:
        docs = multi_hybrid_retrieve(drug_query)
        if not docs:
            return "未在知识库中检索到该药品信息，请基于通用药学知识谨慎回答，并建议用户咨询医生或药师。"
        return format_retrieve_docs(docs)
    except Exception as e:
        logger.error(f"药品查询失败: {e}")
        return f"药品查询异常: {e}"

# 医疗工具列表
medical_tool_list = [assess_symptom_urgency, check_drug_safety]