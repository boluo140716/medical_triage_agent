'''
医疗分诊专用工具：症状评估，医院搜索，药物查询
'''
from langchain.tools import tool
from agent.retriever import multi_hybrid_retrieve
from core.utils import format_retrieve_docs
from core.log_config import logger
from core.symptom_diary import record_symptom, query_symptoms

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
            return "未检索到相关症状信息，请基于通用医学知识谨慎判断。"
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
            return "未检索到该药品信息，请基于通用药学知识谨慎回答，并建议用户咨询医生或药师。"
        return format_retrieve_docs(docs)
    except Exception as e:
        logger.error(f"药品查询失败: {e}")
        return f"药品查询异常: {e}"

@tool
def check_drug_interaction(drugs: str) -> str:
    """
    查询多种药物同时服用是否存在相互作用风险。
    :param drugs: 用逗号分隔的药品名称，如"阿莫西林,布洛芬"
    """
    try:
        drug_list = [d.strip() for d in drugs.replace("，", ",").split(",") if d.strip()]
        if len(drug_list) < 2:
            return "至少需要两种药物才能检查相互作用，请同时列出您正在服用的所有药物。"
        query = " ".join(drug_list) + " 药物相互作用 禁忌"
        docs = multi_hybrid_retrieve(query)
        if not docs:
            return f"未检索到{'、'.join(drug_list)}的相互作用信息，请咨询医生或药师。"
        return format_retrieve_docs(docs)
    except Exception as e:
        logger.error(f"药物相互作用查询失败: {e}")
        return f"药物相互作用查询异常: {e}"


@tool
def interpret_lab_report(lab_query: str) -> str:
    """
    解读化验单指标，检索正常参考范围和异常临床意义。
    :param lab_query: 化验指标名称或数据，如"谷丙转氨酶 120"、"血常规 白细胞偏高"
    """
    try:
        docs = multi_hybrid_retrieve(lab_query)
        if not docs:
            return "未检索到该化验指标信息，请基于通用医学知识谨慎解读。"
        return format_retrieve_docs(docs)
    except Exception as e:
        logger.error(f"化验单解读失败: {e}")
        return f"化验单解读异常: {e}"


@tool
def symptom_diary(query: str) -> str:
    """
    记录和查询用户症状变化历史，用于追踪健康趋势。
    自动判断操作类型：query 包含"记录""新增""记一下" → 记录症状；否则 → 查询历史。
    :param query: 症状描述，如"记录：头痛 7分 右侧太阳穴"或"查询头痛历史"
    """
    try:
        record_kw = ["记录", "新增", "添加", "记一下", "帮我记", "记录下来"]
        if any(kw in query for kw in record_kw):
            result = record_symptom(query)
            # 静默模式：记录时只返回简短提示，避免 LLM 在回答中泄露记录信息
            return f"[后台已记录] {result}"
        return query_symptoms()
    except Exception as e:
        logger.error(f"症状日记操作失败: {e}")
        return f"症状日记操作异常: {e}"


# 医疗工具列表
medical_tool_list = [assess_symptom_urgency, check_drug_safety, check_drug_interaction, interpret_lab_report, symptom_diary]