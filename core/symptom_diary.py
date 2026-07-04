"""
症状日记模块：按会话记录和查询用户症状变化
数据存储在 temp_summary/<session_id>/symptom_diary.json
"""
import json
import os
from datetime import datetime
from core.log_config import logger
from core import session_store


def _get_diary_path() -> str:
    session_id = session_store.get_current_session_id()
    if not session_id:
        return ""
    from core.settings import TEMP_SUMMARY_DIR
    diary_dir = os.path.join(TEMP_SUMMARY_DIR, session_id)
    os.makedirs(diary_dir, exist_ok=True)
    return os.path.join(diary_dir, "symptom_diary.json")


def record_symptom(content: str) -> str:
    """记录一条症状"""
    path = _get_diary_path()
    if not path:
        return "会话未初始化，记录失败。"

    entries = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)

    entry = {
        "time": datetime.now().strftime("%m-%d %H:%M"),
        "content": content,
    }
    entries.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    logger.info(f"症状日记已记录（共{len(entries)}条）")
    return f"✅ 已记录：{content}（当前共{len(entries)}条症状记录）"


def query_symptoms() -> str:
    """查询当前会话全部症状历史"""
    path = _get_diary_path()
    if not path or not os.path.exists(path):
        return "📋 暂无症状记录。"

    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not entries:
        return "📋 暂无症状记录。"

    lines = [f"📋 症状日记（共{len(entries)}条）\n"]
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. [{e['time']}] {e['content']}")

    # 如果超过1条，追加趋势提示
    if len(entries) >= 2:
        lines.append(f"\n📊 已记录{len(entries)}次，请对比前后变化趋势。")

    return "\n".join(lines)