"""
图片分析模块：CnOCR 文字提取 + Qwen-VL 视觉描述
"""
import io
import os
import base64
import threading
from PIL import Image
from core.log_config import logger

try:
    from dashscope import MultiModalConversation
    import dashscope
    _qwen_available = True
except ImportError:
    _qwen_available = False
    logger.warning("dashscope 未安装，Qwen-VL 视觉描述不可用")

_ocr = None
_ocr_lock = threading.Lock()
_ocr_failed = False


def _get_ocr():
    """懒加载 CnOCR（预加载线程负责初始化，请求路径直接返回）"""
    global _ocr, _ocr_failed
    if _ocr is None and not _ocr_failed:
        with _ocr_lock:
            if _ocr is None and not _ocr_failed:
                try:
                    from cnocr import CnOcr
                    _ocr = CnOcr()
                    logger.info("CnOCR 模型加载完成")
                except Exception as e:
                    _ocr_failed = True
                    logger.warning(f"CnOCR 初始化失败: {e}，图片文字提取功能不可用，视觉描述仍正常")
    return _ocr


def preload_ocr():
    """启动时预加载 OCR 模型"""
    logger.info("预加载 CnOCR 模型...")
    _get_ocr()

def _extract_text(image_bytes: bytes) -> str:
    """用 CnOCR 提取图片中的文字"""
    try:
        ocr = _get_ocr()
        if ocr is None:
            return ""
        image = Image.open(io.BytesIO(image_bytes))
        # 大图缩放：长边 > 1024px 时等比缩小
        max_size = 1024
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            logger.info(f"图片缩放: {w}x{h} → {image.size[0]}x{image.size[1]}")
        results = ocr.ocr(image)
        texts = [r["text"] for r in results if r.get("score", 0) > 0.3]
        return "\n".join(texts) if texts else ""
    except Exception as e:
        logger.warning(f"CnOCR 文字提取失败: {e}")
        return ""
    
def _visual_description(image_bytes: bytes) -> str:
    """用 Qwen-VL 描述图片视觉内容"""
    if not _qwen_available:
        return ""
    try:
        api_key = os.getenv("BAILIAN_API_KEY", "")
        if not api_key:
            return ""

        dashscope.api_key = api_key
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{img_b64}"},
                {"text": "请客观描述这张图片中的医学相关内容：如有皮肤症状请描述颜色、形态、范围；如有化验单请列出所有项目和数值；如有药品请列出名称和剂量。只描述，不下诊断。"}
            ]
        }]
        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages,
        )
        # 防御性解析：兼容多种响应格式
        output = response.output
        if output and output.choices:
            choice = output.choices[0]
            msg = choice.message
            content = msg.content
            # 格式1: [{text: "..."}]（标准）
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict):
                    return content[0].get("text", "")
                return str(content[0])
            # 格式2: "..."（纯文本）
            if isinstance(content, str):
                return content
        return ""
    except Exception as e:
        logger.warning(f"Qwen-VL 视觉描述失败: {e}")
        return ""


def analyze_image(image_bytes: bytes) -> str:
    """
    分析医学图片：OCR 提取文字 + 多模态描述视觉内容
    返回合并后的文本描述
    """
    ocr_text = _extract_text(image_bytes)
    visual = _visual_description(image_bytes)

    parts = []
    if ocr_text:
        parts.append(f"【图片文字提取】\n{ocr_text}")
    if visual:
        parts.append(f"【图片视觉描述】\n{visual}")

    if not parts:
        return "图片分析失败，未能提取到有效信息。"

    return "\n\n".join(parts)