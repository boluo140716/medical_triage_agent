"""
图片分析模块：EasyOCR 文字提取 + Qwen-VL 视觉描述
"""
import io
import os
import base64
import time
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

_ocr_reader = None
_ocr_lock = threading.Lock()
_ocr_failed = False  # 标记 OCR 是否初始化失败，避免反复重试


def _get_ocr():
    global _ocr_reader, _ocr_failed
    if _ocr_reader is None and not _ocr_failed:
        with _ocr_lock:
            if _ocr_reader is None and not _ocr_failed:
                for attempt in range(3):
                    try:
                        import easyocr
                        _ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
                        logger.info("EasyOCR 模型加载完成")
                        return _ocr_reader
                    except PermissionError:
                        logger.warning(f"EasyOCR 下载被文件锁阻止（第{attempt+1}次），5秒后重试...")
                        time.sleep(5)
                    except Exception as e:
                        logger.warning(f"EasyOCR 初始化失败（第{attempt+1}次）: {e}")
                        if attempt < 2:
                            time.sleep(3)
                # 3次都失败，标记为不可用
                _ocr_failed = True
                logger.warning("EasyOCR 初始化失败（已重试3次），图片文字提取功能不可用，视觉描述仍正常")
    return _ocr_reader


def preload_ocr():
    """启动时预加载 OCR 模型，避免首次请求卡顿"""
    logger.info("预加载 EasyOCR 模型...")
    _get_ocr()

def _extract_text(image_bytes: bytes) -> str:
    """用 EasyOCR 提取图片中的文字"""
    try:
        reader = _get_ocr()
        image = Image.open(io.BytesIO(image_bytes))
        # 大图缩放：长边 > 1024px 时等比缩小，防止内存爆炸
        max_size = 1024
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            logger.info(f"图片缩放: {w}x{h} → {image.size[0]}x{image.size[1]}")
        arr = list(image.convert("RGB").getdata())
        h, w = image.size[1], image.size[0]
        arr = [arr[i * w:(i + 1) * w] for i in range(h)]
        results = reader.readtext(arr)
        texts = [r[1] for r in results if r[2] > 0.3]
        return "\n".join(texts) if texts else ""
    except Exception as e:
        logger.warning(f"EasyOCR 文字提取失败: {e}")
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