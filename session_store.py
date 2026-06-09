"""
会话级临时 Chroma 存储桥梁（ContextVar 实现，多浏览器标签页隔离）

使用 contextvars.ContextVar 而非全局单例，确保每个 invoke() 执行链路独立，
不同浏览器标签页的临时文档互不污染。
"""
import contextvars

# 当前会话的 Chroma 实例（内存模式，不落盘）
_current_chroma = contextvars.ContextVar('current_chroma', default=None)

# 当前会话已上传文件信息: [{"name": "xxx.pdf", "summary": "第一章..."}, ...]
_current_file_info = contextvars.ContextVar('current_file_info', default=None)


def set_current(chroma, file_info):
    """注入当前会话的 Chroma 和文件信息到执行上下文"""
    _current_chroma.set(chroma)
    _current_file_info.set(file_info)


def get_current_chroma():
    """获取当前会话的 Chroma 实例，无临时文档时返回 None"""
    return _current_chroma.get()


def get_current_file_info():
    """获取当前会话已上传文件信息列表"""
    return _current_file_info.get()


def clear_current():
    """清除当前会话上下文（每次 invoke 结束后调用）"""
    _current_chroma.set(None)
    _current_file_info.set(None)
