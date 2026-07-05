"""
FastAPI 应用实例 + CORS 中间件 + 静态文件托管
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="医疗分诊决策 Agent",
    description="支持流式 SSE 对话、症状评估、科室推荐、药物查询的 RESTful API",
    version="2.0.0",
)

# CORS：允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件目录（CSS/JS/图片等）
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    """首页：返回前端页面"""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "前端页面未部署，请使用 API 接口"}


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "service": "medical-triage-agent"}

# 延迟导入路由，避免循环依赖
from api.routers import chat, upload

app.include_router(chat.router)
app.include_router(upload.router)

# 启动时预加载 CnOCR 模型（后台线程，不阻塞启动）
# 设置 EASYOCR_SKIP_PRELOAD=1 可跳过预加载（如开发时频繁 reload）
import threading as _threading
if not os.getenv("EASYOCR_SKIP_PRELOAD"):
    from tools.image_analyzer import preload_ocr
    _threading.Thread(target=preload_ocr, daemon=True).start()
else:
    from core.log_config import logger
    logger.info("EASYOCR_SKIP_PRELOAD=1，跳过预加载，首次图片上传时按需加载")