"""
文档加载模块：支持 txt / pdf / docx / markdown / excel 多格式文档读取
PDF：文字提取 + 内嵌图片 CnOCR 并行，合并结果；纯扫描件自动整页 OCR 兜底
"""
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from core.log_config import logger

OCR_MIN_TEXT_LENGTH = 50

_ocr_engine = None


def _get_ocr():
    """懒加载 CnOCR 单例"""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from cnocr import CnOcr
            _ocr_engine = CnOcr()
            logger.info("CnOCR 初始化完成（中英文）")
        except ImportError:
            logger.error("CnOCR 未安装，请执行: pip install cnocr")
            raise
    return _ocr_engine


def _ocr_image(image) -> str:
    """对单张图片执行 OCR，返回识别文本（接受 PIL Image 或 numpy array）"""
    try:
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        ocr = _get_ocr()
        results = ocr.ocr(image)
        return "\n".join(r["text"] for r in results if r.get("score", 0) > 0.3)
    except Exception:
        return ""


def _load_pdf_with_ocr(path: str):
    """
    PDF 加载：文字提取 + 内嵌图片 OCR 并行，合并结果
    """
    try:
        import fitz
    except ImportError:
        logger.warning("pymupdf 未安装，回退 PyPDFLoader")
        return PyPDFLoader(path).load()

    all_docs = []
    total_text = ""

    try:
        pdf_doc = fitz.open(path)

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            page_text = page.get_text().strip()
            total_text += page_text

            image_texts = _ocr_page_images(page, pdf_doc)

            combined = page_text
            if image_texts:
                combined += "\n" + "\n".join(image_texts)

            if combined.strip():
                all_docs.append(Document(
                    page_content=combined,
                    metadata={"source": path, "page": page_num + 1}
                ))

        pdf_doc.close()

        if len(total_text) < OCR_MIN_TEXT_LENGTH:
            logger.info(f"PDF 全文仅 {len(total_text)} 字符，启动整页 OCR 兜底")
            ocr_docs = _ocr_pdf_pages_full(path)
            if ocr_docs:
                all_docs = ocr_docs

        logger.info(f"PDF 加载完成: {path}，共 {len(all_docs)} 页")

    except Exception as e:
        logger.error(f"PDF 加载失败: {e}，回退 PyPDFLoader")
        return PyPDFLoader(path).load()

    return all_docs if all_docs else PyPDFLoader(path).load()


def _ocr_page_images(page, pdf_doc) -> list:
    """提取 PDF 页面内嵌图片，CnOCR 识别"""
    image_texts = []
    try:
        from PIL import Image
        import io

        image_list = page.get_images(full=True)
        if not image_list:
            return image_texts

        for img_info in image_list:
            try:
                xref = img_info[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))

                if img.width < 50 or img.height < 50:
                    continue

                text = _ocr_image(img)
                if text:
                    image_texts.append(text)
            except Exception:
                continue
    except Exception:
        pass

    return image_texts


def _ocr_pdf_pages_full(path: str) -> list:
    """整页 OCR 兜底：纯扫描件 PDF 逐页渲染后 OCR"""
    try:
        import fitz
        from PIL import Image
        import io

        ocr_docs = []
        pdf_doc = fitz.open(path)

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = _ocr_image(img)
            if text:
                ocr_docs.append(Document(
                    page_content=text,
                    metadata={"source": path, "page": page_num + 1, "ocr": "full_page"}
                ))

        pdf_doc.close()
        logger.info(f"整页 OCR 完成: {path}，共 {len(ocr_docs)} 页")
    except Exception as e:
        logger.error(f"整页 OCR 失败: {e}")
        return []

    return ocr_docs


def _load_txt_with_fallback(path: str):
    """尝试多种编码加载 txt 文件（UTF-8 → GBK → latin-1）"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]
    for enc in encodings:
        try:
            loader = TextLoader(path, encoding=enc)
            docs = loader.load()
            if enc != "utf-8" and enc != "utf-8-sig":
                logger.info(f"txt 文件使用 {enc} 编码加载: {path}")
            return docs
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 最终回退
    logger.warning(f"所有编码尝试失败，使用 latin-1 回退: {path}")
    return TextLoader(path, encoding="latin-1").load()


def load_documents(file_paths: list[str]):
    """
    批量加载文档，增加异常捕获与日志
    :param file_paths: 文件路径列表
    :return: 解析后的Document列表
    """
    all_docs = []
    for path in file_paths:
        try:
            if not os.path.exists(path):
                logger.warning(f"文件不存在，跳过: {path}")
                continue

            ext = os.path.splitext(path)[1].lower()
            loader = None

            if ext == ".txt":
                docs = _load_txt_with_fallback(path)
            elif ext == ".pdf":
                docs = _load_pdf_with_ocr(path)
            elif ext == ".docx":
                docs = Docx2txtLoader(path).load()
            elif ext == ".md":
                docs = UnstructuredMarkdownLoader(path).load()
            elif ext in (".xlsx", ".xls"):
                docs = UnstructuredExcelLoader(path, mode="elements").load()
            else:
                logger.warning(f"不支持文件格式 {ext}，跳过 {path}")
                continue

            all_docs.extend(docs)
            logger.info(f"成功加载文件: {path}，段落数: {len(docs)}")

        except Exception as e:
            logger.error(f"文件 {path} 加载失败，异常: {str(e)}")
            continue
    return all_docs