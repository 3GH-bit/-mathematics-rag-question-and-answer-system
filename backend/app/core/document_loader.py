import os
# 👉 设置EasyOCR模型缓存路径到E盘，改成你想要的文件夹
os.environ["EASYOCR_MODULE_PATH"] = r"E:\Projects\ai-RAG-chat\backend\.easyocr_models"

import fitz
import easyocr
from typing import List
import socket

socket.setdefaulttimeout(300)

# 初始化reader，会自动去上面这个路径找/下载模型
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)


def ocr_image(img):
    result = reader.readtext(img, detail=0)
    return "\n".join(result)


def load_multimodal_pdf(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        page_text = text

        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            ocr_txt = ocr_image(pix.tobytes("png"))
            page_text += f"\n【图片OCR识别】：\n{ocr_txt}"

        if page_text.strip():
            chunks.append(page_text.strip())
    return chunks


def split_text(text_list: List[str], chunk_size=500, overlap=80) -> List[str]:
    res = []
    for txt in text_list:
        start = 0
        while start < len(txt):
            piece = txt[start:start + chunk_size]
            res.append(piece)
            start += chunk_size - overlap
    return res
