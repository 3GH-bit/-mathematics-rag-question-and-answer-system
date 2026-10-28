from io import BytesIO
from pypdf import PdfReader
from app.parsers.base import BaseParser

class PdfParser(BaseParser):
    def parse(self, file_bytes: bytes) -> str:
        """
        接收pdf二进制字节，返回提取后的完整文本
        """
        reader = PdfReader(BytesIO(file_bytes))
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        return full_text