from abc import ABC, abstractmethod

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes) -> str:
        """
        解析文件二进制内容，返回提取后的文本
        """
        pass