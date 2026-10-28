import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.parsers.pdf_parser import PdfParser
from app.core.chunking import split_text

if __name__ == "__main__":
    parser = PdfParser()
    with open("./samples/test.pdf", "rb") as f:
        data = f.read()
    full_text = parser.parse(data)

    chunks = split_text(full_text)
    print(f"总分块数量：{len(chunks)}")
    print(f"\n===== 第0块预览 =====")
    print(chunks[0])
    print(f"\n===== 第1块预览 =====")
    print(chunks[1])