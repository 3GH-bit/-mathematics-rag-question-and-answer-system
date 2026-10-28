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
    chunks = chunks[:10]

    # 打印类型，检查是否有 None
    for i, c in enumerate(chunks):
        print(f"[{i}] type={type(c)}, len={len(c)}, stripped={bool(c.strip())}")
        if c is None:
            print("!!!发现None!!!")