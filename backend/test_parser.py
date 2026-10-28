from app.parsers.pdf_parser import PdfParser

if __name__ == "__main__":
    parser = PdfParser()
    with open("./samples/test.pdf", "rb") as f:
        data = f.read()
    text = parser.parse(data)
    print("提取文本长度：", len(text))
    print(text[:500])