import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.parsers.pdf_parser import PdfParser
from app.core.chunking import split_text
from app.core.embeddings import EmbeddingModel

if __name__ == "__main__":
    parser = PdfParser()
    with open("./samples/test.pdf", "rb") as f:
        data = f.read()
    full_text = parser.parse(data)
    chunks = split_text(full_text)

    embed_model = EmbeddingModel()
    # 只取前3块测试向量生成，避免一次性加载太多
    test_chunks = chunks[:3]
    embeddings = embed_model.embed_documents(test_chunks)

    print(f"测试块数量：{len(test_chunks)}")
    print(f"向量维度：{len(embeddings[0])}")
    print("向量前10个值：", embeddings[0][:10])

    # 测试query向量
    query_emb = embed_model.embed_query("什么是手拉手模型")
    print(f"\n查询向量维度：{len(query_emb)}")
    print("查询向量前10个值：", query_emb[:10])