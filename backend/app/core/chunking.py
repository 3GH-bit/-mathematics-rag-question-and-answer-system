def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """简单文本分块，过滤空白块"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunk_stripped = chunk.strip()
        # 只有去除空白后不为空，才加入列表
        if chunk_stripped:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks