from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str):
        emb = self.model.encode(query, normalize_embeddings=True)
        return emb.tolist()