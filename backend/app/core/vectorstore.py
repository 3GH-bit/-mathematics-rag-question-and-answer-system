# app/core/vectorstore.py
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer, CrossEncoder

# 设置hf镜像，避免下载超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class VectorStore:
    def __init__(self, collection_name="math_rag"):
        self.collection_name = collection_name
        # 本地持久化向量库，保存到 ./qdrant_db 文件夹
        self.client = QdrantClient(path="./qdrant_db")
        self.embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        # 重排模型：bge-reranker-base
        self.rerank_model = CrossEncoder("BAAI/bge-reranker-base")
        self.vector_size = 512

        # 判断集合是否已经存在，不存在才创建
        exist_collections = [col.name for col in self.client.get_collections().collections]
        if self.collection_name not in exist_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            print(f"创建新集合 {self.collection_name}")
        else:
            print(f"集合 {self.collection_name} 已存在，复用向量库")

    def add_chunks(self, chunks: list[str]):
        """批量向量化并写入向量库"""
        valid_texts = []
        for text in chunks:
            if isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
        print(f"有效文本数量：{len(valid_texts)}")
        if len(valid_texts) == 0:
            return

        embeddings = self.embedding_model.encode(valid_texts)
        points = []
        for idx, (text, vec) in enumerate(zip(valid_texts, embeddings)):
            point = PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload={"text": text}
            )
            points.append(point)
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"成功入库 {len(points)} 个有效块")

    def search(self, query: str, top_k=3, recall_top=8):
        """
        向量检索 + Rerank重排
        recall_top：先粗召回多条候选，再用rerank筛选top_k条最相关的
        """
        query_vec = self.embedding_model.encode(query)
        # 粗召回更多候选
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec.tolist(),
            limit=recall_top
        )
        candidate_texts = [hit.payload["text"] for hit in hits.points]
        if len(candidate_texts) == 0:
            return []

        # Rerank：问题和候选文本两两打分
        pairs = [[query, text] for text in candidate_texts]
        scores = self.rerank_model.predict(pairs)

        # 按分数降序排序
        scored_texts = sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)
        # 取top_k
        final_texts = [item[0] for item in scored_texts[:top_k]]
        return final_texts
