from app.core.vectorstore import VectorStore
from openai import OpenAI

class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        # ====================== 请修改这里 ======================
        self.dashscope_api_key = "sk-ws-H.PEREEXY.dG7e.MEUCIQCwh8JH-xx4M2UIiZdebx9voni3wHehEL-tLnR_6vr-wAIgRQ02mjBbO4Uybvo8X0Ee5eKdq5EghNPk2ZZ6xTV7lHw"
        self.model_name = "qwen-plus"
        # =======================================================
        self.client = OpenAI(
            api_key=self.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def build_prompt(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(contexts)
        prompt = f"""你是一名初中数学老师，专门讲解初中几何。请严格依据下面参考资料回答用户问题。
参考资料：
{context_text}

用户问题：{question}
要求：
1. 如果参考资料信息不足，如实告知，不要编造知识点。
2. 语言通俗易懂，符合初中生认知，讲解条理清晰。
3. 几何术语规范，可以适当举例说明。
"""
        return prompt

    def llm_infer(self, prompt: str):
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"通义千问API调用失败：{str(e)}"

    def query(self, question: str):
        # 向量检索
        contexts = self.vector_store.search(question, n_results=3)
        prompt = self.build_prompt(question, contexts)
        answer = self.llm_infer(prompt)
        return answer, prompt, contexts
