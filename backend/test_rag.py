import os
os.environ["PYTHONUTF8"] = "1"

from openai import OpenAI
from app.core.vectorstore import VectorStore
from app.core.document_loader import load_multimodal_pdf, split_text

# 通义千问 LLM client
client = OpenAI(
    api_key="sk-ws-H.PEREEXY.dG7e.MEUCIQCwh8JH-xx4M2UIiZdebx9voni3wHehEL-tLnR_6vr-wAIgRQ02mjBbO4Uybvo8X0Ee5eKdq5EghNPk2ZZ6xTV7lHw",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def rag_query(vs: VectorStore, question: str):
    # 文本检索+BGE-Rerank召回相关上下文（包含OCR+VL图形描述）
    contexts = vs.search(question, top_k=3, recall_top=8)
    print("===== 检索上下文（含图形描述） =====")
    for idx, c in enumerate(contexts):
        print(f"【片段{idx+1}】{c}")

    prompt = """你是一名初中数学老师，专门讲解初中几何。参考资料里面包含题目文字、OCR识别内容和AI对几何图形的描述。
严格依据下面参考资料回答用户问题。
参考资料：
{contexts}
用户问题：{question}
要求：
1. 如果参考资料信息不足，如实告知，不要编造知识点。
2. 语言通俗易懂，符合初中生认知，讲解条理清晰。
3. 几何术语规范，可以适当举例说明。
4. 回答末尾标注引用片段编号。
"""
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role":"user", "content": prompt.format(contexts=contexts, question=question)}]
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    vs = VectorStore()
    load_data = True

    if load_data:
        # 读取PDF，多模态解析：OCR + Qwen-VL看图描述
        raw_content = load_multimodal_pdf("./math_geo_test.pdf")
        chunks = split_text(raw_content)
        vs.add_chunks(chunks)
        print("✅ PDF多模态解析入库完成！")

    ans = rag_query(vs, "什么是手拉手模型，请结合图形特点说明")
    print("===== 大模型回答 =====")
    print(ans)
vs.client.close()
