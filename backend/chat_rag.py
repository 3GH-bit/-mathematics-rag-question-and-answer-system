import os
# ===================== 必须放在最顶部：HF镜像配置 =====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 模型已下载到本地缓存，强制离线加载，避免 hf-mirror 偶发 SSL 握手失败
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# =====================================================================

import time
import re
import json
import gradio as gr
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from openai import OpenAI, RateLimitError
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from FlagEmbedding import FlagReranker, FlagModel
from rank_bm25 import BM25Okapi
from geometry_render import render_figure, parse_draw_json
from math_unicode import latex_to_unicode

# -------------------------- 配置区 --------------------------
# 多知识库分组：key是前端下拉显示的名字，value是对应PDF路径
PDF_GROUPS = {
    "全部知识库": None,
    "全等三角形专题": r"E:\Projects\ai-RAG-chat\backend\初中数学常用知识及结论.pdf",
}
RERANK_THRESHOLD = -2
# DeepSeek 配置：Key 从环境变量或 backend/.env 读取，不要写死在代码里
def _load_dotenv(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com"
)
MODEL_NAME = "deepseek-chat"

# 几何图输出目录：Gradio 必须拿到真实文件路径才能显示图片
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 初始化组件
COLLECTION_NAME = "math_rag_multi"
qdrant = QdrantClient(path="./qdrant_db")
embedder = FlagModel("BAAI/bge-small-zh-v1.5", use_fp16=False)
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False)

# 全局BM25索引（混合检索用）
bm25 = None
all_chunks_text = []
all_chunks_meta = []
# -----------------------------------------------------------

# 上下标字符映射
SUB_MAP = {"₀":"<sub>0</sub>","₁":"<sub>1</sub>","₂":"<sub>2</sub>","₃":"<sub>3</sub>","₄":"<sub>4</sub>","₅":"<sub>5</sub>","₆":"<sub>6</sub>","₇":"<sub>7</sub>","₈":"<sub>8</sub>","₉":"<sub>9</sub>"}
SUP_MAP = {"⁰":"<sup>0</sup>","¹":"<sup>1</sup>","²":"<sup>2</sup>","³":"<sup>3</sup>","⁴":"<sup>4</sup>","⁵":"<sup>5</sup>","⁶":"<sup>6</sup>","⁷":"<sup>7</sup>","⁸":"<sup>8</sup>","⁹":"<sup>9</sup>","ⁿ":"<sup>n</sup>"}

def format_math_text(text):
    text = re.sub(r"[●▪]", "- ", text)
    text = re.sub(r"\\四周|\\角|防守方向", "", text)
    for k,v in SUB_MAP.items(): text = text.replace(k,v)
    for k,v in SUP_MAP.items(): text = text.replace(k,v)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def load_multimodal_pdf(pdf_path):
    texts = []
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            texts.append(txt)
    return texts

def split_text(text_list, chunk_size=500, overlap=80):
    chunks = []
    full_text = "\n".join(text_list)
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# 初始化向量库 + BM25索引
def init_vector_store():
    global bm25, all_chunks_text, all_chunks_meta
    colls = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in colls:
        print("📖 开始构建多知识库向量库...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        point_id = 0
        all_chunks_text = []
        all_chunks_meta = []
        for group_name, pdf_path in PDF_GROUPS.items():
            if group_name == "全部知识库" or not pdf_path:
                continue
            print(f"  正在解析：{group_name}")
            text_list = load_multimodal_pdf(pdf_path)
            text_list = [format_math_text(t) for t in text_list]
            clean_text = [t.strip() for t in text_list if len(t.strip()) >= 15]
            chunks = split_text(clean_text, chunk_size=500, overlap=80)
            points = []
            for chunk in chunks:
                vec = embedder.encode(chunk)
                points.append(PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={"text": chunk, "group": group_name}
                ))
                all_chunks_text.append(chunk)
                all_chunks_meta.append({"group": group_name, "id": point_id})
                point_id += 1
            qdrant.upsert(COLLECTION_NAME, points)
            print(f"  ✅ {group_name} 入库 {len(chunks)} 个片段")
        # 构建BM25索引
        tokenized_corpus = [list(text) for text in all_chunks_text]
        bm25 = BM25Okapi(tokenized_corpus)
        print("✅ 多知识库+BM25混合索引构建完成")
    else:
        print("✅ 多知识库向量库已存在，加载BM25索引")
        scroll_res = qdrant.scroll(COLLECTION_NAME, limit=10000)
        all_points = scroll_res[0]
        all_chunks_text = [p.payload["text"] for p in all_points]
        all_chunks_meta = [{"group": p.payload["group"], "id": p.id} for p in all_points]
        tokenized_corpus = [list(text) for text in all_chunks_text]
        bm25 = BM25Okapi(tokenized_corpus)

# RRF融合排序
def rrf_fuse(vector_hits, bm25_hits, k=60):
    rrf_score = {}
    for rank, hit in enumerate(vector_hits):
        doc_id = hit["id"]
        rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1/(k + rank + 1)
    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["id"]
        rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1/(k + rank + 1)
    sorted_docs = sorted(rrf_score.items(), key=lambda x:x[1], reverse=True)
    fused = []
    for doc_id, score in sorted_docs:
        meta_idx = [i for i,m in enumerate(all_chunks_meta) if m["id"] == doc_id][0]
        fused.append({"text": all_chunks_text[meta_idx], "group": all_chunks_meta[meta_idx]["group"], "id": doc_id})
    return fused

# 混合检索：向量+BM25+RRF融合+Rerank
def hybrid_search(query, selected_group, top_k=2):
    q_vec = embedder.encode(query)
    query_filter = None
    if selected_group != "全部知识库":
        query_filter = Filter(must=[FieldCondition(key="group", match=MatchValue(value=selected_group))])
    vec_res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=10,
        query_filter=query_filter
    )
    vector_hits = [{"id": p.id, "text": p.payload["text"]} for p in vec_res.points]

    tokenized_query = list(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = np.argsort(bm25_scores)[-10:][::-1]
    bm25_hits = []
    for idx in top_bm25_idx:
        if selected_group != "全部知识库" and all_chunks_meta[idx]["group"] != selected_group:
            continue
        bm25_hits.append({"id": all_chunks_meta[idx]["id"], "text": all_chunks_text[idx]})

    fused = rrf_fuse(vector_hits, bm25_hits)
    if fused:
        pairs = [[query, item["text"]] for item in fused]
        scores = reranker.compute_score(pairs)
        for i in range(len(fused)):
            fused[i]["rerank_score"] = scores[i]
        fused.sort(key=lambda x:x["rerank_score"], reverse=True)
        fused = [x for x in fused if x["rerank_score"] > RERANK_THRESHOLD]
    return fused[:top_k]

# 系统提示词，几何题用结构化 JSON 描述图形
MODE_PROMPTS = {
    "知识点问答": """你是初中数学助教，优先使用检索到的参考资料回答。
如果资料没有相关内容，请明确说明【参考资料中无该内容】，禁止编造定理。
输出规范：
1. 使用标准Markdown排版；
2. 所有数学符号、公式只用单个 $...$ 包裹（三角函数用 cos/sin/tan 表示，如 cos 30°；不要用"因"代替 cos），示例：$\\triangle ABC$、$\\angle B$、$AB \\parallel CD$；禁止输出 \\$；
3. 普通文字不要随意添加反斜杠 \；
4. 条理清晰，标注易错点。
5. 如果是几何讲解题，末尾用 ===DRAW_JSON=== 分割，紧接着输出一个 JSON 描述图形（不要任何多余文字）；非几何题不要加这个标记。JSON 三选一：
   - 全等三角形：{"type":"congruent_triangles","triangle":{"A":[0,0],"B":[3,0],"C":[0.8,2.6]},"labels2":["D","E","F"],"equal_sides":[["AB","DE"],["AC","DF"]],"title":"全等三角形"}
   - 圆内接四边形/四点共圆：{"type":"cyclic_quadrilateral","labels":["A","B","C","D"],"diameter":["A","C"],"title":"圆内接四边形ABCD"}
   - 普通三角形：{"type":"triangle","points":{"A":[0,0],"B":[3,0],"C":[1.2,2.4]},"equal_sides":["AB","AC"],"right_angle":"B","title":"示例"}
   坐标用普通三角形即可，三个点逆时针排列。""",

    "生成练习题": """你是初中数学老师，根据检索到的知识点出题，贴合中考难度，不超纲。
输出规范：
1. 所有数学符号、公式只用单个 $...$ 包裹（三角函数用 cos/sin/tan 表示，如 cos 30°；不要用"因"代替 cos），示例：$\\triangle ABC$、$\\angle B$、$AB \\parallel CD$；禁止输出 \\$；
2. 普通文字禁止随便加反斜杠，严禁出现 \四周、\角 这类错误写法；
3. 先输出题目，再输出参考解答；Markdown整洁排版。
4. 几何题必须在最后单独一行写 ===DRAW_JSON===，紧接着输出 JSON 图形描述，不要混入正文。JSON 三选一：
   - 全等三角形：{"type":"congruent_triangles","triangle":{"A":[0,0],"B":[3,0],"C":[0.8,2.6]},"labels2":["D","E","F"],"equal_sides":[["AB","DE"],["AC","DF"]],"title":"全等三角形"}
   - 圆内接四边形/四点共圆：{"type":"cyclic_quadrilateral","labels":["A","B","C","D"],"diameter":["A","C"],"title":"圆内接四边形ABCD"}
   - 普通三角形：{"type":"triangle","points":{"A":[0,0],"B":[3,0],"C":[1.2,2.4]},"equal_sides":["AB","AC"],"right_angle":"B","title":"示例"}""",

    "整理错题": """你是家教老师，根据用户指定知识点整理典型错题。
输出规范：
1. 列出3道该知识点高频易错题型；
2. 每道题包含：原题、常见错误解法、正确解法、易错提醒；
3. 所有数学符号、公式只用单个 $...$ 包裹（三角函数用 cos/sin/tan 表示，如 cos 30°；不要用"因"代替 cos）；禁止输出 \\$；
4. 普通文字禁止随便加反斜杠，不要出现异常转义字符。
5. 几何题型，每道题末尾用 ===DRAW_JSON=== 输出图形 JSON。JSON 三选一：
   - 全等三角形：{"type":"congruent_triangles","triangle":{"A":[0,0],"B":[3,0],"C":[0.8,2.6]},"labels2":["D","E","F"],"equal_sides":[["AB","DE"],["AC","DF"]],"title":"全等三角形"}
   - 圆内接四边形/四点共圆：{"type":"cyclic_quadrilateral","labels":["A","B","C","D"],"diameter":["A","C"],"title":"圆内接四边形ABCD"}
   - 普通三角形：{"type":"triangle","points":{"A":[0,0],"B":[3,0],"C":[1.2,2.4]},"equal_sides":["AB","AC"],"right_angle":"B","title":"示例"}"""
}

def normalize_math(text):
    """把单 $...$ 行内公式换成括号式，$$...$$ 保持不变（Gradio 括号式更稳）。"""
    if not text:
        return text
    lp = chr(92) + "("
    rp = chr(92) + ")"
    pattern = "(?<![$])[$](?![$])(.+?)(?<![$])[$](?![$])"
    return re.sub(pattern, lambda m: lp + m.group(1) + rp, text)

def _history_for_llm(history):
    """把历史里含图片的消息转成纯文本，避免把 tuple 传给 DeepSeek。"""
    out = []
    for m in history:
        if isinstance(m, dict):
            role, c = m.get("role"), m.get("content", "")
        else:
            role, c = getattr(m, "role", None), getattr(m, "content", "")
        if role not in ("user", "assistant"):
            continue
        if isinstance(c, (tuple, list)):
            continue
        if isinstance(c, str) and c.strip():
            out.append({"role": role, "content": c})
    return out

_GEOMETRY_HINTS = ["三角形", "△", "全等", "相似", "∠", "角", "垂直", "平行",
                   "⊙", "圆", "四边形", "梯形", "勾股", "直径", "弦", "切线", "半径"]

def _looks_geometric(text):
    return any(k in text for k in _GEOMETRY_HINTS)

def _extract_figure_spec(question, answer):
    """单独调用一次，把几何题解析成图形 JSON（不依赖正文里的 DRAW_JSON）。"""
    fig_prompt = (
        """你是几何图形解析器。请把这道初中几何题解析成一个 JSON，只输出 JSON，不要任何解释。
从下面三种里选最合适的一种：
全等三角形：{"type":"congruent_triangles","triangle":{"A":[0,0],"B":[3,0],"C":[0.8,2.6]},"labels2":["D","E","F"],"equal_sides":[["AB","DE"],["AC","DF"]],"title":"全等三角形"}
圆内接四边形：{"type":"cyclic_quadrilateral","labels":["A","B","C","D"],"diameter":["A","C"],"title":"圆内接四边形ABCD"}
普通三角形：{"type":"triangle","points":{"A":[0,0],"B":[3,0],"C":[1.2,2.4]},"equal_sides":["AB","AC"],"right_angle":"B","title":"示例"}
注意：diameter 填题目中为直径的那两个点字母（题目说 AC 是直径就填 ["A","C"]；BD 是直径就填 ["B","D"]）。
题目："""
        + question
        + """
解答（可忽略）："""
        + answer[:600]
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": fig_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        print(f"图形解析失败：{e}")
        return None

_SOLUTION_MARKERS = ["## 解答", "### 解答", "**解答**", "解答：", "解答:",
                     "## 解", "### 解", "**解**", "解：", "## 解析", "解析："]

def _split_problem_solution(text):
    """把回答拆成（题目, 解答）两段，图片插在题目与解答之间。"""
    best = None
    for mk in _SOLUTION_MARKERS:
        idx = text.find(mk)
        if idx > 0 and (best is None or idx < best):
            best = idx
    if best is not None:
        return text[:best].rstrip(), text[best:].lstrip("\n")
    return "", text

# 主聊天逻辑，拆分正文+结构化绘图 JSON，确定性渲染图片
def gradio_chat(message, chat_state, selected_group, selected_mode):
    if not message.strip():
        return "", chat_state, ""
    search_res = hybrid_search(message.strip(), selected_group, top_k=2)
    ctx_text = "\n\n---\n\n".join([f"**📄 片段{i+1}（来源：{item['group']}）**\n{item['text']}" for i, item in enumerate(search_res)])

    sys_prompt = MODE_PROMPTS[selected_mode]
    prompt = f"""参考资料：
{ctx_text}
用户问题：{message}
"""
    messages = [{"role":"system", "content": sys_prompt}]
    messages.extend(_history_for_llm(chat_state))
    messages.append({"role":"user", "content": prompt})

    retry_delay = 1
    ans = ""
    for i in range(3):
        try:
            resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.3)
            ans = resp.choices[0].message.content
            break
        except RateLimitError:
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            return f"⚠️ API错误：{str(e)}", chat_state, ctx_text

    # 拆分正文和结构化绘图 JSON
    spec = parse_draw_json(ans)
    if "===DRAW_JSON===" in ans:
        content_part = ans.split("===DRAW_JSON===", 1)[0].strip()
    else:
        content_part = ans.strip()

    # 几何题：单独解析一次图形，比正文里嵌的 JSON 更可靠；失败则保留正文里的
    if _looks_geometric(content_part):
        parsed = _extract_figure_spec(message, content_part)
        if parsed:
            spec = parsed

    content_part = latex_to_unicode(content_part)
    img = render_figure(spec) if spec else None

    # 题目 / 解答 拆分：图片放在题目下面、解答上面
    problem, solution = _split_problem_solution(content_part)
    chat_state.append({"role":"user", "content": message})
    if img is not None:
        try:
            fig_path = os.path.join(FIG_DIR, f"fig_{int(time.time() * 1000)}.png")
            img.save(fig_path)
            if problem:
                chat_state.append({"role":"assistant", "content": problem})
            chat_state.append({"role":"assistant", "content": (fig_path, "几何图")})
            if solution:
                chat_state.append({"role":"assistant", "content": solution})
        except Exception as e:
            print(f"几何图保存失败：{e}")
            chat_state.append({"role":"assistant", "content": content_part})
    else:
        chat_state.append({"role":"assistant", "content": content_part})
    return "", chat_state, ctx_text if ctx_text else "（无匹配相关片段）"

def clear_chat():
    return [], ""

# -------------------------- Gradio网页界面（内置MathJax） --------------------------
init_vector_store()

with gr.Blocks(title="初中数学RAG家教助手") as demo:
    gr.HTML("""
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$','$$']]
      },
      startup: {
        pageReady: function() {
            return MathJax.startup.defaultPageReady().then(function() {
                console.log("MathJax 加载完成");
            });
        }
      }
    };
    </script>
    """)
    gr.Markdown("# 📐 初中数学RAG家教助手（自带几何图生成）")
    # 顶部控制栏
    with gr.Row():
        group_dropdown = gr.Dropdown(choices=list(PDF_GROUPS.keys()), value="全部知识库", label="选择知识库分组", scale=1)
        mode_radio = gr.Radio(choices=["知识点问答", "生成练习题", "整理错题"], value="知识点问答", label="对话模式", scale=2)
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=550, type="messages")
            msg = gr.Textbox(placeholder="输入知识点，例如：全等三角形SAS判定", lines=2)
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")
        with gr.Column(scale=1):
            gr.Markdown("### 📚 检索参考片段")
            ctx_md = gr.Markdown(height=550, sanitize_html=False)

    msg.submit(gradio_chat, [msg, chatbot, group_dropdown, mode_radio], [msg, chatbot, ctx_md])
    submit_btn.click(gradio_chat, [msg, chatbot, group_dropdown, mode_radio], [msg, chatbot, ctx_md])
    clear_btn.click(clear_chat, None, [chatbot, ctx_md])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, allowed_paths=[FIG_DIR])
