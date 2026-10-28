# -*- coding: utf-8 -*-
"""确定性几何图渲染：LLM 输出结构化 JSON，本模块用 matplotlib 精确绘制。

替代 AI 文生图 —— 图形保证正确、字母清晰、可标注等边/角。
用法：
    from geometry_render import render_figure, parse_draw_json
    spec = parse_draw_json(llm_output)     # 从输出里提取 JSON
    img  = render_figure(spec)             # PIL.Image 或 None
"""
from __future__ import annotations

import io
import json
import math

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

# Windows 中文/负号正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DPI = 150


# ---------------------------- 基础绘图工具 ----------------------------
def _new_ax():
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _draw_segment(ax, p1, p2, color="k", lw=2.0):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw,
            solid_capstyle="round")


def _draw_point(ax, p, name):
    ax.plot(p[0], p[1], "ko", ms=4)
    # 标签放在顶点右上方，避开线段
    ax.text(p[0] + 0.14, p[1] + 0.14, f"${name}$",
            fontsize=14, ha="center", va="center")


def _equal_side_ticks(ax, p1, p2, n=1):
    """等边标记：在边中点画 n 道垂直小竖线（n=1/2 区分不同等边）。"""
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length      # 单位方向
    nx, ny = -uy, ux                       # 单位法向
    for i in range(n):
        off = (i - (n - 1) / 2) * 0.14
        cx, cy = mx + nx * off, my + ny * off
        ax.plot([cx - 0.07 * ux, cx + 0.07 * ux],
                [cy - 0.07 * uy, cy + 0.07 * uy], "k-", lw=1.3)


def _angle_arc(ax, vertex, p1, p2, r=0.32):
    """在顶点处画 p1-vertex-p2 的夹角弧线。"""
    a1 = math.degrees(math.atan2(p1[1] - vertex[1], p1[0] - vertex[0]))
    a2 = math.degrees(math.atan2(p2[1] - vertex[1], p2[0] - vertex[0]))
    d = (a2 - a1) % 360
    if d > 180:
        a1, a2 = a2, a1
        d = 360 - d
    ax.add_patch(Arc(vertex, 2 * r, 2 * r, angle=0,
                     theta1=a1, theta2=a1 + d, color="k", lw=1.0))


def _right_angle_mark(ax, vertex, p1, p2, s=0.25):
    """直角小方块标记。"""
    vx, vy = vertex

    def unit(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        return dx / L, dy / L

    u1x, u1y = unit(vertex, p1)
    u2x, u2y = unit(vertex, p2)
    ax.plot([vx + s * u1x, vx + s * (u1x + u2x), vx + s * u2x],
            [vy + s * u1y, vy + s * (u1y + u2y), vy + s * u2y], "k-", lw=1.3)


def _autoscale(ax, pts, pad=0.9):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def _to_pil(fig):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------- 具体图形模板 ----------------------------
def _render_triangle(spec):
    pts = {k: tuple(v) for k, v in spec["points"].items()}
    names = list(pts.keys())
    fig, ax = _new_ax()
    for i in range(3):
        _draw_segment(ax, pts[names[i]], pts[names[(i + 1) % 3]])
    for side in spec.get("equal_sides", []):
        if len(side) == 2 and side[0] in pts and side[1] in pts:
            _equal_side_ticks(ax, pts[side[0]], pts[side[1]])
    rv = spec.get("right_angle")
    if rv in pts:
        others = [n for n in names if n != rv]
        if len(others) >= 2:
            _right_angle_mark(ax, pts[rv], pts[others[0]], pts[others[1]])
    for n, p in pts.items():
        _draw_point(ax, p, n)
    _autoscale(ax, list(pts.values()))
    if spec.get("title"):
        ax.set_title(spec["title"], fontsize=13, pad=10)
    return _to_pil(fig)


def _render_congruent_triangles(spec):
    """两个严格全等的三角形：第二个由第一个平移+旋转得到。"""
    t1 = {k: tuple(v) for k, v in spec["triangle"].items()}
    names1 = list(t1.keys())
    names2 = spec.get("labels2", ["D", "E", "F"])
    shift_x, shift_y = 4.4, 0.4
    ang = math.radians(18)
    t2 = {}
    for n1, n2 in zip(names1, names2):
        x, y = t1[n1]
        t2[n2] = (x * math.cos(ang) - y * math.sin(ang) + shift_x,
                  x * math.sin(ang) + y * math.cos(ang) + shift_y)
    P = {**t1, **t2}
    fig, ax = _new_ax()
    for i in range(3):
        _draw_segment(ax, t1[names1[i]], t1[names1[(i + 1) % 3]])
        _draw_segment(ax, t2[names2[i]], t2[names2[(i + 1) % 3]])
    # 对应等边：同组边用同样道数的竖线标记
    eq = spec.get("equal_sides", [["AB", "DE"], ["AC", "DF"]])
    for i, pair in enumerate(eq):
        s1, s2 = pair[0], pair[1]
        if (len(s1) == 2 and len(s2) == 2
                and s1[0] in P and s1[1] in P and s2[0] in P and s2[1] in P):
            _equal_side_ticks(ax, P[s1[0]], P[s1[1]], n=i + 1)
            _equal_side_ticks(ax, P[s2[0]], P[s2[1]], n=i + 1)
    for n, p in P.items():
        _draw_point(ax, p, n)
    _autoscale(ax, list(P.values()))
    if spec.get("title"):
        ax.set_title(spec["title"], fontsize=13, pad=10)
    return _to_pil(fig)


def _render_parallel_transversal(spec):
    """两条平行线 + 一条截线（可在此基础上扩展标角）。"""
    fig, ax = _new_ax()
    y1, y2 = 3.2, 0.8
    _draw_segment(ax, (-3.0, y1), (3.0, y1))
    _draw_segment(ax, (-3.0, y2), (3.0, y2))
    _draw_segment(ax, (-2.4, 4.2), (2.4, 0.0))
    # 平行线箭头标记
    for y in (y1, y2):
        ax.plot([-2.7, -2.4], [y, y], "k-", lw=1.2)
        ax.plot([-2.7, -2.4], [y + 0.12, y + 0.12], "k-", lw=1.2)
    _autoscale(ax, [(-3, y1), (3, y1), (-3, y2), (3, y2), (-2.4, 4.2), (2.4, 0.0)])
    return _to_pil(fig)


# ---------------------------- 入口与解析 ----------------------------
def parse_draw_json(text):
    """从 LLM 输出中提取 ===DRAW_JSON=== 之后的 JSON；失败返回 None。"""
    marker = "===DRAW_JSON==="
    if marker not in text:
        return None
    raw = text.split(marker, 1)[1].strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def _render_cyclic_quadrilateral(spec):
    """圆内接四边形：四点共圆，按顺序连接，可标直角。

    优先用 spec["diameter"]（题目里为直径的两个点）把这两点放成对径；
    否则用 right_angle（直角顶点）反推对径点。
    """
    labels = list(spec.get("labels", ["A", "B", "C", "D"]))[:4]
    n = len(labels)
    R = float(spec.get("radius", 2.0))
    rv = spec.get("right_angle")
    dia = spec.get("diameter")
    pts = {}
    diag = None
    if n == 4 and dia and len(dia) == 2 and dia[0] in labels and dia[1] in labels:
        k0 = labels.index(dia[0])
        if labels[(k0 + 2) % 4] == dia[1]:
            diag = (labels[k0], labels[(k0 + 2) % 4],
                    labels[(k0 + 1) % 4], labels[(k0 + 3) % 4])

    if "angles" in spec:
        angs = list(spec["angles"])[:4]
        for lab, a in zip(labels, angs):
            rad = math.radians(a)
            pts[lab] = (R * math.cos(rad), R * math.sin(rad))
    elif diag:
        d0, d1, nxt, prv = diag
        ang_map = {d0: 180, d1: 0, nxt: 250, prv: 90}
        for lab, a in ang_map.items():
            rad = math.radians(a)
            pts[lab] = (R * math.cos(rad), R * math.sin(rad))
    elif n == 4 and rv in labels:
        k = labels.index(rv)
        ang_map = {
            labels[(k - 1) % 4]: 180,
            labels[k]: 270,
            labels[(k + 1) % 4]: 0,
            labels[(k + 2) % 4]: 90,
        }
        for lab, a in ang_map.items():
            rad = math.radians(a)
            pts[lab] = (R * math.cos(rad), R * math.sin(rad))
    else:
        for i, lab in enumerate(labels):
            rad = math.radians(i * 90)
            pts[lab] = (R * math.cos(rad), R * math.sin(rad))
    fig, ax = _new_ax()
    ax.add_patch(Circle((0, 0), R, fill=False, lw=1.2, color="k"))
    ax.plot(0, 0, "k.", ms=2.5)
    order = [lab for lab in labels if lab in pts]
    for i in range(len(order)):
        _draw_segment(ax, pts[order[i]], pts[order[(i + 1) % len(order)]])
    # 直径画成虚线，便于看出哪条是直径
    if diag and diag[0] in pts and diag[1] in pts:
        ax.plot([pts[diag[0]][0], pts[diag[1]][0]],
                [pts[diag[0]][1], pts[diag[1]][1]], "k--", lw=1.0)
    # 标直角：diameter 模式下两个非直径顶点都是直角
    if diag:
        for v in (diag[2], diag[3]):
            kk = labels.index(v)
            p1 = labels[(kk - 1) % 4]
            p2 = labels[(kk + 1) % 4]
            if v in pts and p1 in pts and p2 in pts:
                _right_angle_mark(ax, pts[v], pts[p1], pts[p2])
    elif rv in pts and n == 4:
        kk = labels.index(rv)
        p1 = labels[(kk - 1) % 4]
        p2 = labels[(kk + 1) % 4]
        _right_angle_mark(ax, pts[rv], pts[p1], pts[p2])
    for nm, p in pts.items():
        _draw_point(ax, p, nm)
    _autoscale(ax, list(pts.values()), pad=0.6)
    if spec.get("title"):
        ax.set_title(spec["title"], fontsize=13, pad=10)
    return _to_pil(fig)


def render_figure(spec):
    """按 type 分发到具体渲染器。spec 为 None 或类型未知时返回 None。"""
    if not spec:
        return None
    t = spec.get("type")
    if t == "triangle":
        return _render_triangle(spec)
    if t == "congruent_triangles":
        return _render_congruent_triangles(spec)
    if t == "parallel_transversal":
        return _render_parallel_transversal(spec)
    if t == "cyclic_quadrilateral":
        return _render_cyclic_quadrilateral(spec)
    return None


if __name__ == "__main__":
    # 自测：生成一张全等三角形图
    spec = {
        "type": "congruent_triangles",
        "triangle": {"A": [0, 0], "B": [3, 0], "C": [0.8, 2.6]},
        "labels2": ["D", "E", "F"],
        "equal_sides": [["AB", "DE"], ["AC", "DF"]],
        "title": "全等三角形 SAS 判定示意",
    }
    img = render_figure(spec)
    img.save("_geometry_test.png")
    print("已生成 _geometry_test.png，尺寸", img.size)
