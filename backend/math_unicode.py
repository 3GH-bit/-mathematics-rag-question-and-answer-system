# -*- coding: utf-8 -*-
"""把 LaTeX 公式/命令转成 Unicode 纯文本，保证任何渲染器都能显示。"""
import re

_B = chr(92)  # 反斜杠

_LATEX_UNICODE = [
    (_B + "triangle", "△"), (_B + "odot", "⊙"), (_B + "angle", "∠"),
    (_B + "parallel", "∥"), (_B + "perp", "⊥"), (_B + "cong", "≅"),
    (_B + "sim", "∽"), (_B + "neq", "≠"), (_B + "leq", "≤"), (_B + "geq", "≥"),
    (_B + "le", "≤"), (_B + "ge", "≥"), (_B + "ne", "≠"),
    (_B + "times", "×"), (_B + "div", "÷"), (_B + "pm", "±"), (_B + "cdot", "·"),
    (_B + "pi", "π"), (_B + "alpha", "α"), (_B + "beta", "β"),
    (_B + "gamma", "γ"), (_B + "delta", "δ"), (_B + "theta", "θ"),
    (_B + "infty", "∞"), (_B + "because", "∵"), (_B + "therefore", "∴"),
    (_B + "circ", "°"), (_B + "Rightarrow", "⇒"), (_B + "rightarrow", "→"),
    (_B + "to", "→"),
    # 纯排版命令：直接去掉
    (_B + "boxed", ""), (_B + "text", ""), (_B + "mathrm", ""),
    (_B + "mathbf", ""), (_B + "operatorname", ""), (_B + "displaystyle", ""),
    # 空白命令
    (_B + "qquad", "  "), (_B + "quad", " "),
]
_LATEX_UNICODE.sort(key=lambda x: -len(x[0]))

# \frac{a}{b} / \dfrac{a}{b} 里 a、b 不含花括号
_FRAC_PAT = (_B * 2 + "d?frac" + _B + "{" + "([^{}]*)" + _B + "}"
             + _B + "{" + "([^{}]*)" + _B + "}")

_FUNCS = ["arccos", "arcsin", "arctan", "cos", "sin", "tan", "cot", "sec", "csc",
          "log", "ln", "lim", "max", "min"]


def _frac_repl(m):
    a, b = m.group(1), m.group(2)
    if any(ch in b for ch in "/+-×·"):
        b = "(" + b + ")"
    if any(ch in a for ch in "/+-×·"):
        a = "(" + a + ")"
    return a + "/" + b


def latex_to_unicode(text):
    if not text:
        return text
    s = text
    s = s.replace(_B + "left", "").replace(_B + "right", "")
    # 根号 \sqrt{a} -> √a（连花括号一起吃掉）
    s = re.sub(_B * 2 + "sqrt" + _B + "{" + "([^{}]*)" + _B + "}",
               lambda m: "√" + m.group(1), s)
    s = s.replace(_B + "sqrt", "√")
    # 分数：多轮替换，先吃掉无嵌套的内层
    prev = None
    while prev != s:
        prev = s
        s = re.sub(_FRAC_PAT, _frac_repl, s)
    # 角度与上标
    s = s.replace("^" + _B + "circ", "°")
    s = s.replace("^{2}", "²").replace("^2", "²")
    s = s.replace("^{3}", "³").replace("^3", "³")
    # 命令 -> Unicode / 去掉
    for cmd, uni in _LATEX_UNICODE:
        s = s.replace(cmd, uni)
    # 函数名去反斜杠
    for fn in _FUNCS:
        s = s.replace(_B + fn, fn)
    # 去定界符与空白命令
    s = s.replace("$$", "").replace("$", "")
    s = s.replace(_B + "(", "").replace(_B + ")", "")
    s = s.replace(_B + "[", "").replace(_B + "]", "")
    s = s.replace(_B + ",", "").replace(_B + ";", "").replace(_B + "!", "")
    s = s.replace(_B + " ", " ")
    # 去花括号
    s = s.replace("{", "").replace("}", "")
    # 清理仍残留的 \命令（去掉反斜杠）
    s = re.sub(_B * 2 + "[a-zA-Z]+", lambda m: m.group(0)[1:], s)
    # 修正模型偶尔把 cos 误写成“因”（如“因30°”）
    s = re.sub("因" + _B + "s*(" + _B + "d+)" + _B + "s*°",
               lambda m: "cos " + m.group(1) + "°", s)
    return s
