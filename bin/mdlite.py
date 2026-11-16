"""mdlite: the Markdown that the book's text files accept, on top of their HTML.

Used by bin/build-roofline.py (on the private text source and the _roofline/*.html files) and by
bin/site-edit (the live preview), so that the preview and the built page agree.

Inline, anywhere outside tags and math:
    **bold**   *italic*   ~~struck~~   `code`   [text](url)      (<u>underline</u> is plain HTML)
Blocks, separated by blank lines:
    - item / * item / 1. item lines     a list (indent by two spaces to nest)
    ### Heading, #### Heading            a heading
    a chunk of plain text                a paragraph (in a file whose text is not already in <p>)
    a <p> cut in two by a blank line     two paragraphs
HTML is left as it is, and nothing inside <script>, <style>, <pre>, <code>, <svg>, comments or math
(\\( \\), \\[ \\], $$ $$) is touched.
"""
import re

RAW = re.compile(r"<(script|style|pre|svg|textarea)\b.*?</\1\s*>|<!--.*?-->", re.S | re.I)
# Inline material that is never rewritten: math, code and MathML.
MATH = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|<(code|math|kbd|samp)\b.*?</\1\s*>", re.S | re.I)
TAG = re.compile(r"<[^>]*>")
BLOCK_TAG = re.compile(r"</?(p|div|ul|ol|li|table|thead|tbody|tr|td|th|section|figure|figcaption|details|summary|"
                       r"h[1-6]|blockquote|dl|dt|dd|main|article|header|footer|nav|aside|hr|br|form)\b", re.I)
ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])[ \t]+(.*)$")
HEAD = re.compile(r"^(\s*)(#{3,6})[ \t]+(.+?)\s*#*\s*$")
PH = re.compile(r"[\x00\x01\x02](\d+)[\x00\x01\x02]")


def inline(s: str) -> str:
    """Inline Markdown in text whose tags, math and raw elements are already placeholders."""
    tags = []
    s = TAG.sub(lambda m: tags.append(m.group(0)) or f"\x02{len(tags) - 1}\x02", s)
    codes = []
    s = re.sub(r"`([^`\n]+)`", lambda m: codes.append("<code>" + m.group(1).replace("&", "&amp;").replace("<", "&lt;") + "</code>")
               or f"\x02c{len(codes) - 1}\x02", s)
    s = re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"(?<![\w*\\])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![\w*])", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*\\])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"(?<!~)~~(?=\S)(.+?)(?<=\S)~~(?!~)", r"<s>\1</s>", s)
    s = re.sub(r"\x02c(\d+)\x02", lambda m: codes[int(m.group(1))], s)
    return re.sub(r"\x02(\d+)\x02", lambda m: tags[int(m.group(1))], s)


def _list(lines):
    """Lines of a list (marker lines and their continuations) -> nested <ul>/<ol>."""
    out, stack = [], []  # stack of [indent, tag]

    def close():
        out.append(f"</li></{stack.pop()[1]}>")
    for ln in lines:
        m = ITEM.match(ln)
        if not m:  # continuation of the previous item
            out.append(" " + inline(ln.strip()))
            continue
        ind, mark, text = len(m.group(1).expandtabs(4)), m.group(2), m.group(3)
        while stack and ind < stack[-1][0]:
            close()
        if stack and ind == stack[-1][0]:
            out.append("</li>")
        else:
            stack.append([ind, "ol" if mark[0].isdigit() else "ul"])
            out.append(f"<{stack[-1][1]}>")
        out.append(f"<li>{inline(text)}")
    while stack:
        close()
    return "".join(out)


def _chunk(c: str, wrap: bool) -> str:
    lines = c.split("\n")
    s = c.strip()
    if not s:
        return c
    indent = re.match(r"\s*", c).group(0)
    m = HEAD.match(s) if "\n" not in s else None
    if m:
        n = len(m.group(2))
        return f"{indent}<h{n}>{inline(m.group(3))}</h{n}>"
    # A list: from the first marker line on, every line is a marker line or an indented continuation.
    first = next((i for i, ln in enumerate(lines) if ITEM.match(ln)), None)
    if first is not None and not BLOCK_TAG.search("\n".join(lines[first:])) and \
            all(ITEM.match(ln) or ln.startswith((" ", "\t")) for ln in lines[first + 1:] if ln.strip()):
        head = "\n".join(lines[:first])
        ul = indent + _list([ln for ln in lines[first:] if ln.strip()])
        return (_chunk(head, wrap) + "\n" + ul) if head.strip() else ul
    out = inline(c)
    has_block = BLOCK_TAG.search(s) or "\x00" in s
    if not has_block:
        return f"{indent}<p>{out.strip()}</p>" if wrap else out
    # A <p> cut in two by a blank line: close the first half, open the second.
    opens, closes = len(re.findall(r"<p\b", s)), len(re.findall(r"</p>", s))
    if s.startswith("<p") and opens > closes:
        out = out.rstrip() + "</p>"
    elif s.endswith("</p>") and closes > opens:
        out = indent + "<p>" + out.lstrip()
    return out


def render(text: str, wrap: bool = True) -> str:
    """Markdown blocks and inline syntax in HTML text. wrap=False leaves plain chunks unwrapped
    (text that the page already puts inside an element)."""
    raw = []
    t = RAW.sub(lambda m: raw.append(m.group(0)) or f"\x00{len(raw) - 1}\x00", text)
    maths = []
    t = MATH.sub(lambda m: maths.append(m.group(0)) or f"\x01{len(maths) - 1}\x01", t)
    parts = re.split(r"(\n[ \t]*\n(?:[ \t]*\n)*)", t)
    t = "".join(p if k % 2 else _chunk(p, wrap) for k, p in enumerate(parts))
    t = re.sub(r"\x01(\d+)\x01", lambda m: maths[int(m.group(1))], t)
    return re.sub(r"\x00(\d+)\x00", lambda m: raw[int(m.group(1))], t)


def render_inline(text: str) -> str:
    raw = []
    t = RAW.sub(lambda m: raw.append(m.group(0)) or f"\x00{len(raw) - 1}\x00", text)
    maths = []
    t = MATH.sub(lambda m: maths.append(m.group(0)) or f"\x01{len(maths) - 1}\x01", t)
    t = inline(t)
    t = re.sub(r"\x01(\d+)\x01", lambda m: maths[int(m.group(1))], t)
    return re.sub(r"\x00(\d+)\x00", lambda m: raw[int(m.group(1))], t)
