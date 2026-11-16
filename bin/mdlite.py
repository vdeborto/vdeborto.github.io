"""mdlite: the Markdown that the book's text files accept, on top of their HTML.

Used by bin/build-roofline.py (on the private text source and the _roofline/*.html files) and by
the site-edit editor (~/code/site-edit, its live preview), so that the preview and the built page agree.

Inline, anywhere outside tags and math:
    **bold**   *italic*   ~~struck~~   `code`   [text](url)      (<u>underline</u> is plain HTML)
Blocks, separated by blank lines:
    - item / * item / 1. item lines     a list (indent by two spaces to nest)
    ### Heading, #### Heading            a heading
    a chunk of plain text                a paragraph (in a file whose text is not already in <p>)
    a <p> cut in two by a blank line     two paragraphs
Footnotes: [^label] in the text and, anywhere in the book, a line "[^label]: the note" (continued on
the following lines of text, up to a blank line or a line of HTML). A note line written without the
colon, "[^label] the note", counts as one when it starts a block (after a blank line or a line of
HTML), the same text marks [^label] elsewhere, and defines it nowhere else. footnotes() numbers them per page, in order of first use,
and lists the notes at the end of the page.
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
# A note continues on the following lines of text, up to a blank line, a line that opens with HTML (or
# a comment, such as a figure's placeholder) and the next note.
FNDEF = re.compile(r"^[ \t]*\[\^([\w-]+)\]:[ \t]*(.*(?:\n(?![ \t]*\n)(?![ \t]*\[\^[\w-]+\]:)(?![ \t]*[<\x00]).*)*)", re.M)
FNREF = re.compile(r"\[\^([\w-]+)\](?!:)")
# "[^label] the note", the colon forgotten, at the start of a line (see the module's docstring).
COLONLESS = re.compile(r"^([ \t]*)\[\^([\w-]+)\][ \t]+(?=\S)", re.M)


def _add_colons(t: str) -> str:
    """Read a note line written without its colon as a note (see the docstring)."""
    defined = set(re.findall(r"^[ \t]*\[\^([\w-]+)\]:", t, re.M))

    def fix(m):
        lab = m.group(2)
        lines = t[:m.start()].split("\n")  # the last item is this line's indentation
        prev = lines[-2] if len(lines) >= 2 else ""
        starts_block = len(lines) < 2 or not prev.strip() or prev.rstrip().endswith(">") or prev.lstrip().startswith("\x00")
        used_elsewhere = len(re.findall(r"\[\^" + re.escape(lab) + r"\](?!:)", t)) >= 2
        if lab in defined or not starts_block or not used_elsewhere:
            return m.group(0)
        return f"{m.group(1)}[^{lab}]: "
    return COLONLESS.sub(fix, t)
DEFDIV = re.compile(r'<div class="fn-def" data-fn="([\w-]+)" hidden>(.*?)</div>', re.S)
REFSUP = re.compile(r'<sup class="fn-ref" data-fn="([\w-]+)">\[[\w-]+\]</sup>')
PH = re.compile(r"[\x00\x01\x02](\d+)[\x00\x01\x02]")


def inline(s: str) -> str:
    """Inline Markdown in text whose tags, math and raw elements are already placeholders."""
    tags = []
    s = TAG.sub(lambda m: tags.append(m.group(0)) or f"\x02{len(tags) - 1}\x02", s)
    codes = []
    s = re.sub(r"`([^`\n]+)`", lambda m: codes.append("<code>" + m.group(1).replace("&", "&amp;").replace("<", "&lt;") + "</code>")
               or f"\x02c{len(codes) - 1}\x02", s)
    s = FNREF.sub(lambda m: f'<sup class="fn-ref" data-fn="{m.group(1)}">[{m.group(1)}]</sup>', s)
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


def starts_block(line: str) -> bool:
    """A line that opens or closes a block element, or holds a comment or raw element: it ends a
    paragraph. A line that starts with an inline tag (a link, </a>, <em>) does not."""
    t = line.lstrip()
    return t.startswith("\x00") or bool(re.match(r"</?(p|div|ul|ol|li|table|thead|tbody|tr|td|th|section|figure|figcaption|"
                                                  r"details|summary|h[1-6]|blockquote|dl|dt|dd|header|footer|nav|aside|hr|pre|svg|script|style)\b", t, re.I))


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
    # A "</li>" left at the end of an item, from text that was HTML before, is dropped.
    lines = [re.sub(r"\s*</li>\s*$", "", ln) if ITEM.match(ln) else ln for ln in lines]
    first = next((i for i, ln in enumerate(lines) if ITEM.match(ln)), None)
    if first is not None:
        # The list runs to the first line that opens with HTML or a comment (a figure's placeholder,
        # a <p>): that line and the ones after it are a chunk of their own.
        last = first + 1
        while last < len(lines) and (ITEM.match(lines[last]) or (lines[last].startswith((" ", "\t")) and not lines[last].lstrip().startswith(("<", "\x00")))):
            last += 1
        body, tail = lines[first:last], "\n".join(lines[last:])
        if not BLOCK_TAG.search("\n".join(body)) and (not tail.strip() or tail.lstrip().startswith(("<", "\x00"))):
            head = "\n".join(lines[:first])
            ul = indent + _list([ln for ln in body if ln.strip()])
            out = (_chunk(head, wrap) + "\n" + ul) if head.strip() else ul
            return out + ("\n" + _chunk(tail, wrap) if tail.strip() else "")
    out = inline(c)
    has_block = BLOCK_TAG.search(s) or "\x00" in s
    if not has_block:
        return f"{indent}<p>{out.strip()}</p>" if wrap else out
    # Lines of plain text right after a closed element (a heading, a paragraph, a figure's placeholder),
    # up to the next line of HTML: a paragraph written without a blank line before it.
    if wrap:
        ls, out_ls, k = out.split("\n"), [], 0
        closed = re.compile(r"(</(p|h[1-6]|ul|ol|div|table|figure|section|details|blockquote|dl)>|-->|\x00\d+\x00)\s*$")
        while k < len(ls):
            ln = ls[k]
            # a paragraph starts with text; its later lines may start with an inline tag (</a>, <em>)
            if ln.strip() and not ln.lstrip().startswith(("<", "\x00")) and (not out_ls or closed.search(out_ls[-1])):
                j = k
                while j < len(ls) and ls[j].strip() and not starts_block(ls[j]):
                    j += 1
                run = "\n".join(ls[k:j])
                # text that closes an element opened earlier (the rest of a list item) is not a paragraph
                unbalanced = any(run.count(f"</{t}>") > len(re.findall(rf"<{t}\b", run)) for t in set(re.findall(r"</(\w+)>", run)))
                if unbalanced:
                    out_ls += ls[k:j]
                else:
                    ind = re.match(r"\s*", ln).group(0)
                    out_ls.append(ind + "<p>" + " ".join(x.strip() for x in ls[k:j]) + "</p>")
                k = j
            else:
                out_ls.append(ln)
                k += 1
        out = "\n".join(out_ls)
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

    def fndef(m):  # a footnote definition: kept out of the text, on a line of its own
        body = inline(" ".join(ln.strip() for ln in m.group(2).split("\n")))
        raw.append(f'<div class="fn-def" data-fn="{m.group(1)}" hidden>{body}</div>')
        return f"\n\x00{len(raw) - 1}\x00\n"
    t = FNDEF.sub(fndef, _add_colons(t))
    parts = re.split(r"(\n[ \t]*\n(?:[ \t]*\n)*)", t)
    out = []
    for k, p in enumerate(parts):
        if k % 2:
            out.append(p)
            continue
        c = _chunk(p, wrap)
        # Items separated by blank lines are one list, as in Markdown: a list chunk right after
        # another (only blank lines between) continues it.
        m = re.match(r"(\s*)<(ul|ol)>", c)
        if m and k >= 2 and _pure_list(p) and _pure_list(parts[k - 2]) and out[-2].rstrip().endswith(f"</{m.group(2)}>"):
            out.pop()  # the blank lines
            out[-1] = out[-1].rstrip()[:-len(f"</{m.group(2)}>")]
            c = c[m.end():]
        out.append(c)
    t = "".join(out)
    t = re.sub(r"\x00(\d+)\x00", lambda m: raw[int(m.group(1))], t)
    return re.sub(r"\x01(\d+)\x01", lambda m: maths[int(m.group(1))], t)


def _pure_list(chunk: str) -> bool:
    """Whether a chunk is only list lines (markers and indented continuations)."""
    lines = [ln for ln in chunk.split("\n") if ln.strip()]
    return bool(lines) and bool(ITEM.match(lines[0])) and all(ITEM.match(ln) or ln.startswith((" ", "\t")) for ln in lines[1:]) \
        and not BLOCK_TAG.search(re.sub(r"\s*</li>\s*$", "", chunk, flags=re.M))


def render_inline(text: str) -> str:
    raw = []
    t = RAW.sub(lambda m: raw.append(m.group(0)) or f"\x00{len(raw) - 1}\x00", text)
    maths = []
    t = MATH.sub(lambda m: maths.append(m.group(0)) or f"\x01{len(maths) - 1}\x01", t)
    t = inline(t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: raw[int(m.group(1))], t)
    return re.sub(r"\x01(\d+)\x01", lambda m: maths[int(m.group(1))], t)


def collect(html: str) -> dict:
    """The footnote definitions in rendered HTML: label -> note."""
    return {m.group(1): m.group(2) for m in DEFDIV.finditer(html)}


def defs_in_source(text: str) -> dict:
    """The footnote definitions of a source file, rendered (for the live preview)."""
    return collect(render(text))


def footnotes(html: str, defs=None, missing=None) -> str:
    """Number the footnote references of a page, 1, 2, ... in order of first use; drop the hidden
    definitions and list the notes at the end. `defs` adds definitions from elsewhere (the whole
    book); labels with no definition are appended to `missing`."""
    d = dict(defs or {})
    d.update(collect(html))
    html = DEFDIV.sub("", html)
    order = {}

    def ref(m):
        lab = m.group(1)
        first = lab not in order
        if first:
            order[lab] = len(order) + 1
        tip = re.sub(r"<[^>]+>", "", d.get(lab, "")).replace('"', "&quot;")
        anchor = f' id="fnref-{lab}"' if first else ""
        return f'<sup class="fn-ref"{anchor}><a href="#fn-{lab}" title="{tip}">{order[lab]}</a></sup>'
    html = REFSUP.sub(ref, html)
    if not order:
        return html
    if missing is not None:
        missing += [lab for lab in order if lab not in d]
    items = "".join(f'<li id="fn-{lab}">{d.get(lab, "<em>missing note</em>")} <a href="#fnref-{lab}" class="fn-back" '
                    f'aria-label="Back to the text">↩</a></li>' for lab in order)
    notes = f'<section class="footnotes" aria-label="Notes"><h3>Notes</h3><ol>{items}</ol></section>\n'
    body = html.rstrip()
    if body.endswith("</section>"):  # inside the chapter's own section, so that it keeps the text column
        return body[:-len("</section>")] + notes + "</section>"
    return body + "\n" + notes


def labels(text: str) -> set:
    return set(re.findall(r"\[\^([\w-]+)\]", text))
