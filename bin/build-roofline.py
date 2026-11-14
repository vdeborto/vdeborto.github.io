#!/usr/bin/env python3
"""Build the website version of the Block Diffusion Rooflines note.

Source of truth for the text, the cost model and the calculator: the private repo
~/code/diffusion-roofline (site/block-diffusion-rooflines.html). This script turns that standalone
page into a page of this Jekyll site, in the site's own layout and palette, and adds the
interactive figures kept in this repo:

  _roofline/figures.html        markup of the interactive figures, one <!-- fig:NAME --> block each
  _roofline/cards.html          "more" panels of the TL;DR cards, one <!-- card:TITLE PREFIX --> block each
  _roofline/gists.html          one-sentence summary of each definition and proposition box
  assets/roofline/reading.js    contents rail, reading progress, and the intuition / full-math switch
Every AR quantity in the math (I_{AR}, W_{AR}, F^{AR}_{...}, B^\\star_{AR}, ...) is wrapped in
\\class{mj-ar}{...} and every TD quantity in \\class{mj-td}{...}, so that it takes the figures' colours.
  assets/roofline/theme.css     the note's colour and font tokens mapped to the site's palette,
                                layout overrides, and the figures' styles (hand-written)
  assets/roofline/figures.js    the interactive figures (hand-written; reads window.BDModel / BDR)
  assets/roofline/mathjax.js    restricts the site's MathJax to \\( \\) and \\[ \\] on this page

Generated (do not edit by hand; rerun this script):
  assets/roofline/note.css      the note's own CSS, scoped under .bdr, tokens renamed --bdr-*
  assets/roofline/note.js       the note's cost model, figures and calculator
  notes/block-diffusion-rooflines-6f924020/index.html          the book's first page: introduction
                                                               and contents
  notes/block-diffusion-rooflines-6f924020/<chapter>/index.html one page per section of the note

The note is published as a short book: CHAPTERS below maps each section of the source to a page,
with "Part N of" and previous / next links, a contents rail listing every chapter, and links
between chapters rewritten to the right page. Old single-page links (#prop1, ?B=64) are
forwarded by the first page to the chapter that now holds the anchor.

Every page is unlisted: no link from the site, robots noindex (front matter `robots`), excluded
from the sitemap (_config.yml defaults for notes/). Usage:
    python3 -I bin/build-roofline.py      (exit 1 if an anchor is missing)
then preview, commit and push.
"""
import json
import re
import sys
from pathlib import Path

SRC = Path.home() / "code/diffusion-roofline/site/block-diffusion-rooflines.html"
ROOT = Path(__file__).resolve().parent.parent
SLUG = "block-diffusion-rooflines-6f924020"
BOOK = ROOT / "notes" / SLUG
BASE = f"/notes/{SLUG}/"
TITLE = "Block Diffusion Rooflines"
# (page slug, part number, title, source element id, one-line description); "" is the first page.
CHAPTERS = [
    ("", "0", "Introduction", "top", "the question, a quiz, the headline numbers, and the contents."),
    ("summary", "", "Summary", "tldr", "the symbols, what moves the gain, and the three regimes that matter in practice."),
    ("setup", "1", "Setup", "setup", "notation, costs, intensities, penalties."),
    ("gain", "2", "The gain", "gain", "Proposition 1, and the split of each intensity into a linear part and an attention part."),
    ("linear", "3", "The linear part", "linear", "the MoE penalty, when the linear layers dominate, and the gain in that regime."),
    ("attention", "4", "The attention part", "attention", "dense, windowed and sparse attention; the selection penalty as the twin of the MoE penalty; when attention dominates, and the gain there."),
    ("whole", "5", "The whole picture", "whole", "when TD wins, the batch needed to saturate compute, the eight regimes, the exact maps."),
    ("calculator", "6", "Calculator", "calc", "any accelerator, model, attention layout and decoding setup, with the exact accounting."),
    ("examples", "7", "Worked examples", "examples", "a smaller canvas, a dense model, a low-ridge accelerator."),
    ("appendix", "A", "Appendix", "appendix", "the exact accounting, a check against measurements, sources."),
]
ASSETS = ROOT / "assets/roofline"
FIGS = ROOT / "_roofline/figures.html"
CARDS = ROOT / "_roofline/cards.html"
GISTS = ROOT / "_roofline/gists.html"

# Note tokens that do not exist on the site; --ink, --rule and --accent are the site's own.
TOKENS = ["ink-2", "bg", "surface", "muted", "accent-soft", "ar", "td", "critical", "div-neg", "div-mid",
          "div-pos", "int-mem", "int-comp", "font-body", "font-ui", "font-mono", "s3", "s4"]
TOKEN_RE = re.compile(r"--(" + "|".join(re.escape(t) for t in sorted(TOKENS, key=len, reverse=True)) + r")(?![\w-])")
# Class names that the site's CSS (Tailwind, al-folio) also defines.
CLASSES = {"card": "bdr-card", "col": "bdr-col", "outline": "bdr-outline", "title": "bdr-title", "toc": "bdr-toc"}

problems = []


def need(text, old, what):
    n = text.count(old)
    if n != 1:
        problems.append(f"{what}: expected 1 occurrence, found {n}: {old[:90]!r}")
    return n == 1


def sub_once(text, old, new, what):
    return text.replace(old, new) if need(text, old, what) else text


def rename_tokens(s):
    return TOKEN_RE.sub(lambda m: "--bdr-" + m.group(1), s)


def rename_class_attr(s):
    def fix(m):
        return m.group(1) + " ".join(CLASSES.get(c, c) for c in m.group(2).split(" ")) + m.group(3)
    return re.sub(r'(class=\\?")([^"$\\]*)(\\?")', fix, s)


# ---------------------------------------------------------------- CSS
DROP_SEL = re.compile(r"^(html|body|\*|h1|h2|h3|p|a|:root.*|html.*)$|\.rail|\.topnav|\.layout\b|\.progress|\.theme-btn"
                      r"|\.to-top|\.main\b|\.page\b|\.eyebrow|\.hero h1")


def css_blocks(css):
    """Split CSS into (prelude, body) pairs at the top level."""
    out, i = [], 0
    while True:
        j = css.find("{", i)
        if j < 0:
            break
        depth, k = 1, j + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        out.append((css[i:j].strip(), css[j + 1:k - 1]))
        i = k
    return out


def scope_css(css):
    res = []
    for pre, body in css_blocks(css):
        if pre.startswith("@media print"):
            continue
        if pre.startswith("@media"):
            inner = scope_css(body)
            if inner.strip():
                res.append(f"{pre} {{\n{inner}}}")
            continue
        keep = []
        for sel in (s.strip() for s in pre.split(",")):
            if DROP_SEL.search(sel):
                continue
            for old, new in CLASSES.items():
                sel = re.sub(r"\." + old + r"(?![\w-])", "." + new, sel)
            keep.append(".bdr " + sel)
        if keep:
            res.append(", ".join(keep) + " {" + rename_tokens(body) + "}")
    return "\n".join(res) + "\n"


# ---------------------------------------------------------------- figures markup
def figures():
    text = FIGS.read_text()
    blocks = re.split(r"<!-- fig:([\w-]+) -->\n", text)
    return {blocks[i]: blocks[i + 1].rstrip() + "\n" for i in range(1, len(blocks), 2)}


def add_cards(body):
    """Insert each card's "more" panel just before the card's closing tag."""
    blocks = re.split(r"<!-- card:(.+?) -->\n", CARDS.read_text())
    for i in range(1, len(blocks), 2):
        prefix, more = blocks[i], blocks[i + 1].rstrip()
        pat = re.compile(r'(<div class="card[^"]*"><div class="card-k">' + re.escape(prefix) + r'.*?</p>)(</div>)', re.S)
        body, n = pat.subn(lambda m: m.group(1) + "\n" + more + "\n" + m.group(2), body, count=1)
        if n != 1:
            problems.append(f"card {prefix!r} not found")
    return body


# An AR or TD quantity in TeX: a letter or macro, optional ^\star, then _{AR} or ^{AR} with an
# optional subscript (same for TD).
_TOK = r"(?:\\[a-zA-Z]+|[A-Za-z])(?:\^\\star)?"
_SUB = r"(?:_\{(?:[^{}]|\{[^{}]*\})*\}|_[A-Za-z0-9])"
QTY = re.compile(_TOK + r"(?:_\{(AR|TD)\}|\^\{(AR|TD)\}" + _SUB + r"?)")
MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", re.S)


def color_math(body):
    def qty(m):
        side = (m.group(1) or m.group(2)).lower()
        return f"\\class{{mj-{side}}}{{{m.group(0)}}}"
    def seg(m):
        return QTY.sub(qty, m.group(0))
    return MATH.sub(seg, body)


TAKEAWAY_FIGS = [("fig-reuse", "Figure 1"), ("fig-bestbatch", "Figure 3"), ("fig-context", "Figure 4"), ("fig-accel", "Figure 6")]


def link_takeaways(body):
    """Each of the four takeaways ends with a link to the figure that shows it."""
    m = re.search(r'<div class="takeaways">.*?</ol>', body, re.S)
    if not m or m.group(0).count("</li>") != len(TAKEAWAY_FIGS):
        problems.append("takeaways block not found or not four items")
        return body
    parts = m.group(0).split("</li>")
    for i, (fid, name) in enumerate(TAKEAWAY_FIGS):
        parts[i] += f' <a class="tk-fig" href="#{fid}">Try it in {name}</a>'
    return body.replace(m.group(0), "</li>".join(parts))


def add_gists(body):
    blocks = re.split(r"<!-- gist:([\w-]+) -->\n", GISTS.read_text())
    for i in range(1, len(blocks), 2):
        bid, gist = blocks[i], blocks[i + 1].strip()
        pat = re.compile(r'(<div class="box [^"]*" id="' + re.escape(bid) + r'">\s*<div class="title">.*?</div>)', re.S)
        body, n = pat.subn(lambda m: m.group(1) + f'\n    <p class="gist"><span>In short.</span> {gist}</p>', body, count=1)
        if n != 1:
            problems.append(f"gist box {bid!r} not found")
    return body


def main():
    src = SRC.read_text()
    figs = figures()

    # CSS
    css = src[src.index("<style>") + 7: src.index("</style>")]
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    (ASSETS / "note.css").write_text("/* Generated by bin/build-roofline.py from the diffusion-roofline note. Do not edit. */\n" + scope_css(css))

    # HTML body of the note
    body = src[src.index('<main class="main">') + len('<main class="main">'): src.index("</main>")]
    body = re.sub(r'\s*<div class="eyebrow">.*?</div>', "", body, count=1)
    body = re.sub(r"\s*<h1>.*?</h1>", "", body, count=1)
    body = sub_once(body, "and the site lives in the private GitHub repository <code>vdeborto/diffusion-roofline</code>, whose history is the changelog.",
                    "and the page has no other dependency.", "private repo sentence")
    # Interactive figures, in reading order. Figure numbers follow the page.
    body = sub_once(body, '  <div class="formula-card">', figs["quiz"] + '  <div class="formula-card">', "quiz anchor")
    body = sub_once(body, '  <div class="takeaways">', figs["accounting"] + '  <div class="takeaways">', "accounting anchor")
    body = link_takeaways(body)
    body = sub_once(body, '  <div class="table-wrap"><table class="notation">', figs["reuse"] + '  <div class="table-wrap"><table class="notation">', "reuse anchor")
    body = sub_once(body, '  <div class="box prop" id="cor1">', figs["roofline"] + '  <div class="box prop" id="cor1">', "roofline anchor")
    old_fig1 = re.search(r'  <figure class="mapfig">\n    <figcaption>Figure 1\. .*?</figure>\n', body, re.S)
    if old_fig1:
        body = body.replace(old_fig1.group(0), figs["bestbatch"])
    else:
        problems.append("old Figure 1 block not found")
    body = sub_once(body, '  <div class="table-wrap breakout"><table class="cls t6">', figs["context"] + '  <div class="table-wrap breakout"><table class="cls t6">', "context anchor")
    body = sub_once(body, "<figcaption>Figure 2. Smallest compute-bound batch", "<figcaption>Figure 5. Smallest compute-bound batch", "Figure 2 caption")
    body = sub_once(body, "  <h3>Across accelerators and models</h3>\n", "  <h3>Across accelerators and models</h3>\n" + figs["accel"], "accel anchor")
    body = add_cards(body)
    body = add_gists(body)
    body = sub_once(body, '  <div class="tiles hero-tiles">', '  <p class="math-key">In the equations, <span class="q-ar">AR quantities are in blue</span> and <span class="q-td">TD quantities in orange</span>, as in the figures. Select any formula to see what its symbols mean.</p>\n  <div class="tiles hero-tiles">', "math key anchor")
    body = color_math(body)
    body = rename_class_attr(rename_tokens(body))

    # JS: the cost model, the note's figures and the calculator
    tail = src[src.index("</main>"):]
    scripts = re.findall(r"<script>(.*?)</script>", tail, re.S)
    if len(scripts) != 2:
        problems.append(f"expected 2 scripts after </main>, found {len(scripts)}")
    js = "\n".join(scripts)
    js = sub_once(js, "    initTheme();\n", "", "initTheme call")
    js = sub_once(js, "    initRail();\n", "", "initRail call")
    js = sub_once(js, "    rooflineIntuition($('roof-int'), REF.hw, REF.o, REF.w);\n", "", "Figure 1 call")
    js = sub_once(js, ".box[id] > .title, section.wide[id] > .col > h2", ".box[id] > .bdr-title, section.wide[id] > .bdr-col > h2", "anchor selector")
    js = sub_once(js, "  init();\n})();", "  window.BDR = { HW, HWBYID, REF, boe, omega, fmt, comma, kTok, css, model };\n  init();\n})();", "expose internals")
    # Each page holds one chapter: a missing element is a detached stand-in, the maps and charts
    # skip elements that are not on the page, and the calculator is wired only where it is.
    js = sub_once(js, "  const $ = (id) => document.getElementById(id);\n",
                  "  const GHOST = {}; // stand-ins for elements that live on another page of the book\n"
                  "  const $ = (id) => document.getElementById(id) || GHOST[id] || (GHOST[id] = document.createElement('div'));\n", "$ helper")
    for fn in ("intensityMap", "lineChart", "regimeMap", "bstarChart", "ratioChart", "ctxChart", "etaChart"):
        js, n = re.subn(r"(  function " + fn + r"\(el\b[^)]*\) \{\n)", r"\1    if (!el.isConnected) return;\n", js)
        if n != 1:
            problems.append(f"chart function {fn}: found {n}")
    js = sub_once(js, "  function init() {\n", """  function init() {
    if (!document.getElementById('calc')) {
      initTabs(); initAnchors(); staticFigures(); initPrint();
      matchMedia('(prefers-color-scheme: dark)').addEventListener('change', staticFigures);
      new MutationObserver(staticFigures).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
      const land = () => { const t = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1))); if (t) t.scrollIntoView({ behavior: 'instant' }); };
      if (window.MathJax && MathJax.startup && MathJax.startup.promise) MathJax.startup.promise.then(() => setTimeout(land, 50));
      return;
    }
""", "init guard")
    js = rename_class_attr(rename_tokens(js))
    (ASSETS / "note.js").write_text("// Generated by bin/build-roofline.py from the diffusion-roofline note. Do not edit.\n" + js)

    for leak in ("diffusion-roofline", "vendor/", 'href="index.html"'):
        if leak in body:
            problems.append(f"page still contains {leak!r}")

    if problems:
        for p in problems:
            print("PROBLEM:", p)
        return 1
    pages = split_book(body)
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        return 1
    total = 0
    for slug, text in pages.items():
        path = BOOK / slug / "index.html" if slug else BOOK / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        total += path.stat().st_size
    print(f"wrote {len(pages)} pages under {BOOK.relative_to(ROOT)}/ ({total:,} bytes), assets/roofline/note.css, note.js")
    return 0


def url_of(slug):
    return BASE + (slug + "/" if slug else "")


def label_of(ch):
    slug, num, title = ch[:3]
    return f"{num} · {title}" if num and slug else title


def split_book(body):
    """Cut the note at its header and top-level sections, one page per entry of CHAPTERS."""
    chunks = {}
    m = re.search(r'<header class="bdr-col hero">.*?</header>', body, re.S)
    if m:
        chunks["top"] = m.group(0)
    for m in re.finditer(r'<section class="[^"]*" id="([\w-]+)">.*?\n</section>', body, re.S):
        chunks[m.group(1)] = m.group(0)
    want = [c[3] for c in CHAPTERS]
    if sorted(chunks) != sorted(want):
        problems.append(f"sections {sorted(chunks)} do not match CHAPTERS {sorted(want)}")
        return {}

    # The first page: the header, its outline replaced by the book's contents.
    contents = '  <nav class="book-toc" id="contents" aria-label="Contents">\n    <h2 class="book-toc-title">Contents</h2>\n    <ol>\n'
    for ch in CHAPTERS[1:]:
        slug, num, title, _, what = ch
        part = f"Part {num}" if num.isdigit() else ("Appendix" if num else "Before Part 1")
        contents += f'      <li><a href="{url_of(slug)}"><span class="toc-part">{part}</span><span class="toc-title">{title}</span><span class="toc-what">{what[0].upper() + what[1:]}</span></a></li>\n'
    contents += "    </ol>\n  </nav>\n"
    top = chunks["top"]
    # The contents come right after the quiz, where the note had its outline at the end of the header.
    top, n = re.subn(r'\s*<h2 class="outline-title" id="outline">.*?</ol>\n', "\n", top, count=1, flags=re.S)
    if n != 1:
        problems.append("outline not found")
    top = sub_once(top, '  <div class="formula-card">', contents + '  <div class="formula-card">', "contents anchor")
    chunks["top"] = top

    # Which page holds each id; links to another page go there.
    home = {}
    for ch in CHAPTERS:
        for i in re.findall(r'\bid="([\w-]+)"', chunks[ch[3]]):
            home.setdefault(i, ch)
        home[ch[3]] = ch
    home["outline"] = CHAPTERS[0]
    for ch in CHAPTERS:
        src = ch[3]
        def relink(m, ch=ch):
            i = m.group(1)
            t = home.get(i)
            if t is None:
                problems.append(f"link to #{i} from {ch[0] or 'first page'}: no such id")
                return m.group(0)
            if t is ch:
                return m.group(0)
            return f'href="{url_of(t[0])}' + ("" if i == t[3] else f"#{i}") + '"'
        chunks[src] = re.sub(r'href="#([\w-]+)"', relink, chunks[src])

    book = {"title": TITLE, "home": BASE, "chapters": [{"slug": c[0], "num": c[1], "title": c[2], "url": url_of(c[0])} for c in CHAPTERS]}
    pages = {}
    for k, ch in enumerate(CHAPTERS):
        slug, num, title, src, what = ch
        prev, nxt = (CHAPTERS[k - 1] if k else None), (CHAPTERS[k + 1] if k + 1 < len(CHAPTERS) else None)
        text = chunks[src]
        if slug:
            # The page title replaces the section heading.
            text = re.sub(r"\s*<h2>.*?</h2>", "", text, count=1, flags=re.S)
            part = f"Part {num} of" if num.isdigit() else ("The appendix of" if num == "A" else "The summary of")
            links = " | ".join(f'<a href="{url_of(c[0])}" rel="{r}">{w}: {label_of(c)}</a>' for c, r, w in ((prev, "prev", "Previous"), (nxt, "next", "Next")) if c)
            head = f'<p class="book-part">{part} <a href="{BASE}">{TITLE}</a> <span class="book-links">({links})</span></p>\n'
            foot = '<nav class="book-nav" aria-label="Chapters">' + "".join(
                f'<a class="{r}" href="{url_of(c[0])}"><span>{w}</span>{label_of(c)}</a>' for c, r, w in ((prev, "prev", "Previous"), (nxt, "next", "Next")) if c) + "</nav>\n"
            desc = what[0].upper() + what[1:]
        else:
            head = ""
            foot = f'<nav class="book-nav" aria-label="Chapters"><a class="next" href="{url_of(nxt[0])}"><span>Start reading</span>{label_of(nxt)}</a></nav>\n'
            desc = "when block text diffusion beats autoregressive decoding · a short book on its inference cost"
        meta = dict(book, current=slug)
        if not slug:  # forwards old single-page anchors and calculator links
            meta["ids"] = {i: c[0] for i, c in home.items() if c[0]}
        front = f"""---
layout: page
title: {title if slug else TITLE}
description: "{desc.replace('"', "'")}"
permalink: {url_of(slug)}
robots: noindex, nofollow
sitemap: false
---
<!-- Generated by bin/build-roofline.py. Do not edit by hand. -->
<link rel="stylesheet" href="/assets/roofline/note.css">
<link rel="stylesheet" href="/assets/roofline/theme.css">
<script type="application/json" id="bdr-book">{json.dumps(meta, separators=(",", ":"))}</script>
<script defer src="/assets/roofline/mathjax.js"></script>
<script defer src="/assets/roofline/note.js"></script>
<script defer src="/assets/roofline/figures.js"></script>
<script defer src="/assets/roofline/reading.js"></script>
{{% raw %}}
<div class="bdr{' book-chapter' if slug else ''}" id="top">
<noscript><p>This page needs JavaScript for its equations, figures and calculator.</p></noscript>
"""
        pages[slug] = front + head + text.strip() + "\n" + foot + "</div>\n{% endraw %}\n"
    return pages


if __name__ == "__main__":
    sys.exit(main())
