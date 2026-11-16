#!/usr/bin/env python3
"""Build the website version of the Block Diffusion Rooflines note.

Source of truth for the text, the cost model and the calculator: the private repo
~/code/diffusion-roofline (site/block-diffusion-rooflines.html). This script turns that standalone
page into a page of this Jekyll site, in the site's own layout and palette, and adds the
interactive figures kept in this repo:

  _roofline/figures.html        markup of the interactive figures, one <!-- fig:NAME --> block each
                                (main() below places each one at an anchor of the note's text)
  _roofline/cards.html          "more" panels of the TL;DR cards, one <!-- card:TITLE PREFIX --> block each
  _roofline/gists.html          one-sentence summary of each definition and proposition box
  _roofline/intros.html         opening paragraph of each chapter page
  bin/roofline_practice.py      figures and tables of the practice chapters (Parts 0, 8-11), from
                                _roofline/practice-data.json (exported by roofscope's pages/book/export.py)
  assets/roofline/reading.js    contents rail, reading progress, the intuition / full-math switch, the
                                steppers, the symbol key, the citation's copy button and the comment thread
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

Every chapter ends with a comment thread (the citation block, cite_block, is off: CITATION)
(comments_block): giscus, one GitHub Discussion per chapter in the repository named by GISCUS. It
needs Discussions enabled there and the giscus app installed on it (https://github.com/apps/giscus).

Every page is unlisted: no link from the site, robots noindex (front matter `robots`), excluded
from the sitemap (_config.yml defaults for notes/). Usage:
    python3 -I bin/build-roofline.py      (exit 1 if an anchor is missing)
then preview, commit and push.
"""
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mdlite  # Markdown on top of the HTML of the text files (bin/mdlite.py)  # noqa: E402

SRC = Path.home() / "code/diffusion-roofline/site/block-diffusion-rooflines.html"
ROOT = Path(__file__).resolve().parent.parent
SLUG = "block-diffusion-rooflines-6f924020"
BOOK = ROOT / "notes" / SLUG
BASE = f"/notes/{SLUG}/"
TITLE = "Block Diffusion Rooflines"
# (page slug, part number, title, source element id, one-line description); "" is the first page.
CHAPTERS = [
    ("", "0", "Introduction", "intro", "what is already measured, the question, the measurement setting, how the book is organised and how to read it."),
    ("summary", "", "Summary", "tldr", "the headline numbers, a quiz, four takeaways, the symbols, what moves the gain, and the three regimes that occur."),
    ("setup", "1", "Setup", "setup", "notation, costs, intensities, penalties."),
    ("gain", "2", "The gain", "gain", "Proposition 1, and the split of each intensity into a linear part and an attention part."),
    ("linear", "3", "The linear part", "linear", "the expert and linear penalties, when the linear layers dominate, and the gain in that regime."),
    ("attention", "4", "The attention part", "attention", "dense, windowed and sparse attention; the selection penalty as the twin of the expert penalty; when attention dominates, and the gain there."),
    ("whole", "5", "The whole picture", "whole", "when TD wins, the batch needed to saturate compute, the eight regimes, other accelerators and models, the exact maps."),
    ("calculator", "6", "Calculator", "calc", "any accelerator, model, attention layout and decoding setup, with the exact accounting."),
    ("examples", "7", "Worked examples", "examples", "a smaller canvas, a dense model, a low-ridge accelerator."),
    ("effective", "8", "Empirical rooflines", "effective", "the ceilings one H200 sustains, the experts each decoder actually touches, and the bound they give."),
    ("stack", "9", "The measurement stack", "stack", "how every kernel of a step is timed and attributed, three defects of the stock engine and one of the harness, and where the stock kernels stand."),
    ("kernels", "10", "Kernels", "kernelwork", "two rewrites, of the experts and of global attention, and what they change in efficiency, throughput and quality."),
    ("conclusion", "11", "Conclusion", "conclusion", "the bound against the measurement, what the roofline gets right, limitations, code and data, related work, open problems."),
    ("appendix", "A", "Appendix", "appendix", "the exact accounting, a check against measurements, sources."),
]
# Citation, and the comment threads (ids from https://giscus.app for the repository and category).
CITATION = False  # the citation block at the end of the first page; removed on 2026-11-16, True brings it back
AUTHOR = ("De Bortoli", "Valentin")
YEAR = "2026"
SITE = "https://vdeborto.github.io"
GISCUS = {"repo": "vdeborto/vdeborto.github.io", "repo-id": "MDEwOlJlcG9zaXRvcnkxMTY1OTk1ODA=",
          "category": "Announcements", "category-id": "DIC_kwDOBvMrHM4DLMnm"}
ASSETS = ROOT / "assets/roofline"
FIGS = ROOT / "_roofline/figures.html"
CARDS = ROOT / "_roofline/cards.html"
GISTS = ROOT / "_roofline/gists.html"
INTROS = ROOT / "_roofline/intros.html"

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
    return {blocks[i]: mdlite.render(blocks[i + 1].rstrip()) + "\n" for i in range(1, len(blocks), 2)}


def add_cards(body):
    """Insert each card's "more" panel just before the card's closing tag."""
    blocks = re.split(r"<!-- card:(.+?) -->\n", CARDS.read_text())
    for i in range(1, len(blocks), 2):
        prefix, more = blocks[i], mdlite.render(blocks[i + 1].rstrip())
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


TAKEAWAY_FIGS = [("fig-reuse", "Figure 5"), ("fig-bestbatch", "Figure 10"), ("fig-context", "Figure 14"), ("fig-accel", "Figure 17")]


def link_takeaways(body):
    """Each of the four takeaways ends with a link to the figure that shows it; the figure's number is
    filled in by number_figures once the numbering is known (placeholder {fig:ID})."""
    m = re.search(r'<div class="takeaways">.*?</ol>', body, re.S)
    if not m or m.group(0).count("</li>") != len(TAKEAWAY_FIGS):
        problems.append("takeaways block not found or not four items")
        return body
    parts = m.group(0).split("</li>")
    for i, (fid, name) in enumerate(TAKEAWAY_FIGS):
        parts[i] += f' <a class="tk-fig" href="#{fid}">Try it in {name}</a>'
    return body.replace(m.group(0), "</li>".join(parts))


def add_practice(body):
    """Figures and tables of the practice chapters (Parts 0, 8-11): each <!-- pfig:NAME --> in the
    note is replaced by bin/roofline_practice.py's block NAME."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import roofline_practice
    blocks = roofline_practice.blocks()
    used = set()
    def put(m):
        name = m.group(1)
        if name not in blocks:
            problems.append(f"practice figure {name!r} unknown")
            return m.group(0)
        used.add(name)
        return blocks[name]
    body = re.sub(r"[ \t]*<!-- pfig:([\w-]+) -->\n?", put, body)
    for name in sorted(set(blocks) - used):
        problems.append(f"practice figure {name!r} not placed")
    return body


def add_code(body):
    """Code cards (Parts 8-11): each <!-- code:NAME --> in the note is replaced by
    bin/roofline_code.py's card NAME, an excerpt of roofscope at a pinned commit."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import roofline_code
    blocks = roofline_code.blocks()
    used = set()
    def put(m):
        name = m.group(1)
        if name not in blocks:
            problems.append(f"code card {name!r} unknown")
            return m.group(0)
        used.add(name)
        return blocks[name]
    body = re.sub(r"[ \t]*<!-- code:([\w-]+) -->\n?", put, body)
    for name in sorted(set(blocks) - used):
        problems.append(f"code card {name!r} not placed")
    return body


def add_gists(body):
    blocks = re.split(r"<!-- gist:([\w-]+) -->\n", GISTS.read_text())
    for i in range(1, len(blocks), 2):
        bid, gist = blocks[i], mdlite.render_inline(blocks[i + 1].strip())
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
    body = mdlite.render(body)
    body = sub_once(body, "The page is self-contained: the cost model is the <code>BDModel</code> script embedded in it (view the page source), and the site lives in the private GitHub repository <code>vdeborto/diffusion-roofline</code>, whose history is the changelog.",
                    "The cost model is the <code>BDModel</code> script in <code>/assets/roofline/note.js</code>, which every chapter loads.", "private repo sentence")
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
        problems.append("old Figure 5 block not found")
    body = sub_once(body, '  <div class="table-wrap breakout"><table class="cls t6">', figs["context"] + '  <div class="table-wrap breakout"><table class="cls t6">', "context anchor")
    body = sub_once(body, "<figcaption>Figure 3. Smallest compute-bound batch", "<figcaption>Figure 15. Smallest compute-bound batch", "Figure 15 caption")
    body = sub_once(body, "  <h3>Across accelerators and models</h3>\n", "  <h3>Across accelerators and models</h3>\n" + figs["accel"], "accel anchor")
    # Parts 3-5: experts and weights (Proposition 3), where the weights dominate (Table 2), what a pass reads of
    # the cache (Table 4), the sparse penalty (Proposition 5), attention's shares (Table 5), thresholds (Table 8).
    # A figure whose numbers are in a folded table goes before the fold, so that it is shown.
    for name, anchor in (("experts", "  <p>For DiffusionGemma, \\(E/k = 16\\)"),
                         ("weights", '  <div class="box prop" id="prop4">'),
                         ("linmap", '  <details class="numbers"><summary>Table 2: '),
                         ("attnread", '  <div class="box prop" id="prop5">'),
                         ("sparse", '  <div class="box prop" id="prop6">'),
                         ("share", '  <details class="numbers"><summary>Table 5: '),
                         ("thresholds", '  <details class="numbers"><summary>Table 8: '),
                         ("models", '  <details class="numbers"><summary>Table 9: ')):
        body = sub_once(body, anchor, figs[name] + anchor, f"{name} anchor")
    body = card_legacy_figures(body)
    body = add_practice(body)
    body = add_code(body)
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
    js = sub_once(js, "    rooflineIntuition($('roof-int'), REF.hw, REF.o, REF.w);\n", "", "Figure 5 call")
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
    global MEAS
    MEAS = measured_gains()
    pages = split_book(body)
    # Figures and tables are numbered in reading order: 1, 2, 3, ... across the chapters.
    for kind, rx in (("Figure", r'(?:<div class="fig-k">|<figcaption>)Figure (\d+)'), ("Table", r"<caption>Table (\d+)\.")):
        seen = [int(n) for ch in CHAPTERS for n in re.findall(rx, pages.get(ch[0], ""))]
        if seen != list(range(1, len(seen) + 1)):
            problems.append(f"{kind} numbers not in reading order: {seen}")
    # The takeaways name their figures by number: check the names against the numbering.
    alltext = "".join(pages.values())
    for fid, name in TAKEAWAY_FIGS:
        m = re.search(rf'id="{fid}".*?<div class="fig-k">(Figure \d+)', alltext, re.S)
        if not m or m.group(1) != name:
            problems.append(f"takeaway link {fid} says {name}, the figure is {m.group(1) if m else 'missing'}")
    pages = {k: name_headings(v) for k, v in pages.items()}
    pages = link_references(pages)
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        return 1
    total = 0
    for slug, text in pages.items():
        # The front matter is YAML: undo any cross-reference link that landed in its description.
        if text.startswith("---\n"):
            end = text.index("\n---\n", 4)
            front = re.sub(r'<a class="xref" href="[^"]*">(.*?)</a>', r"\1", text[:end])
            if '<' in front or front.count('"') % 2:
                problems.append(f"front matter of {slug or 'first page'} is not plain text")
            text = front + text[end:]
        path = BOOK / slug / "index.html" if slug else BOOK / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        total += path.stat().st_size
    print(f"wrote {len(pages)} pages under {BOOK.relative_to(ROOT)}/ ({total:,} bytes), assets/roofline/note.css, note.js")
    return 0


def url_of(slug):
    return BASE + (slug + "/" if slug else "")


# The note's own static figures (the maps and the batch chart) get the same card as the figures
# of figures.js: a kicker with the number, a title that states the figure's claim, and the
# original caption as a subtitle.
LEGACY_CLAIMS = {
    15: "As the context grows, the batch at which a pass turns compute-bound falls for a canvas of 256 and rises for AR (Proposition 8).",
    # Figure 20 has four tabs, each with its own claim; figures.js shows the title of the selected tab.
    20: ('<span data-tab-title="m-gain">At context 1k AR catches up with TD at batch 110, and at a larger batch as the context grows, beyond 1,024 from 64k tokens on.</span>'
         '<span data-tab-title="m-ar" hidden>An AR step never reaches the ridge, and at long context its intensity tends to 8 whatever the batch.</span>'
         '<span data-tab-title="m-td" hidden>A TD canvas crosses the ridge at a batch that falls as the context grows, from 5 at 1k to 1 at 256k, and its intensity tends to 2,048.</span>'
         '<span data-tab-title="m-sparse" hidden>With sparse global layers, AR catches up with TD near batch 90–110 at every context up to 1M tokens.</span>'),
    21: "With a 32-token canvas, TD is faster than AR only between batch 9 and batch 110 at context 1k.",
    22: "For a dense model, the gain at context 1k is about 13 at batch 1 and falls like 1/B, to parity at batch 15.",
    23: "On an H20, whose ridge point is 37, TD is faster than AR only up to batch 4 at short context.",
}


def card_legacy_figures(body):
    def one(m):
        n, rest = int(m.group(3)), m.group(4)
        if n not in LEGACY_CLAIMS:
            problems.append(f"no claim for legacy Figure {n}")
            return m.group(0)
        return (f'{m.group(1)}<div class="bdr-card">\n    <div class="fig-k">Figure {n}</div>\n    <p class="fig-title">{LEGACY_CLAIMS[n]}</p>\n'
                f'  {m.group(2)}\n    <p class="fig-sub">{rest}</p>')
    body, k = re.subn(r'( *)(<figure class="mapfig">)\n\s*<figcaption>Figure (\d+)\. (.*?)</figcaption>', one, body)
    # close the card after each converted figure
    out, pos = [], 0
    for mm in re.finditer(r'<div class="bdr-card">.*?</figure>', body, re.S):
        out.append(body[pos:mm.end()] + "\n  </div>"); pos = mm.end()
    body = "".join(out) + body[pos:]
    tabs = '<div class="fig-k">Figure 20 · interactive</div>'
    if tabs in body:
        body = body.replace(tabs, f'{tabs}\n  <p class="fig-title tabs-title">{LEGACY_CLAIMS[20]}</p>', 1)
    else:
        problems.append("Figure 20 kicker not found")
    if k != 4:
        problems.append(f"legacy figures carded: {k}, expected 4")
    return body


def name_headings(html):
    """Give every h3 without an id one made from its text, so that the rail lists it and it can be linked."""
    used = set(re.findall(r'id="([^"]+)"', html))
    def one(m):
        text = re.sub(r"<[^>]+>|\\\(.*?\\\)", "", m.group(2))
        base = "s-" + (re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "section")
        k, i = base, 2
        while k in used:
            k, i = f"{base}-{i}", i + 1
        used.add(k)
        return f'<h3{m.group(1)} id="{k}">{m.group(2)}</h3>'
    return re.sub(r'<h3((?:\s+(?!id=)[\w-]+="[^"]*")*)>(.*?)</h3>', one, html)


def link_references(pages):
    """Anchor every figure and table label (#figure-N, #table-N) and link each "Figure N" or
    "Table N" in the prose to it, on its own page or across pages. Text inside links, scripts,
    SVG, captions of the target itself and figure labels is left alone."""
    where = {}
    lab = {"Figure": re.compile(r'(<div class="fig-k")(>Figure (\d+))'), "Table": re.compile(r'(<caption)(>Table (\d+)\.)')}
    for slug in pages:
        for kind, rx in lab.items():
            def put(m, kind=kind, slug=slug):
                n = int(m.group(3)); where[(kind, n)] = slug
                return f'{m.group(1)} id="{kind.lower()}-{n}"{m.group(2)}'
            pages[slug] = rx.sub(put, pages[slug])
    # numbered statements: the box that carries the tag, wherever it is
    for slug in pages:
        for m in re.finditer(r'id="([\w-]+)"[^>]*>\s*<(?:summary|div class="bdr-title"|div class="title")>\s*<span class="tag">(Proposition|Definition|Corollary|Remark) (\d+)</span>', pages[slug]):
            where[(m.group(2), int(m.group(3)))] = (slug, m.group(1))
    for ch in CHAPTERS:
        if ch[1].isdigit():
            where[("Part", int(ch[1]))] = (ch[0], None)
    ref = re.compile(r'\b(Figures?|Tables?|Propositions?|Definitions?|Corollary|Remarks?|Parts?)((?:\s|&nbsp;)\d+(?:(?:\s*(?:–|,|and|or)\s*)\d+)*)')
    num = re.compile(r'\d+')
    skip_open = re.compile(r'<(a|script|svg|style|caption|button|summary|title|h1|h2|nav|pre)\b', re.I)
    for slug, html in pages.items():
        parts = re.split(r'(<[^>]+>)', html)
        stack, out = [], []
        for part in parts:
            if part.startswith('<'):
                m = re.match(r'<(/?)([a-zA-Z][\w-]*)', part)
                if m:
                    name = m.group(2).lower()
                    if m.group(1) and stack and stack[-1] == name:
                        stack.pop()
                    elif not m.group(1) and skip_open.match(part) and not part.endswith('/>'):
                        stack.append(name)
                    elif not m.group(1) and name == 'div' and ('class="fig-k"' in part or 'class="mn-h"' in part):
                        stack.append('div')
                    elif not m.group(1) and name == 'span' and 'class="tag"' in part:
                        stack.append('span')
                    elif m.group(1) and name in ('div', 'span') and stack and stack[-1] == name:
                        stack.pop()
                out.append(part)
                continue
            if stack or not ref.search(part):
                out.append(part); continue
            def href_of(kind, n):
                tgt = where.get((kind, n))
                if tgt is None:
                    return None
                if isinstance(tgt, tuple):  # (page, element id) or a chapter (page, None)
                    page, anchor = tgt
                    if anchor is None:
                        return None if page == slug else url_of(page)
                    return f"#{anchor}" if page == slug else f"{url_of(page)}#{anchor}"
                return f"#{kind.lower()}-{n}" if tgt == slug else f"{url_of(tgt)}#{kind.lower()}-{n}"
            def link(m):
                word = m.group(1)
                kind = next(k for k in ("Figure", "Table", "Proposition", "Definition", "Corollary", "Remark", "Part") if word.startswith(k))
                nums = num.findall(m.group(2))
                if len(nums) == 1:  # "Figure 27": the whole reference is the link
                    h = href_of(kind, int(nums[0]))
                    return f'<a class="xref" href="{h}">{m.group(0)}</a>' if h else m.group(0)
                def one(mm):
                    h = href_of(kind, int(mm.group(0)))
                    return f'<a class="xref" href="{h}">{mm.group(0)}</a>' if h else mm.group(0)
                # "Parts 3 and 4": the word goes with the first number, so that each target is wide enough
                first = num.search(m.group(2))
                h0 = href_of(kind, int(first.group(0)))
                head = m.group(1) + m.group(2)[:first.end()]
                head = f'<a class="xref" href="{h0}">{head}</a>' if h0 else head
                return head + num.sub(one, m.group(2)[first.end():])
            out.append(ref.sub(link, part))
        pages[slug] = "".join(out)
    return pages


def practice_tags(slug):
    if slug not in PRACTICE_SLUGS:
        return ""
    return f'<script type="application/json" id="bdr-practice">{practice_json()}</script>\n<script defer src="/assets/roofline/measured.js"></script>'


def label_of(ch):
    slug, num, title = ch[:3]
    return f"{num} · {title}" if num and slug else title


def measured_gains():
    """Measured gains of Part 11 for the quiz: {"B,C": [measured, bound (c)]}."""
    d = json.loads((ROOT / "_roofline" / "practice-data.json").read_text())
    return json.dumps({k: [round(v["measured"], 2), round(v["roof"], 2)] for k, v in d["gains"].items()}, separators=(",", ":"))


MEAS = None
PRACTICE_SLUGS = {"", "effective", "stack", "kernels", "conclusion"}


def practice_json():
    """The measurements that assets/roofline/measured.js charts, rounded to 4 digits."""
    d = json.loads((ROOT / "_roofline" / "practice-data.json").read_text())
    r4 = lambda v: round(v, 4) if isinstance(v, float) else v
    clean = lambda x: {k: clean(v) for k, v in x.items()} if isinstance(x, dict) else r4(x)
    keep = {"gains": ("roof_ind_ds", "roof_ind", "roof", "measured", "eta_td", "eta_ar")}
    out = {"Bs": d["Bs"], "Cs": d["Cs"],
           "gains": {k: {kk: v.get(kk) for kk in keep["gains"]} for k, v in d["gains"].items()},
           "eta2": d["eta2"], "eta3": d["eta3"], "share": d["share"], "recoverable": d["recoverable"],
           "vs2": {"td": {k: {"ratio": v["ratio"]} for k, v in d["vs2"]["td"].items()}},
           "prop1": {k: {kk: v[kk] for kk in ("gain_ind", "gain_meas", "Pi_ind", "Pi_meas")} for k, v in d["prop1"]["fp8_empirical"]["pts"].items()},
           "routing": {m: {k: {kk: v[kk] for kk in ("N", "uni", "pop", "meas", "meas_lo", "meas_hi", "neff")} for k, v in d["routing"][m].items()} for m in d["routing"]}}
    return json.dumps(clean(out), separators=(",", ":"))


def bibtex(url):
    return ("@misc{debortoli2026rooflines,\n"
            f"  title = {{{TITLE}}},\n"
            f"  author = {{{AUTHOR[0]}, {AUTHOR[1]}}},\n"
            f"  howpublished = {{\\url{{{url}}}}},\n"
            f"  year = {{{YEAR}}}\n"
            "}")


def cite_block():
    """How to cite the book, at the end of its first page."""
    return (f'<section class="book-cite" id="cite">\n  <h3>Citation</h3>\n'

            f'  <div class="cite-entry"><pre class="cite-text">{AUTHOR[0]}, {AUTHOR[1][0]}. ({YEAR}). {TITLE}. {SITE + BASE}</pre></div>\n'
            f'  <p>or as a BibTeX entry:</p>\n'
            f'  <div class="cite-entry"><button type="button" class="cite-copy">Copy</button><pre class="cite-bib">{html.escape(bibtex(SITE + BASE))}</pre></div>\n'
            f'</section>\n')


def comments_block(ch):
    """The comment thread of a chapter: a GitHub Discussion found by its title (reading.js loads giscus)."""
    slug, num, title = ch[:3]
    term = f"{TITLE} · " + (f"Part {num} · {title}" if num.isdigit() else title)
    g = GISCUS
    return (f'<section class="book-comments" id="bdr-comments" data-repo="{g["repo"]}" data-repo-id="{g["repo-id"]}" data-category="{g["category"]}" '
            f'data-category-id="{g["category-id"]}" data-term="{html.escape(term, quote=True)}">\n'
            f'  <h3>Comments</h3>\n'
            f'  <div class="comments-frame"></div>\n'
            f'  <noscript><p class="small">The comment thread needs JavaScript.</p></noscript>\n'
            f'</section>\n')


def split_book(body):
    """Cut the note at its header and top-level sections, one page per entry of CHAPTERS."""
    chunks = {}
    m = re.search(r'<header class="bdr-col hero">.*?</header>', body, re.S)
    if m:
        chunks["top"] = m.group(0)
    for m in re.finditer(r'<section class="[^"]*" id="([\w-]+)">.*?\n</section>', body, re.S):
        chunks[m.group(1)] = m.group(0)
    want = [c[3] for c in CHAPTERS] + ["top"]
    if sorted(chunks) != sorted(want):
        problems.append(f"sections {sorted(chunks)} do not match CHAPTERS {sorted(want)}")
        return {}

    # The book opens on the Introduction; the note's header (quiz, formula, headline numbers,
    # accounting, takeaways) leads the Summary, which states the results first.
    top = chunks.pop("top")
    top, n = re.subn(r'\s*<h2 class="outline-title" id="outline">.*?</ol>\n', "\n", top, count=1, flags=re.S)
    if n != 1:
        problems.append("outline not found")
    top = re.sub(r'\s*<div class="jump">.*?</div>', "", top, count=1, flags=re.S)
    top = re.sub(r'\s*<p class="lede">.*?</p>', "", top, count=1, flags=re.S)
    inner = re.sub(r'^<header[^>]*>|</header>$', "", top.strip()).strip("\n")
    chunks["tldr"] = sub_once(chunks["tldr"], "</h2>", "</h2>\n" + inner + "\n", "summary heading")

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

    def intro_html(t):
        """The intro in Markdown or HTML (bin/mdlite.py): one paragraph is <p class="chapter-intro">,
        several blocks (paragraphs, a list) are a <div class="chapter-intro">."""
        h = mdlite.render(t).strip()
        one = re.fullmatch(r"<p>(.*)</p>", h, re.S)
        if one and "<p" not in one.group(1) and "</p>" not in one.group(1):
            return f'<p class="chapter-intro">{one.group(1)}</p>'
        return f'<div class="chapter-intro">{h}</div>'

    blocks = re.split(r"<!-- intro:([\w-]+) -->\n", INTROS.read_text())
    intros = {blocks[i]: blocks[i + 1].strip() for i in range(1, len(blocks), 2)}
    intros[""] = intros.pop("introduction", "")  # the Introduction is the first page
    for slug in intros:
        if slug not in [c[0] for c in CHAPTERS]:
            problems.append(f"intro for unknown chapter {slug!r}")

    book_defs = mdlite.collect(body)
    for t in intros.values():
        book_defs.update(mdlite.collect(mdlite.render(t)))
    used, missing = set(), []
    book = {"title": TITLE, "home": BASE, "chapters": [{"slug": c[0], "num": c[1], "title": c[2], "url": url_of(c[0])} for c in CHAPTERS]}
    pages = {}
    for k, ch in enumerate(CHAPTERS):
        slug, num, title, src, what = ch
        prev, nxt = (CHAPTERS[k - 1] if k else None), (CHAPTERS[k + 1] if k + 1 < len(CHAPTERS) else None)
        text = chunks[src]
        # The page title replaces the section heading.
        text = re.sub(r"\s*<h2>.*?</h2>", "", text, count=1, flags=re.S)
        if intros.get(slug):
            text = re.sub(r"(<section[^>]*>(?:\s*<div class=\"bdr-col\">)?)", lambda m: m.group(1) + "\n  " + intro_html(color_math(intros[slug])), text, count=1)
        elif not re.search(r"<section[^>]*>\s*<p>", text):
            problems.append(f"chapter {slug!r} has no intro and does not open with a paragraph")
        else:
            text = re.sub(r"(<section[^>]*>\s*)<p>", r'\1<p class="chapter-intro">', text, count=1)
        # one line for the book, one for the previous chapter, one for the next
        links = "".join(f'<span class="bl-row">{w}: <a href="{url_of(c[0])}" rel="{r}">{label_of(c)}</a></span>' for c, r, w in ((prev, "prev", "Previous"), (nxt, "next", "Next")) if c)
        if slug:
            part = f"Part {num} of" if num.isdigit() else ("The appendix of" if num == "A" else "The summary of")
            head = f'<p class="book-part">{part} <a href="{BASE}">{TITLE}</a><span class="book-links">{links}</span></p>\n'
            desc = ""
        else:
            head = f'<p class="book-part">Part 0 · Introduction<span class="book-links">{links}</span></p>\n'
            desc = "when block text diffusion beats autoregressive decoding · a blogpost on its inference cost"
        foot = '<nav class="book-nav" aria-label="Chapters">' + "".join(
            f'<a class="{r}" href="{url_of(c[0])}"><span>{w}</span>{label_of(c)}</a>' for c, r, w in ((prev, "prev", "Previous"), (nxt, "next", "Next")) if c) + "</nav>\n"
        # A chapter opens with its introduction, so the layout's subtitle would repeat it.
        # Chapters get a description for link previews; the layout's subtitle is hidden with _styles.
        desc_line = (f'description: "{f'Part {num} of' if num.isdigit() else f'{title} of'} {TITLE}: {what}"\n_styles: ".post-header .post-description {{ display: none; }}"\n'
                     if slug else f'description: "{desc}"\n')
        meta = dict(book, current=slug)
        if not slug:  # forwards old single-page anchors and calculator links
            meta["ids"] = {i: c[0] for i, c in home.items() if c[0]}
        front = f"""---
layout: page
title: {title if slug else TITLE}
{desc_line}permalink: {url_of(slug)}
robots: noindex, nofollow
sitemap: false
---
<!-- Generated by bin/build-roofline.py. Do not edit by hand. -->
<link rel="stylesheet" href="/assets/roofline/note.css">
<link rel="stylesheet" href="/assets/roofline/theme.css">
<script type="application/json" id="bdr-book">{json.dumps(meta, separators=(",", ":"))}</script>
<script type="application/json" id="bdr-meas">{MEAS}</script>
{practice_tags(slug)}
<script defer src="/assets/roofline/mathjax.js"></script>
<script defer src="/assets/roofline/note.js"></script>
<script defer src="/assets/roofline/figures.js"></script>
<script defer src="/assets/roofline/reading.js"></script>
{{% raw %}}
<div class="bdr book-chapter" id="top">
<noscript><p>This page needs JavaScript for its equations, figures and calculator.</p></noscript>
"""
        # Footnotes: numbered per page; a note may be defined anywhere in the book.
        used.update(re.findall(r'<sup class="fn-ref" data-fn="([\w-]+)">', text))
        text = mdlite.footnotes(text, book_defs, missing)
        cite = cite_block() if CITATION and not slug else ""
        pages[slug] = front + head + text.strip() + "\n" + cite + foot + comments_block(ch) + "</div>\n{% endraw %}\n"
    for lab in sorted(set(missing)):
        problems.append(f"footnote [^{lab}] is used but not defined")
    for lab in sorted(set(book_defs) - used):
        problems.append(f"footnote [^{lab}] is defined but not used")
    pages["introduction"] = f"""---
layout: null
permalink: {url_of("introduction")}
sitemap: false
---
<!doctype html><html><head><meta charset="utf-8"><meta name="robots" content="noindex, nofollow"><link rel="canonical" href="{BASE}"><meta http-equiv="refresh" content="0; url={BASE}"></head><body><a href="{BASE}">{TITLE}</a></body></html>
"""
    return pages


if __name__ == "__main__":
    sys.exit(main())
