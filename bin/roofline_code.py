"""Code cards of the rooflines book: excerpts of the roofscope harness, quoted at a pinned commit.

Each <!-- code:NAME --> in the note is replaced by the card NAME below. A card quotes one or more
line ranges of one file of github.com/vdeborto/roofscope at commit SHA, read with `git show` from
the local clone ~/code/roofscope, so that the quoted lines and the line numbers of the GitHub links
always agree. A range is given by two regular expressions, its first and its last line, so the
cards survive edits elsewhere in a file; the build fails if a pattern is not found.

To move the book to a newer commit: push it, set SHA, rebuild, and check the cards.
"""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

from pygments.lexers import BashLexer, PythonLexer
from pygments.token import STANDARD_TYPES

REPO = "vdeborto/roofscope"
CLONE = Path.home() / "code/roofscope"
SHA = "1b49abb4ba0a24949b1dd7ef6c7dcdbf72612d02"
GH = f"https://github.com/{REPO}"
# While the repository is private its GitHub links would 404 for readers: print the commit and
# line numbers as plain text, and set PUBLIC = True (then rebuild) once it is public.
PUBLIC = False


def _a(href: str, text: str) -> str:
    return f'<a href="{href}">{text}</a>' if PUBLIC else text


def logo(size: int, uid: str, label: bool = False) -> str:
    """The roofscope mark: a lens magnifying the ridge point of a roofline, measured kernels
    below the roof. Colours come from the book's tokens, so it follows the light and dark themes."""
    aria = 'role="img" aria-label="roofscope logo"' if label else 'aria-hidden="true"'
    lens = ""
    if size >= 28:
        lens = (f'<defs><clipPath id="{uid}"><circle cx="30" cy="22" r="8.5"/></clipPath></defs>'
                f'<circle cx="30" cy="22" r="10" fill="var(--bdr-surface)"/>'
                f'<g clip-path="url(#{uid})"><path d="M10 50L30 22H58" fill="none" stroke="var(--bdr-td)" stroke-width="7" stroke-linejoin="round"/></g>')
    else:
        lens = '<circle cx="30" cy="22" r="10" fill="var(--bdr-surface)"/><path d="M16 50L30 22H44" fill="none" stroke="var(--bdr-td)" stroke-width="5" stroke-linejoin="round"/>'
    return (f'<svg class="rs-logo" width="{size}" height="{size}" viewBox="0 0 64 64" {aria}>'
            '<path d="M10 8V54H58" fill="none" stroke="var(--bdr-muted)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
            '<path d="M10 50L30 22H58" fill="none" stroke="var(--bdr-td)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
            '<circle cx="18.5" cy="47" r="3" fill="var(--bdr-ar)"/><circle cx="41" cy="44" r="3" fill="var(--bdr-ar)"/>'
            '<circle cx="53" cy="30.5" r="3" fill="var(--bdr-ar)"/>'
            + lens +
            '<circle cx="30" cy="22" r="10" fill="none" stroke="var(--bdr-ink-2)" stroke-width="3"/>'
            '<path d="M37 29L43 35" stroke="var(--bdr-ink-2)" stroke-width="4.5" stroke-linecap="round"/></svg>')



# Every mention of roofscope in the book's prose: the name links to the repository, after a small
# logo. Used by bin/build-roofline.py on every page and by the site editor's preview. The repository
# is private for now, so the link shows readers GitHub's 404 page until it is public.
MENTION_LINKS = True
_KEEP = re.compile(r'<details class="code-card".*?</details>|<div class="repo-card">.*?</div></div></div>|'  # they carry the logo already
                   r"<(a|code|pre|script|style|svg|summary|title|textarea)\b.*?</\1>|<[^>]+>|^---\n.*?\n---\n", re.S)
_WORD = re.compile(r"\broofscope\b(?![/.\w-])")


def mark_mentions(html: str, uid: str) -> str:
    n = [0]

    def word(_m):
        n[0] += 1
        mark = logo(16, f"rsm-{uid}-{n[0]}") + "roofscope"
        return f'<a class="rs-mention" href="{GH}">{mark}</a>' if MENTION_LINKS else f'<span class="rs-mention">{mark}</span>'
    parts, pos = [], 0
    for m in _KEEP.finditer(html):  # tags, links, code, drawings and the front matter stay as they are
        parts += [_WORD.sub(word, html[pos:m.start()]), m.group(0)]
        pos = m.end()
    return "".join(parts) + _WORD.sub(word, html[pos:])

# name: (file, [(first-line regex, last-line regex), ...], caption, open)
CARDS = {
    "ceilings": ("roofscope/ert.py",
                 [(r"^def sustained\(", r"^    return statistics\.median\(ts\)"),
                  (r"^def measure\(", r'^    return \{"t": t')],
                 "How the sustained rates of Definition 5 are timed: the median over three repeats of the mean "
                 "time per call over about half a second of back-to-back calls, with the SM clock and the power "
                 "sampled every 50 ms by <code>nvidia-smi</code> meanwhile. The ceiling is the highest rate of any "
                 "benchmark; the FP8 and BF16 ceilings come from shorter bursts (CUDA-graph replays of about 2 ms, "
                 "and 20 back-to-back GEMMs).", False),
    "routing": ("roofscope/routing.py",
                [(r"^def _rs_routing_hist\(", r"^    tl\.atomic_add\("),
                 (r"^    def capture\(", r"^            _rs_routing_step\[")],
                "The routing counter. vLLM's router calls <code>capture</code> in every MoE layer; one Triton "
                "kernel adds 1 to <code>counts[step, layer, expert]</code> for each (token, choice), and the "
                "last layer advances a step index kept on the GPU. Both kernels are captured into the CUDA "
                "graph, so the timed passes are counted as well.", False),
    "ufreq": ("pages/campaign3/data3.py",
              [(r"^        meas = \(T > 0\)", r"^        pop = \(1 - q\)\.sum")],
              "The counts of Definition 6 from one run: <code>meas</code> is the mean number of distinct "
              "experts per layer over the timed passes, and <code>pop</code> the count that independent "
              "tokens with the run's expert frequencies would touch, with each term estimated without bias "
              "by a ratio of binomial coefficients.", True),
    "prompts": ("roofscope/prompts.py",
                [(r"^ESSAY_HEAD = ", r'^              "describe its people'),
                 (r'^    if mode == "essay":', r"^        return lambda w: head")],
                "The essay prompt: a passage of exactly the length that makes the whole chat-template prompt "
                "\\(C\\) tokens, between a fixed head and tail.", True),
    "windows": ("roofscope/collect.py",
                [(r"^        # 3\. profiled window P", r"^            prof_wall = "),
                 (r"^        # 4\. timed window T", r"^        wall = time\.perf_counter\(\) - t0")],
                "The two measurement windows of a point: the profiled window records every kernel, and the "
                "timed window, with the profiler off, gives the step time \\(t\\).", False),
    "labels": ("roofscope/analyze2.py",
               [(r"^    def label_step\(", r"^            out\.append\(lab\)")],
               "Labelling the kernels of a graph-mode step. A kernel takes the module of the eager kernel "
               "with the same name, grid and block, counted in order of occurrence; the head and the "
               "sampler keep their own labels; compiler-fused <code>triton_</code> kernels with no eager "
               "counterpart become <code>fused_elementwise</code>, which counts as outside the model.", False),
    "bytecode": ("roofscope/patches/__init__.py",
                 [(r"^def _drop_bytecode\(", r"^        f\.unlink\(\)")],
                 "Every source patch ends by deleting the cached bytecode of the file it edits.", True),
    "sampler": ("roofscope/patches/sampler.py",
                [(r"^    # ---- pass 1: block statistics", r"^        m = mn$")],
                "The sampler's single read of the logits. Per block of the vocabulary it computes the "
                "soft-capped, temperature-scaled logits, their maximum, the sum of their exponentials and "
                "the entropy term, and merges them into running values for the row; later steps read only "
                "one block again, to draw the sample, and the slice of probabilities kept for "
                "self-conditioning.", False),
    "moe": ("roofscope/kmoe/kernel.py",
            [(r"^    xs = tl\.load\(xs_ptr \+ tok", r"^        h = _gelu_tanh\(gate\) \* up"),
             (r"^    tw = tl\.load\(tw_ptr", r"^        tl\.atomic_add\(o_ptrs")],
            "The two fused epilogues of <code>moe_ws</code>. In the first GEMM a tile holds 128 gate "
            "columns next to their 128 up columns, so the activation is applied in registers; in the "
            "second, each row is scaled by its routing weight and added into its token's output.", True),
    "exchange": ("roofscope/kattn/make_pair.py",
                 [(r"^    def exchange_S\(", r"^        base = \(slot \* self\.num_mma_threads"),
                  (r"^            elif const_expr\(self\.xmode in \(8, 9, 10, 11\)\):", r"^                cute\.copy\(atom, srcv, dstv, mbar_ptr"),
                  (r"^            # wait for the partner's fragment and add it", r"^                acc_S\[i\] = acc_S\[i\] \+ ")],
                 "The score exchange of the pair kernel, in the generator that writes it into FlashAttention 4's "
                 "source. The engine runs mode 9: each thread converts its fragment of the partial scores to "
                 "FP16 and pushes it into the partner CTA's shared memory with 16-byte asynchronous stores, "
                 "into one of two alternating buffers, with completion counted in bytes on the partner's "
                 "barrier; it then waits for the partner's fragment and adds it.", False),
}

REPRO = """# the grid of Parts 8-11, one H200 per configuration
modal run cloud/modal_c2.py::runs --spec specs/c4.json
modal run cloud/modal_c2.py::fetch --tag c4
python3 -m roofscope.analyze2 results/c2/c4 results/c2/c4
python3 pages/campaign3/data3.py c4
# writes this book's practice-data.json
python3 pages/book/export.py"""


def _show(path: str) -> list[str]:
    r = subprocess.run(["git", "-C", str(CLONE), "show", f"{SHA}:{path}"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"code card: cannot read {path} at {SHA[:7]} in {CLONE}: {r.stderr.strip()}")
    return r.stdout.split("\n")


def _find(lines: list[str], rx: str, start: int, what: str) -> int:
    c = re.compile(rx)
    for i in range(start, len(lines)):
        if c.search(lines[i]):
            return i
    raise SystemExit(f"code card: pattern {rx!r} ({what}) not found")


def _hl(code: str, lexer) -> list[str]:
    """Highlighted HTML, one string per line (spans never cross a line)."""
    out, cur = [], []
    for ttype, val in lexer.get_tokens(code):
        cls = ""
        t = ttype
        while t is not None and not cls:
            cls = STANDARD_TYPES.get(t, "")
            t = t.parent
        for j, piece in enumerate(val.split("\n")):
            if j:
                out.append("".join(cur)); cur = []
            if piece:
                e = html.escape(piece, quote=False)
                cur.append(f'<span class="{cls}">{e}</span>' if cls else e)
    out.append("".join(cur))
    if out and out[-1] == "":
        out.pop()
    return out


def _rows(lines: list[str], lexer, starts: list[int] | None) -> str:
    hl = _hl("\n".join(lines) + "\n", lexer)
    rows = []
    for i, h in enumerate(hl):
        n = f'<span class="ln" aria-hidden="true">{starts[i]}</span>' if starts else ""
        rows.append(f'<span class="cl">{n}{h or " "}</span>')
    return "".join(rows)


def card(name: str) -> str:
    path, ranges, caption, is_open = CARDS[name]
    src = _show(path)
    spans, start = [], 0
    for a_rx, b_rx in ranges:
        a = _find(src, a_rx, start, f"{name} first line")
        b = _find(src, b_rx, a, f"{name} last line")
        spans.append((a, b)); start = b + 1
    # common indentation of all quoted lines
    body = [l for a, b in spans for l in src[a:b + 1] if l.strip()]
    ind = min(len(l) - len(l.lstrip(" ")) for l in body)
    py = PythonLexer()
    parts, links = [], []
    for k, (a, b) in enumerate(spans):
        if k:
            parts.append('<span class="cl gap"><span class="ln" aria-hidden="true">⋮</span> </span>')
        parts.append(_rows([l[ind:] for l in src[a:b + 1]], py, list(range(a + 1, b + 2))))
        links.append(_a(f"{GH}/blob/{SHA}/{path}#L{a + 1}-L{b + 1}", f"lines {a + 1}–{b + 1}"))
    d, f = path.rsplit("/", 1)
    nlines = sum(b - a + 1 for a, b in spans)
    return (f'<details class="code-card" id="code-{name}"{" open" if is_open else ""}>'
            f'<summary><span class="code-top">{logo(18, f"rsl-{name}")}<span class="code-file">'
            f'<span class="code-dir">{d}/</span>{f}</span><span class="code-n">{nlines} lines</span></span>'
            f'<span class="code-cap">{caption}</span></summary>'
            f'<div class="code-body mathjax_ignore tex2jax_ignore"><pre><code>{"".join(parts)}</code></pre></div>'
            f'<div class="code-foot">{"On GitHub at" if PUBLIC else "roofscope at commit"} {_a(f"{GH}/tree/{SHA}", f"<code>{SHA[:7]}</code>")}: {" · ".join(links)}</div>'
            '</details>\n')


def repo_card() -> str:
    return ('<div class="repo-card">' + logo(52, "rsl-repo", label=True) +
            f'<div><div class="repo-name">{_a(f"{GH}/tree/{SHA}", "roofscope")}</div>'
            '<div class="repo-desc">The harness behind Parts 8–11: it runs vLLM on Modal, records every '
            'kernel and every routing decision, and turns them into the components, rooflines and '
            'efficiencies of these chapters. The code excerpts quote it at commit '
            f'<code>{SHA[:7]}</code>, with their line numbers'
            + (', each linked to GitHub.' if PUBLIC else '; the repository is not yet public.') + '</div></div></div>\n')


def repro_card() -> str:
    return ('<details class="code-card" id="code-reproduce" open>'
            f'<summary><span class="code-top">{logo(18, "rsl-repro")}<span class="code-file">'
            '<span class="code-dir">roofscope/</span>README.md</span><span class="code-n">pipeline</span></span>'
            '<span class="code-cap">From the GPU runs to the numbers of Parts 8–11. Raw traces and routing '
            'arrays stay on the Modal volume; the repository keeps the analysed summaries.</span></summary>'
            f'<div class="code-body nolines mathjax_ignore tex2jax_ignore"><pre><code>{_rows(REPRO.split(chr(10)), BashLexer(), None)}</code></pre></div>'
            f'<div class="code-foot">{_a(f"{GH}/tree/{SHA}#readme", f"the README of roofscope at commit <code>{SHA[:7]}</code>")}</div>'
            '</details>\n')


def blocks() -> dict[str, str]:
    b = {name: card(name) for name in CARDS}
    b["repo"] = repo_card()
    b["reproduce"] = repro_card()
    return b
