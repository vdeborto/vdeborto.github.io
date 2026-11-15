"""Figures and tables of the book's practice chapters (Parts 0 and 8-11), as static HTML/SVG.

Imported by bin/build-roofline.py, which replaces each <!-- pfig:NAME --> placeholder in the
note's text with blocks()[NAME]. Numbers come from _roofline/practice-data.json, written by
~/code/roofscope-campaign2/pages/book/export.py from the campaign-3 measurements (H200, vLLM 0.28,
FP8 weights). Colours are the book's tokens: AR blue (--bdr-ar), TD orange (--bdr-td).
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "_roofline/practice-data.json"

AR, TD, MUT, INK, RULE = "var(--bdr-ar)", "var(--bdr-td)", "var(--bdr-muted)", "var(--ink)", "var(--rule)"
GREEN, AMBER, CRIT = "var(--bdr-s3)", "var(--bdr-s4)", "var(--bdr-critical)"


def f(x, n=2):
    return "–" if x is None or not math.isfinite(x) else f"{x:.{n}f}"


def kc(C):
    return f"{C // 1024}k"


def sub(a, b):
    return f'{a}<tspan baseline-shift="sub" font-size="0.75em">{b}</tspan>'


_TEXT = re.compile(r"<text\b([^>]*)>")
_ATTR = re.compile(r'\s(fill|font-size|font-weight|letter-spacing)="([^"]*)"')


def _inline(m):
    """The book's CSS sets fill and font-size on every figure <text>, and CSS beats SVG
    presentation attributes: move them into an inline style, which beats the CSS."""
    attrs = m.group(1)
    props = {k: v for k, v in _ATTR.findall(attrs)}
    rest = _ATTR.sub("", attrs)
    st = re.search(r'\sstyle="([^"]*)"', rest)
    pre = st.group(1).rstrip(";") + ";" if st else ""
    rest = re.sub(r'\sstyle="[^"]*"', "", rest)
    css = pre + ";".join(f"{k}:{v}{'px' if k == 'font-size' and v.replace('.', '').isdigit() else ''}" for k, v in props.items())
    return f'<text{rest} style="{css}">' if css else f"<text{rest}>"


def svg(w, h, label, body, extra=""):
    body = _TEXT.sub(_inline, body)
    out = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}" '
           f'class="pfig-svg{" pfig-wide" if w >= 700 else ""}"{extra}>' + body + "</svg>")
    # wide diagrams keep a readable size on phones and scroll sideways instead of shrinking
    return f'<div class="pfig-scroll">{out}</div>' if w >= 700 else out


def arrow_defs(i, color="currentColor"):
    return (f'<defs><marker id="pa{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker></defs>')


def fig(fid, kicker, title, lede, inner, cap, wide=False):
    return (f'  <div class="bdr-fig pfig{" breakout" if wide else ""}" id="{fid}">\n'
            f'    <div class="fig-k">{kicker}</div>\n    <p class="fig-title">{title}</p>\n'
            + (f'    <p class="fig-lede">{lede}</p>\n' if lede else "")
            + f'    {inner}\n    <p class="fig-cap">{cap}</p>\n  </div>\n')


def table(caption, head, rows, cls="pt"):
    th = "".join(f'<th class="l">{h}</th>' for h in head)
    tr = "".join("<tr>" + "".join(f'<td class="l">{c}</td>' for c in r) + "</tr>" for r in rows)
    return (f'  <div class="table-wrap"><table class="{cls}">\n    <caption>{caption}</caption>\n'
            f'    <thead><tr>{th}</tr></thead>\n    <tbody>{tr}</tbody>\n  </table></div>\n')


def rng(xs, n=2):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return f"{min(xs):.{n}f}–{max(xs):.{n}f}" if xs else "–"


# ---------------------------------------------------------------- Part 0: diagrams
def fig_decoders():
    W, H, x0, sw, gap = 760, 250, 120, 36, 4
    n = 15
    s = [arrow_defs(0)]
    s.append(f'<text x="{x0}" y="22" fill="{MUT}" font-size="12">forward passes, in time order; every pass reads the weights once</text>')
    for row, (name, col, y) in enumerate((("AR", AR, 44), ("TD", TD, 134))):
        s.append(f'<text x="{x0 - 14}" y="{y + 26}" text-anchor="end" font-weight="600" fill="{col}">{name}</text>')
        for i in range(n):
            x = x0 + i * (sw + gap)
            if row == 0:
                s.append(f'<rect x="{x}" y="{y}" width="{sw}" height="40" rx="3" fill="{col}" fill-opacity="0.85"/>')
                s.append(f'<text x="{x + sw / 2}" y="{y + 25}" text-anchor="middle" font-size="11" fill="var(--bdr-surface)">1</text>')
            else:
                commit = i == n - 1
                s.append(f'<rect x="{x}" y="{y}" width="{sw}" height="40" rx="3" fill="{col}" fill-opacity="{0.85 if commit else 0.28}"/>')
        s.append(f'<text x="{x0 + n * (sw + gap) + 8}" y="{y + 18}" font-size="12.5" fill="{INK}">{"15 tokens" if row == 0 else "256 tokens"}</text>')
        s.append(f'<text x="{x0 + n * (sw + gap) + 8}" y="{y + 34}" font-size="11.5" fill="{MUT}">{"one per pass" if row == 0 else "one canvas"}</text>')
    xt = x0 + 7 * (sw + gap)
    s.append(f'<text x="{x0}" y="{94 + 6}" font-size="11.5" fill="{MUT}">each pass commits one token per sequence</text>')
    s.append(f'<path d="M{x0},{184} v8 h{14 * (sw + gap) - gap} v-8" fill="none" stroke="{TD}" stroke-width="1.2"/>')
    s.append(f'<text x="{x0 + 7 * (sw + gap) - gap}" y="{208}" text-anchor="middle" font-size="12" fill="{INK}">S = 14 denoising passes over the 256 positions of the canvas</text>')
    xc = x0 + 14 * (sw + gap) + sw / 2
    s.append(f'<line x1="{xc}" y1="{226}" x2="{xc}" y2="{180}" stroke="{TD}" marker-end="url(#pa0)"/>')
    s.append(f'<text x="{xc}" y="{242}" text-anchor="middle" font-size="12" fill="{INK}">commit pass</text>')
    return svg(W, H, "In the same 15 passes AR commits 15 tokens and TD commits a canvas of 256", "".join(s))


# ---------------------------------------------------------------- Part 8
def heat(d, key, title, scale, Bs=None, Cs=None, fmtv=None, color=None, size=(62, 30)):
    g = d["gains"]
    Bs, Cs = Bs or d["Bs"], Cs or d["Cs"]
    cw, ch = size
    ml, mt = 58, 42
    W, H = ml + cw * len(Cs) + 8, mt + ch * len(Bs) + 30
    s = [f'<text x="{ml}" y="16" font-size="12.5" font-weight="600" fill="{INK}">{title}</text>']
    for j, C in enumerate(Cs):
        s.append(f'<text x="{ml + j * cw + cw / 2}" y="{mt - 8}" text-anchor="middle" font-size="11.5" fill="{MUT}">{kc(C)}</text>')
    for i, B in enumerate(reversed(Bs)):
        y = mt + i * ch
        s.append(f'<text x="{ml - 8}" y="{y + ch / 2 + 4}" text-anchor="end" font-size="11.5" fill="{MUT}">B = {B}</text>')
        for j, C in enumerate(Cs):
            r = g.get(f"{B},{C}")
            v = None if r is None else (key(r) if callable(key) else r.get(key))
            x = ml + j * cw
            if v is None:
                s.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" fill="{RULE}" fill-opacity="0.5"/>'
                         f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" font-size="11.5" fill="{MUT}">–</text>')
                continue
            fill, a = (color or (lambda v: (TD if v >= 1 else AR, min(1, abs(math.log(v)) / scale))))(v)
            a = max(0.06, a)
            txt = "#fff" if a > 0.55 else INK
            s.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" fill="{fill}" fill-opacity="{a:.2f}"><title>B = {B}, C = {kc(C)}: {(fmtv or (lambda v: f"{v:.2f}"))(v)}</title></rect>'
                     f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" font-size="11.5" style="fill:{txt};font-weight:{600 if a > 0.55 else 400}">{(fmtv or (lambda v: f"{v:.2f}"))(v)}</text>')
    s.append(f'<text x="{ml + cw * len(Cs) / 2}" y="{H - 6}" text-anchor="middle" font-size="11.5" fill="{MUT}">context C</text>')
    return svg(W, H, title, "".join(s))


def gain_scale(d, keys):
    vals = [r[k] for r in d["gains"].values() for k in keys if r.get(k)]
    return max(abs(math.log(v)) for v in vals)


def maps(d, specs):
    sc = gain_scale(d, [k for k, _ in specs])
    return '<div class="pfig-grid">' + "".join(f"<div>{heat(d, k, t, sc)}</div>" for k, t in specs) + "</div>"


# ---------------------------------------------------------------- Part 9: diagrams
# ---------------------------------------------------------------- Part 10: diagrams (from the kernel note)
MOE_ROWS = [("quantise x, per tensor (2 kernels)", "", None, ""), ("sort rows by expert", "", "sort rows by expert", ""),
            ("GEMM 1 (gate, up)", "184 MB", "GEMM 1 + quantise x + gelu(gate)·up", "100 MB"), ("act_and_mul", "92 MB", None, ""),
            ("quantise h, per tensor (2 kernels)", "46 MB", "quantise h, per row (1 kernel)", "50 MB"),
            ("GEMM 2, one row per (token, expert)", "369 MB", "GEMM 2 + p·row, atomic add into y", "in L2"), ("moe_sum (read 369 MB)", "46 MB", None, "")]


def fig_moe():
    W, rh, top = 780, 46, 64
    H = top + rh * len(MOE_ROWS) + 16
    s = [arrow_defs(6), f'<text x="170" y="24" text-anchor="middle" font-weight="600" fill="{INK}">vLLM Triton path</text>',
         f'<text x="570" y="24" text-anchor="middle" font-weight="600" fill="{INK}">new path (moe_ws)</text>',
         f'<text x="170" y="44" text-anchor="middle" font-size="11.5" fill="{MUT}">kernel · bytes it writes to memory</text>',
         f'<text x="570" y="44" text-anchor="middle" font-size="11.5" fill="{MUT}">kernel · bytes it writes to memory</text>']
    for i, (lb, lw, rb, rw) in enumerate(MOE_ROWS):
        y = top + i * rh
        s.append(f'<rect x="20" y="{y}" width="270" height="32" rx="4" fill="none" stroke="{MUT}"/><text x="155" y="{y + 21}" text-anchor="middle" font-size="12.5" fill="{INK}">{lb}</text>')
        if lw:
            s.append(f'<text x="298" y="{y + 21}" font-size="12" fill="{TD}">{lw}</text>')
        if rb:
            s.append(f'<rect x="420" y="{y}" width="290" height="32" rx="4" fill="none" stroke="{MUT}"/><text x="565" y="{y + 21}" text-anchor="middle" font-size="12.5" fill="{INK}">{rb}</text>')
            if rw:
                s.append(f'<text x="718" y="{y + 21}" font-size="12" fill="{TD}">{rw}</text>')
        else:
            s.append(f'<rect x="420" y="{y}" width="290" height="32" rx="4" fill="none" stroke="{MUT}" stroke-dasharray="4 4" opacity="0.5"/><text x="565" y="{y + 21}" text-anchor="middle" font-size="12" fill="{MUT}">removed (fused)</text>')
        if i:
            s.append(f'<line x1="155" y1="{y - 14}" x2="155" y2="{y - 2}" stroke="currentColor" marker-end="url(#pa6)"/><line x1="565" y1="{y - 14}" x2="565" y2="{y - 2}" stroke="currentColor" marker-end="url(#pa6)"/>')
    return svg(W, H, "vLLM's expert path runs seven passes and writes intermediate tensors to memory; the new path fuses three of them away", "".join(s))


def fig_smem():
    W, x0, sc = 780, 170, 2.3
    rows = [("FA4, head dim. 512", [("Q", 64, AR), ("K", 64, GREEN), ("V", 64, AMBER), ("P", 8, MUT)]),
            ("pair kernel, per CTA", [("Q half", 64, AR), ("K st. 1", 32, GREEN), ("K st. 2", 32, GREEN), ("V st. 1", 32, AMBER), ("V st. 2", 32, AMBER), ("scores ×2", 32, TD)])]
    s = []
    for i, (lab, segs) in enumerate(rows):
        y = 20 + i * 52
        s.append(f'<text x="{x0 - 10}" y="{y + 21}" text-anchor="end" font-size="12.5" fill="{INK}">{lab}</text>')
        x = x0
        for name, kb, col in segs:
            w = kb * sc
            s.append(f'<rect x="{x}" y="{y}" width="{w - 2}" height="32" rx="3" fill="{col}" fill-opacity="0.8"/>')
            if w > 40:
                s.append(f'<text x="{x + w / 2 - 1}" y="{y + 20}" text-anchor="middle" font-size="11.5" style="fill:#fff">{name}</text>')
            x += w
        s.append(f'<text x="{x0 + 227 * sc + 10}" y="{y + 21}" font-size="12" fill="{INK}">{sum(k for _, k, _ in segs)} KB</text>')
    lx = x0 + 227 * sc
    s += [f'<line x1="{lx}" y1="10" x2="{lx}" y2="122" stroke="{CRIT}" stroke-dasharray="5 4"/>',
          f'<text x="{lx}" y="140" text-anchor="middle" font-size="12" fill="{CRIT}">227 KB per SM</text>']
    return svg(W, 150, "Shared memory per thread block: FA4 at head dimension 512 fits one pipeline stage; the pair kernel fits two stages and its exchange buffers", "".join(s))


CHW = 0.61  # IBM Plex Mono advance width, in em


def tw(text, fs):
    """Estimated rendered width of a monospace label (tspans count by their characters)."""
    return len(re.sub(r"<[^>]+>", "", text)) * CHW * fs


FIT = []  # labels that do not fit their box, reported by blocks()


def fits(text, fs, width, where):
    if tw(text, fs) > width:
        FIT.append(f"{where}: {re.sub(r'<[^>]+>', '', text)!r} needs {tw(text, fs):.0f} px, has {width}")
    return text


def tbox(x, y, w, h, lines, col=MUT, title=None, part=None, where=""):
    """A box with an optional small part label, a bold title and lines of text, left-aligned."""
    r = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="var(--bdr-surface)" stroke="{col}" stroke-width="1.4"/>']
    yy = y + 18
    if part:
        r.append(f'<text x="{x + 10}" y="{yy}" font-size="10" letter-spacing=".06em" fill="{MUT}">{fits(part, 10, w - 20, where)}</text>')
        yy += 17
    if title:
        r.append(f'<text x="{x + 10}" y="{yy}" font-size="12.5" font-weight="600" fill="{INK}">{fits(title, 12.5, w - 20, where)}</text>')
        yy += 17
    for t in lines:
        r.append(f'<text x="{x + 10}" y="{yy}" font-size="11" fill="var(--bdr-ink-2)">{fits(t, 11, w - 20, where)}</text>')
        yy += 15
    return "".join(r)


def fig_map():
    W, H, cw, gap, x0 = 924, 360, 168, 16, 10
    X = lambda i: x0 + i * (cw + gap)
    y1, y2, h = 44, 236, 92
    s = [arrow_defs(1)]
    s.append(f'<text x="{x0}" y="26" font-size="11.5" font-weight="600" letter-spacing=".08em" fill="{MUT}">ON PAPER · PARTS 1–7</text>')
    s.append(f'<text x="{x0}" y="{y2 + 30}" font-size="11.5" font-weight="600" letter-spacing=".08em" fill="{MUT}">ON AN H200</text>')
    s.append(f'<text x="{x0}" y="{y2 + 47}" font-size="11.5" font-weight="600" letter-spacing=".08em" fill="{MUT}">PARTS 8–11</text>')
    top = [("PART 1", "Setup", ["costs F and Q,", "intensities, penalties"]),
           ("PART 2", "The gain", ["Proposition 1: Γ", "from two intensities"]),
           ("PARTS 3 · 4", "Linear, attention", ["Γ when the weights or", "the KV cache dominate"]),
           ("PART 5", "Whole picture", ["regimes and exact", "maps of Γ"]),
           ("PARTS 6 · 7", "Calculator", ["and three worked", "examples"])]
    bot = [("PART 8", "Effective rooflines", ["measured π, β and", "expert counts U"]),
           ("PART 9", "Measurement stack", ["the time of every", "kernel, by component"]),
           ("PART 10", "Kernels", ["two rewrites raise", "their efficiency"]),
           ("PART 11", "Conclusion", ["measured Γ against", "the bound"])]
    for i, (p, t, l) in enumerate(top):
        s.append(tbox(X(i), y1, cw, h, l, MUT, t, p, "map"))
        if i:
            s.append(f'<line x1="{X(i) - gap + 2}" y1="{y1 + h / 2}" x2="{X(i) - 2}" y2="{y1 + h / 2}" stroke="currentColor" marker-end="url(#pa1)"/>')
    for i, (p, t, l) in enumerate(bot, start=1):
        s.append(tbox(X(i), y2, cw, h, l, TD, t, p, "map"))
        if i > 1:
            s.append(f'<line x1="{X(i) - gap + 2}" y1="{y2 + h / 2}" x2="{X(i) - 2}" y2="{y2 + h / 2}" stroke="currentColor" marker-end="url(#pa1)"/>')
    cx1 = X(1) + cw / 2
    s.append(f'<line x1="{cx1}" y1="{y1 + h + 2}" x2="{cx1}" y2="{y2 - 3}" stroke="currentColor" marker-end="url(#pa1)"/>')
    s.append(f'<text x="{cx1 + 8}" y="{(y1 + h + y2) / 2 + 4}" font-size="11" fill="{MUT}">inputs of Proposition 1</text>')
    xa, xb = X(3) + cw / 2, X(4) + cw / 2
    s.append(f'<line x1="{xa}" y1="{y1 + h + 2}" x2="{xb}" y2="{y2 - 3}" stroke="currentColor" marker-end="url(#pa1)"/>')
    s.append(f'<text x="{xb - 2}" y="{y1 + h + 30}" font-size="11" fill="{MUT}">maps to compare</text>')
    return svg(W, H, "Map of the book: Parts 1 to 7 bound the gain on paper; Parts 8 to 11 measure it on an H200 and compare", "".join(s))


def fig_ladder(d):
    g = d["gains"]
    W, H, bw, gap, x0 = 900, 270, 196, 30, 13
    rungs = [("(a) book's bound", ["datasheet π and β,", "independent routing"], "roof_ind_ds", MUT),
             ("(b) ceilings", ["π and β measured", "on this H200"], "roof_ind", MUT),
             ("(c) routing", ["U counted in", "the same runs"], "roof", MUT),
             ("(d) measured", ["vLLM with the fixes", "and the new kernels"], "measured", TD)]
    steps = ["Part 8", "Part 8", "Parts 9–10"]
    s = [arrow_defs(2)]
    for i, (t, l, key, col) in enumerate(rungs):
        x = x0 + i * (bw + gap)
        s.append(tbox(x, 34, bw, 222, l, col, t, None, "ladder"))
        for j, B in enumerate((1, 64)):
            v = g[f"{B},1024"][key]
            y = 136 + j * 62
            s.append(f'<text x="{x + 10}" y="{y}" font-size="11" fill="{MUT}">batch {B}, context 1k</text>')
            s.append(f'<text x="{x + 10}" y="{y + 30}" font-size="24" font-weight="600" fill="{TD if v >= 1 else AR}">Γ = {v:.2f}</text>')
        if i < 3:
            xa = x + bw + 2
            s.append(f'<line x1="{xa}" y1="145" x2="{xa + gap - 4}" y2="145" stroke="currentColor" marker-end="url(#pa2)"/>')
            s.append(f'<text x="{x + bw + gap / 2}" y="22" text-anchor="middle" font-size="11" fill="{MUT}">{steps[i]}</text>')
    return svg(W, H, "From the book's bound to the measured gain in four steps, each changing one input", "".join(s))


def fig_eff_roofline(d):
    ce = d["ceil"]
    W, H, ml, mr, mt, mb = 720, 380, 70, 170, 24, 50
    x_lo, x_hi, y_lo, y_hi = 10, 10000, 40, 3000
    X = lambda I: ml + (math.log10(I) - math.log10(x_lo)) / (math.log10(x_hi) - math.log10(x_lo)) * (W - ml - mr)
    Y = lambda p: mt + (1 - (math.log10(p) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))) * (H - mt - mb)
    s = []
    for v in (10, 100, 1000, 10000):
        s.append(f'<line x1="{X(v)}" y1="{mt}" x2="{X(v)}" y2="{H - mb}" stroke="{RULE}"/><text x="{X(v)}" y="{H - mb + 18}" text-anchor="middle" font-size="12" fill="{MUT}">{v:,}</text>')
    for v in (50, 100, 200, 500, 1000, 2000):
        s.append(f'<line x1="{ml}" y1="{Y(v)}" x2="{W - mr}" y2="{Y(v)}" stroke="{RULE}"/><text x="{ml - 8}" y="{Y(v) + 4}" text-anchor="end" font-size="12" fill="{MUT}">{v:,}</text>')
    s.append(f'<text x="{(ml + W - mr) / 2}" y="{H - 10}" text-anchor="middle" font-size="12" fill="{MUT}">arithmetic intensity I (FLOP per byte of HBM traffic)</text>')
    s.append(f'<text x="{ml - 60}" y="{mt - 10}" font-size="12" fill="{MUT}">attainable TFLOP/s</text>')
    def roof(pi, beta, col, dash, lab, above):
        ridge = pi / beta
        pts = [(x_lo, x_lo * beta / 1e12), (ridge, pi / 1e12), (x_hi, pi / 1e12)]
        out = f'<polyline points="{" ".join(f"{X(a)},{Y(b)}" for a, b in pts)}" fill="none" stroke="{col}" stroke-width="2.2" stroke-dasharray="{dash}"/>'
        out += f'<circle cx="{X(ridge)}" cy="{Y(pi / 1e12)}" r="3.5" fill="{col}"/>'
        ty = Y(pi / 1e12) - 10 if above else Y(pi / 1e12) + 20
        out += f'<text x="{X(ridge) + 6}" y="{ty}" font-size="11.5" fill="{col}">ridge {ridge:.0f}</text>'
        out += f'<text x="{W - mr + 8}" y="{Y(pi / 1e12) + 4 - (6 if above else -6)}" font-size="11.5" fill="{col}">{lab}</text>'
        return out
    s.append(roof(1979e12, 4.8e12, MUT, "6 4", "datasheet FP8 1,979", True))
    s.append(roof(ce["pi_fp8"], ce["beta"], TD, "", f"empirical FP8 {ce['pi_fp8'] / 1e12:,.0f}", False))
    pa = ce["pi_attn"] / 1e12
    s.append(f'<line x1="{X(ce["pi_attn"] / ce["beta"])}" y1="{Y(pa)}" x2="{W - mr}" y2="{Y(pa)}" stroke="{AR}" stroke-width="2"/>')
    s.append(f'<text x="{W - mr + 8}" y="{Y(pa) + 4}" font-size="11.5" fill="{AR}">attention {pa:,.0f}</text>')
    s.append(f'<text x="{X(11)}" y="{Y(1500)}" font-size="11.5" fill="{MUT}">slope: bandwidth, 4.80 TB/s</text>')
    s.append(f'<text x="{X(11)}" y="{Y(1500) + 16}" font-size="11.5" fill="{MUT}">datasheet, {ce["beta"] / 1e12:.2f} empirical</text>')
    return svg(W, H, "The empirical FP8 roofline lies below the datasheet one, more in compute than in bandwidth, so its ridge point moves left", "".join(s))


def spread(ys, gap=15):
    """Push label positions apart so that consecutive labels are at least `gap` apart."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def fig_eff_routing(d, C=1024):
    R, Bs, G = d["routing"], d["Bs"], d["gains"]
    out = []
    for mode, col, name in (("td", TD, "TD: 256 positions per sequence"), ("ar", AR, "AR: one token per sequence")):
        W, H, ml, mr, mt, mb = 400, 330, 44, 112, 50, 44
        X = lambda i: ml + i / (len(Bs) - 1) * (W - ml - mr)
        Y = lambda u: mt + (1 - u / 140) * (H - mt - mb)
        s = [f'<text x="{ml - 40}" y="16" font-size="12.5" font-weight="600" fill="{col}">{fits(name, 12.5, W - 10, "routing")}</text>',
             f'<text x="{ml - 40}" y="34" font-size="11" fill="{MUT}">distinct experts per layer and pass, U</text>']
        for v in range(0, 141, 20):
            s.append(f'<line x1="{ml}" y1="{Y(v)}" x2="{W - mr}" y2="{Y(v)}" stroke="{RULE}"/><text x="{ml - 6}" y="{Y(v) + 4}" text-anchor="end" font-size="11" fill="{MUT}">{v}</text>')
        for i, B in enumerate(Bs):
            s.append(f'<text x="{X(i)}" y="{H - mb + 16}" text-anchor="middle" font-size="11" fill="{MUT}">{B}</text>')
        s.append(f'<text x="{(ml + W - mr) / 2}" y="{H - 8}" text-anchor="middle" font-size="11" fill="{MUT}">batch B</text>')
        pts = [(i, R[mode].get(f"{B},{C}")) for i, B in enumerate(Bs)]
        pts = [(i, dict(v, meas=G[f"{Bs[i]},{C}"]["U_td" if mode == "td" else "U_ar"])) for i, v in pts if v and f"{Bs[i]},{C}" in G]
        band = " ".join(f"{X(i)},{Y(v['meas_hi'])}" for i, v in pts) + " " + " ".join(f"{X(i)},{Y(v['meas_lo'])}" for i, v in reversed(pts))
        s.append(f'<polygon points="{band}" fill="{col}" fill-opacity="0.15"/>')
        keys = [("uni", "2 4", "uniform", 1.8), ("pop", "7 4", "popularity", 1.8), ("meas", "", "measured", 2.4)]
        i_, last = pts[-1]
        ly = spread([Y(last[k]) + 4 for k, *_ in keys], 15)
        for (key, dash, lab, w_), yl in zip(keys, ly):
            c = col if key == "meas" else MUT
            s.append(f'<polyline points="{" ".join(f"{X(i)},{Y(v[key])}" for i, v in pts)}" fill="none" stroke="{c}" stroke-width="{w_}" stroke-dasharray="{dash}"/>')
            if key == "meas":
                s += [f'<circle cx="{X(i)}" cy="{Y(v[key])}" r="3" fill="{c}"><title>B = {Bs[i]}: U = {v[key]:.1f} (repeats {v["meas_lo"]:.0f}–{v["meas_hi"]:.0f})</title></circle>' for i, v in pts]
            s.append(f'<text x="{X(i_) + 8}" y="{yl}" font-size="11" fill="{c}">{lab}</text>')
        out.append(svg(W, H, f"{name}: distinct experts per layer against batch, measured and predicted for independent tokens", "".join(s)))
    return '<div class="pfig-pair">' + "".join(f"<div>{o}</div>" for o in out) + "</div>"



def fig_flags():
    W, H = 900, 290
    s = [arrow_defs(4), arrow_defs(5, TD)]
    def cell(x, y, w, t, col, dash=""):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="4" fill="none" stroke="{col}" stroke-dasharray="{dash}"/>'
                f'<text x="{x + w / 2}" y="{y + 22}" text-anchor="middle" font-size="11.5" fill="{INK}">{fits(t, 11.5, w - 12, "flags")}</text>')
    s.append(f'<text x="20" y="22" font-size="12.5" font-weight="600" fill="{INK}">vLLM 0.28: converted at every step</text>')
    s.append(f'<text x="480" y="22" font-size="12.5" font-weight="600" fill="{INK}">fix: one persistent int32 buffer</text>')
    s.append(cell(20, 44, 200, "bool buffer, live flags", MUT))
    s.append(cell(20, 150, 190, "int32 copy, capture", MUT))
    s.append(cell(240, 150, 190, "int32 copy, step n", CRIT, "5 3"))
    s.append(f'<line x1="60" y1="78" x2="60" y2="148" stroke="currentColor" marker-end="url(#pa4)"/>')
    s.append(f'<text x="68" y="110" font-size="11" fill="{MUT}">.to(int32)</text><text x="68" y="124" font-size="11" fill="{MUT}">at capture</text>')
    s.append(f'<line x1="190" y1="78" x2="320" y2="148" stroke="currentColor" marker-end="url(#pa4)"/>')
    s.append(f'<text x="282" y="96" font-size="11" fill="{MUT}">.to(int32)</text><text x="282" y="110" font-size="11" fill="{MUT}">every step</text>')
    s.append(cell(20, 238, 410, "replayed graph reads the captured address", TD))
    s.append(f'<line x1="60" y1="184" x2="60" y2="236" stroke="{TD}" stroke-width="2" marker-end="url(#pa5)"/>')
    s.append(f'<text x="68" y="215" font-size="11" fill="{TD}">reads the capture-time flags</text>')
    s.append(f'<text x="335" y="205" text-anchor="middle" font-size="11" fill="{CRIT}">never read</text>')
    s.append(cell(480, 150, 260, "int32 buffer, written in place", MUT))
    s.append(f'<text x="610" y="134" text-anchor="middle" font-size="11" fill="{MUT}">same address at every step</text>')
    s.append(cell(480, 238, 410, "replayed graph reads the same address", TD))
    s.append(f'<line x1="520" y1="184" x2="520" y2="236" stroke="{TD}" stroke-width="2" marker-end="url(#pa5)"/>')
    s.append(f'<text x="528" y="215" font-size="11" fill="{TD}">reads this step\'s flags</text>')
    return svg(W, H, "A replayed CUDA graph reads the address recorded at capture, so a flag tensor recreated at every step is never read", "".join(s))


def fig_pair():
    s = [arrow_defs(7), arrow_defs(8, TD)]
    for r, x0 in ((0, 20), (1, 440)):
        cx = x0 + 140
        s += [f'<rect x="{x0}" y="16" width="280" height="264" rx="8" fill="none" stroke="{MUT}"/>',
              f'<text x="{cx}" y="40" text-anchor="middle" font-weight="600" font-size="12" fill="{INK}">{fits(f"CTA {r}, SM {chr(97 + r)}: columns {256 * r}–{256 * r + 255}", 12, 260, "pair")}</text>',
              f'<text x="{cx}" y="62" text-anchor="middle" font-size="11" fill="var(--bdr-ink-2)">tiles {sub("Q", r)}, {sub("K", r)}, {sub("V", r)} in shared memory</text>']
        for j, (y, lab) in enumerate(((84, f'{sub("S", r)} = {sub("Q", r)} {sub("K", r)}ᵀ'), (154, f'S = {sub("S", 0)} + {sub("S", 1)}, P = softmax(S)'), (224, f'{sub("O", r)} += P {sub("V", r)}'))):
            s += [f'<rect x="{x0 + 14}" y="{y}" width="252" height="36" rx="4" fill="none" stroke="{MUT}"/>',
                  f'<text x="{cx}" y="{y + 23}" text-anchor="middle" font-size="11.5" fill="{INK}">{fits(lab, 11.5, 240, "pair")}</text>']
            if j:
                s.append(f'<line x1="{cx}" y1="{y - 32}" x2="{cx}" y2="{y - 2}" stroke="currentColor" marker-end="url(#pa7)"/>')
    s += [f'<line x1="268" y1="104" x2="452" y2="168" stroke="{TD}" stroke-width="2" marker-end="url(#pa8)"/>',
          f'<line x1="472" y1="104" x2="288" y2="168" stroke="{TD}" stroke-width="2" marker-end="url(#pa8)"/>']
    for k, t in enumerate(("partial scores", "m × n, FP16, to", "partner's memory")):
        s.append(f'<text x="370" y="{210 + 15 * k}" text-anchor="middle" font-size="11" fill="{TD}">{t}</text>')
    return svg(740, 300, "Two thread blocks on two SMs each hold half of the head dimension and exchange their partial scores", "".join(s))






def fig_eager():
    """Eager mode against CUDA-graph replay: CPU and GPU lanes, same structure for both."""
    W, H, x0, L, nx = 660, 290, 70, 58, 392
    s = [arrow_defs(9)]
    def row(y0, title, launches, kernels, notes):
        r = [f'<text x="10" y="{y0}" font-size="12.5" font-weight="600" fill="{INK}">{title}</text>',
             f'<text x="{x0 - 10}" y="{y0 + 31}" text-anchor="end" font-size="11.5" fill="{MUT}">CPU</text>',
             f'<text x="{x0 - 10}" y="{y0 + 71}" text-anchor="end" font-size="11.5" fill="{MUT}">GPU</text>']
        for x, t in launches:
            r.append(f'<rect x="{x}" y="{y0 + 14}" width="{L}" height="26" rx="3" fill="none" stroke="{MUT}"/>'
                     f'<text x="{x + L / 2}" y="{y0 + 31}" text-anchor="middle" font-size="11.5" fill="{INK}">{t}</text>')
        for x, t in kernels:
            r.append(f'<rect x="{x}" y="{y0 + 54}" width="38" height="26" rx="3" fill="{MUT}" fill-opacity="0.35"/>'
                     f'<text x="{x + 19}" y="{y0 + 71}" text-anchor="middle" font-size="11.5" fill="{INK}">{t}</text>')
        for k, t in enumerate(notes):
            r.append(f'<text x="{nx}" y="{y0 + 31 + 40 * k}" font-size="11.5" fill="{MUT}">{fits(t, 11.5, W - nx - 4, "eager")}</text>')
        return "".join(r)
    s.append(row(16, "eager mode", [(x0 + i * (L + 4), "launch") for i in range(4)],
                 [(x0 + L + i * (L + 4), f"k{i + 1}") for i in range(4)], ["one launch per kernel", "GPU idle between launches"]))
    s.append(row(140, "CUDA graph", [(x0, "replay")], [(x0 + L + i * 40, f"k{i + 1}") for i in range(4)],
                 ["one launch for the pass", "kernels back to back"]))
    s.append(f'<line x1="{x0}" y1="256" x2="{W - 10}" y2="256" stroke="currentColor" marker-end="url(#pa9)"/>')
    s.append(f'<text x="{(x0 + W) / 2}" y="280" text-anchor="middle" font-size="11.5" fill="{MUT}">time</text>')
    return svg(W, H, "In eager mode the CPU launches each kernel separately and the GPU waits between short kernels; a CUDA graph is launched once and its kernels run back to back", "".join(s))


def fig_run():
    """One measured point: windows of passes in time order, and the eager run of the same point."""
    W, H = 720, 330
    s = [arrow_defs(10)]
    segs = [(10, 156, "prefill", ["the C context", "tokens"]),
            (174, 150, "warm-up", ["2 canvases,", "30 passes"]),
            (332, 186, "profiled window", ["45 passes, profiler on:", "kernels, durations"]),
            (526, 186, "timed window", ["45 passes, profiler off:", "step time t"])]
    s.append(f'<text x="10" y="28" font-size="12.5" font-weight="600" fill="{INK}">graph run (three repeats, different prompts)</text>')
    for x, w, t, lines in segs:
        s.append(tbox(x, 42, w, 78, lines, MUT, t, None, "run"))
    s.append(f'<text x="332" y="142" font-size="11" fill="{MUT}">counter in every MoE layer</text>')
    s.append(tbox(332, 150, 380, 34, ["tokens per expert, per layer and pass"], MUT, None, None, "run"))
    s.append(f'<line x1="10" y1="208" x2="{W - 8}" y2="208" stroke="currentColor" marker-end="url(#pa10)"/>')
    s.append(f'<text x="10" y="228" font-size="11" fill="{MUT}">passes in time order</text>')
    s.append(f'<text x="10" y="264" font-size="12.5" font-weight="600" fill="{INK}">eager run (once)</text>')
    s.append(tbox(10, 274, 420, 34, ["same point; each kernel launched from its module"], MUT, None, None, "run"))
    return svg(W, H, "One measured point: after prefill and warm-up, a profiled window records every kernel, then a timed window without the profiler gives the step time; an expert counter covers both windows; one eager run of the same point follows", "".join(s))


def fig_attrib():
    """Kernel labelling: the eager trace gives each kernel its module; graph kernels take the label
    of the eager kernel with the same name and launch shape, in order of occurrence."""
    W, H, x0, bw, gap = 790, 270, 136, 98, 11
    ks = ["norm", "gemm_fp8", "attn_fwd", "gemm_fp8", "moe_gemm1", "moe_gemm2"]
    mods = ["norm", "qkv proj.", "attention", "out proj.", "experts", "experts"]
    comps = ["outside", "dense", "attention", "dense", "experts", "experts"]
    X = lambda i: x0 + i * (bw + gap)
    s = [arrow_defs(11)]
    left = [(62, "eager run", 12.5, 600, INK), (78, "name, module", 11, 400, MUT),
            (112, "match: same", 11, 400, MUT), (126, "name and shape,", 11, 400, MUT), (140, "in order", 11, 400, MUT),
            (184, "graph run", 12.5, 600, INK), (200, "name only", 11, 400, MUT), (240, "component", 11, 400, MUT)]
    for y, t, fs, fw, col in left:
        s.append(f'<text x="10" y="{y}" font-size="{fs}" font-weight="{fw}" fill="{col}">{fits(t, fs, x0 - 16, "attrib")}</text>')
    for i, (k, m, c) in enumerate(zip(ks, mods, comps)):
        x = X(i)
        s.append(f'<text x="{x + bw / 2}" y="28" text-anchor="middle" font-size="11" fill="{MUT}">{fits(m, 11, bw + gap - 4, "attrib")}</text>')
        s.append(f'<path d="M{x + 2},40 v-5 h{bw - 4} v5" fill="none" stroke="{MUT}"/>')
        s.append(f'<rect x="{x}" y="46" width="{bw}" height="36" rx="4" fill="var(--bdr-surface)" stroke="{MUT}"/>'
                 f'<text x="{x + bw / 2}" y="68" text-anchor="middle" font-size="12" fill="{INK}">{fits(k, 12, bw - 8, "attrib")}</text>')
        s.append(f'<line x1="{x + bw / 2}" y1="84" x2="{x + bw / 2}" y2="164" stroke="{MUT}" stroke-dasharray="3 3" marker-end="url(#pa11)"/>')
        s.append(f'<rect x="{x}" y="168" width="{bw}" height="36" rx="4" fill="var(--bdr-surface)" stroke="{MUT}"/>'
                 f'<text x="{x + bw / 2}" y="190" text-anchor="middle" font-size="12" fill="{INK}">{k}</text>')
        s.append(f'<text x="{x + bw / 2}" y="240" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">{c}</text>')
    return svg(W, H, "Graph-run kernels carry their names but not their modules; each takes the module of the eager kernel with the same name and launch shape, in order", "".join(s))


# ---------------------------------------------------------------- tables
def t_ceilings(d):
    ce = d["ceil"]
    rows = [["FP8 GEMM (TFLOP/s)", "1,979", f"{ce['pi_fp8'] / 1e12:,.0f}", f"{ce['pi_fp8'] / 1979e12:.2f}", "the model's own projections, CUDA-graph bursts of 2 ms"],
            ["BF16 GEMM (TFLOP/s)", "989", f"{ce['pi_bf16'] / 1e12:,.0f}", f"{ce['pi_bf16'] / 989e12:.2f}", "8,192³, short burst"],
            ["attention (TFLOP/s)", "989", f"{ce['pi_attn'] / 1e12:,.0f}", f"{ce['pi_attn'] / 989e12:.2f}", "FlashAttention 3, head dim. 128, 2 × 16k tokens, sustained"],
            ["HBM bandwidth (TB/s)", "4.80", f"{ce['beta'] / 1e12:.2f}", f"{ce['beta'] / 4.8e12:.2f}", "read of a 4 GB buffer, sustained"]]
    return table("Table 11. Datasheet and empirical ceilings of one H200 SXM. The empirical ceiling is the highest rate any measurement reached on random data; the ridge point for FP8 linear layers moves from 412 to "
                 f"{ce['pi_fp8'] / ce['beta']:.0f} FLOP/byte.", ["rate", "datasheet", "empirical", "ratio", "measured on"], rows)


def t_neff(d):
    R, Bs, Cs = d["routing"], d["Bs"], d["Cs"]
    rows = []
    for B in Bs:
        r = [f"B = {B}"]
        for m in ("td", "ar"):
            for C in (1024, 65536):
                v = R[m].get(f"{B},{C}")
                r.append(f"{v['neff']:.1f}" if v and v.get("neff") else "–")
        rows.append(r)
    return table("Table 12. Effective number of independent tokens per pass, \\(N^{\\mathrm{eff}}\\), the number of uniformly routed independent tokens that would touch the measured \\(U\\) (Definition 6). A TD pass holds \\(256B\\) positions, an AR step \\(B\\).",
                 ["", "TD, 1k", "TD, 64k", "AR, 1k", "AR, 64k"], rows)


def t_prop1(d):
    p = d["prop1"]["fp8_empirical"]["pts"]
    g = d["gains"]
    rows = []
    for B in d["Bs"]:
        k = f"{B},1024"
        if k not in p:
            continue
        x = p[k]
        rows.append([f"B = {B}", f"{x['Pi_ind']:.2f}", f"{x['Pi_meas']:.2f}", f"{x['gain_ind']:.2f}", f"{x['gain_meas']:.2f}", f"{g[k]['measured']:.2f}"])
    return table("Table 13. Proposition 1 at independent and at measured routing, against the measured gain; DiffusionGemma against Gemma 4 26B-A4B on one H200, FP8 weights, empirical ceilings, context 1k. Proposition 1 leaves out the vocabulary head, activations and kernel efficiency.",
                 ["", "\\(\\Pi_{\\mathrm{lin}}\\), independent", "\\(\\Pi_{\\mathrm{lin}}\\), measured", "\\(\\Gamma\\), Prop. 1, independent", "\\(\\Gamma\\), Prop. 1, measured routing", "\\(\\Gamma\\), measured"], rows)


COMP = [("experts", "experts"), ("dense", "dense GEMMs and router"), ("attnW", "window attention"), ("attnG", "global attention"), ("head", "vocabulary head and sampler")]


def t_eta(d):
    rows = []
    # the points with batch 8 and above that both campaigns measured
    common = set(d["eta3"]["td"]) & set(d["eta2"]["td"])
    big = lambda k: int(k.split(",")[0]) >= 8 and k in common
    for c, name in COMP:
        r = [name]
        for src in ("eta2", "eta3"):
            r.append(rng([v.get(c) for k, v in d[src]["td"].items() if big(k)]))
        r.append(rng([v.get(c) for k, v in d["eta3"]["ar"].items() if big(k)]))
        rows.append(r)
    rows.append(["whole step", rng([v["step"] for k, v in d["eta2"]["td"].items() if big(k)]), rng([v["step"] for k, v in d["eta3"]["td"].items() if big(k)]),
                 rng([v["step"] for k, v in d["eta3"]["ar"].items() if big(k)])])
    return table("Table 15. Efficiency against the empirical roofline, \\(\\eta_c = r_c/t_c\\), at the 13 points with batch 8 and above that both sets of runs share. Campaign 2: the three engine fixes of Part 9 with vLLM's kernels; campaign 3: the same with the two kernels of this Part. AR's kernels are unchanged between the two.",
                 ["component", "TD, stock kernels", "TD, new kernels", "AR"], rows)


def t_stock(d):
    rows = []
    for c, name in COMP:
        rows.append([name] + [rng([v.get(c) for k, v in d["eta2"]["td"].items() if int(k.split(",")[0]) in Bset]) for Bset in ((1, 2, 4), (8, 16, 32, 64))])
    return table("Table 14. TD with the three engine fixes and vLLM's own kernels (campaign 2): efficiency of each component against its empirical roofline, at batch 1–4 and at batch 8–64, over all contexts.",
                 ["component", "B = 1–4", "B = 8–64"], rows)


def t_vs2(d):
    v = d["vs2"]
    def r(m, Bp, C):
        xs = [x["ratio"] for k, x in v[m].items() if Bp(int(k.split(",")[0])) and int(k.split(",")[1]) == C]
        return rng(xs)
    rows = [[kc(C), r("td", lambda b: b == 1, C), r("td", lambda b: b >= 2, C), r("ar", lambda b: True, C)] for C in d["Cs"]]
    return table("Table 16. Throughput with the new kernels (campaign 3) over throughput with vLLM's kernels (campaign 2), medians of three repeats per point. AR's kernels did not change, so its ratio measures the drift between the two days.",
                 ["context", "TD, B = 1", "TD, B = 2–64", "AR, all B"], rows)


def t_moe():
    return table("Table 17. The expert path per denoising pass in the engine, vLLM's kernel against the new one, and its efficiency against the empirical roofline at the measured routing. These are dedicated runs of the expert kernel (three per arm), separate from the grid of Parts 8–11, so their efficiencies differ slightly from Table 15's.",
                 ["TD point (B, C)", "expert path per pass", "fraction of empirical roofline", "TD step"],
                 [["(8, 1k)", "15.03 → 12.23 ms (−19%)", "0.27 → 0.33", "43.56 → 40.07 ms (−8.0%)"],
                  ["(32, 1k)", "46.60 → 34.26 ms (−26%)", "0.34 → 0.47", "158.75 → 147.81 ms (−6.9%)"],
                  ["(32, 16k)", "45.59 → 33.37 ms (−27%)", "0.35 → 0.48", "227.44 → 217.54 ms (−4.4%)"],
                  ["(64, 1k)", "85.78 → 61.61 ms (−28%)", "0.37 → 0.52", "326.30 → 299.23 ms (−8.3%)"]])


def t_attn():
    return table("Table 18. Global attention in the engine, TFLOP/s and fraction of the 638 TFLOP/s attention ceiling, one run each; the last column is the TD step with FA4 at 64-token pages against the pair kernel at 80-token pages.",
                 ["(B, C)", "FA4, page 64", "pair kernel, page 80", "TD step"],
                 [["(8, 16k)", "335 (0.52)", "455 (0.71)", "58.05 → 53.27 ms (−8.2%)"],
                  ["(32, 16k)", "320 (0.50)", "436 (0.68)", "222.37 → 208.24 ms (−6.4%)"],
                  ["(8, 64k)", "330 (0.52)", "453 (0.71)", "107.92 → 88.47 ms (−18.0%)"]])


def t_recover(d):
    rec = d["recoverable"]["td"]
    names = dict(COMP + [("outside", "outside the model"), ("idle", "idle")])
    rows = []
    for B in d["Bs"]:
        r = [f"B = {B}"]
        for C in (1024, 65536):
            x = rec.get(f"{B},{C}")
            if not x:
                r.append("–")
                continue
            top = sorted(x.items(), key=lambda kv: -kv[1])[:2]
            r.append("; ".join(f"{names[c]} {100 * v:.0f}%" for c, v in top))
        rows.append(r)
    return table("Table 19. The two largest recoverable shares of TD's step after the kernel work: measured time minus the time of the target kernel (the best standalone kernel at the same shape, else 0.8 of the empirical roofline), as a share of the step. Time outside the model and idle time count in full.",
                 ["", "context 1k", "context 64k"], rows)


# ---------------------------------------------------------------- assembly
def blocks():
    d = json.loads(DATA.read_text())
    sc_dev = max(abs(math.log(r["measured"] / r["roof"])) for r in d["gains"].values())
    dev = heat(d, lambda r: r["measured"] / r["roof"], "(e) measured / bound (c)", sc_dev,
               color=lambda v: (TD if v >= 1 else AR, min(1, abs(math.log(v)) / sc_dev)))
    sc4 = gain_scale(d, ["roof_ind_ds", "roof_ind", "roof", "measured"])
    m4 = lambda specs: '<div class="pfig-grid">' + "".join(f"<div>{heat(d, k, t, sc4)}</div>" for k, t in specs) + "</div>"
    out = {
        "decoders": fig("fig-decoders", "Figure 7", "In the same number of passes, AR commits one token per pass and TD a whole canvas.",
                        "", fig_decoders(),
                        "One sequence, DiffusionGemma's setting: a canvas of \\(b = 256\\) positions, \\(S = 14\\) denoising passes and one commit pass, so \\(\\omega = 15\\) passes per canvas and \\(\\mathrm{TPF} = b/\\omega = 17.1\\) tokens per forward pass. Each TD pass processes 256 positions instead of one, so it does more FLOPs and its positions touch more experts; Parts 1–5 weigh these costs against the shared read of the weights."),
        "map": fig("fig-map", "Figure 8", "The book in two halves: bounds on paper, then measurements on one H200.", "", fig_map(),
                   "Parts 1–7 derive the gain \\(\\Gamma\\) as a ratio of roofline bounds and map it over batch and context. Parts 8–11 take the inputs of those bounds from measurements, time every kernel of both decoders in vLLM, rewrite the two kernels that lag most, and compare the measured gain with the bound.", wide=True),
        "ladder": fig("fig-ladder", "Figure 9", "From the book's bound to the measured gain in four steps.", "", fig_ladder(d),
                      "DiffusionGemma (TD) against Gemma 4 26B-A4B (AR) on one H200, FP8 weights, at two of the 25 measured points. (a) is the book's accounting at FP8 with datasheet rates and independent routing. Each later column changes one input; Part 11 shows all four as maps over batch and context.", wide=True),
        "roofline": fig("fig-eff-roofline", "Figure 10", "The measured roofline sits below the datasheet's, more in compute than in bandwidth.", "", fig_eff_roofline(d),
                        "Attainable throughput \\(\\min(\\pi, I\\beta)\\) against arithmetic intensity, log–log, for FP8 linear layers. Dashed: datasheet (1,979 TFLOP/s, 4.8 TB/s). Solid orange: empirical ceilings of Table 11. Blue: the attention ceiling, which applies to attention kernels whatever their intensity. The ridge point, where a kernel turns compute-bound, moves from 412 to 340 FLOP/byte."),
        "routing": fig("fig-eff-routing", "Figure 11", "Both decoders touch fewer experts than independent routing predicts, for different reasons.", "", fig_eff_routing(d),
                       "Distinct experts per MoE layer and forward pass \\(U\\), averaged over the 30 layers and the timed passes, against batch size at context 1k: the median of three repeats with different prompts, and the band spanning them. Dotted: independent tokens with uniform routing, \\(U_{E,k}(N)\\) of Definition 1. Dashed: independent tokens with each expert's measured popularity (Definition 6). AR follows the dashed curve; TD lies below it.", wide=True),
        "maps_abc": fig("fig-eff-maps", "Figure 12", "The gain bound moves with each measured input, and routing moves it most.", "",
                        m4([("roof_ind_ds", "(a) datasheet, independent routing"), ("roof_ind", "(b) empirical ceilings"), ("roof", "(c) and measured routing")]),
                        "Roofline bound on \\(\\Gamma\\) over batch \\(B\\) and context \\(C\\), with FP8 weights. Each map changes one input of the previous one. Orange: TD faster; blue: AR faster; the shade grows with \\(|\\log\\Gamma|\\) on one scale shared by Figures 12 and 20. Missing cells did not fit in memory.", wide=True),
        "eager": fig("fig-eager", "Figure 13", "In eager mode the GPU waits for each launch; a CUDA graph runs the kernels of a pass back to back.", "", fig_eager(),
                     "Four kernels of one forward pass, as in Part 8's small GEMMs: launching a kernel costs the CPU a few µs, and with 64 tokens or fewer a projection runs for 5–12 µs, so in eager mode the GPU waits a large share of the time. A CUDA graph replays the recorded sequence of kernels with one launch."),
        "run": fig("fig-run", "Figure 14", "One measured point: a profiled window for the kernels, a timed window for the step time.", "", fig_run(),
                   "TD at one batch size \\(B\\) and context \\(C\\). The profiler slows the passes down, so the step time comes from the second window, where it is off. For AR the windows are 48 steps each, after 1,488 warm-up steps, which put AR's context at the middle of the windows at TD's.", wide=True),
        "attrib": fig("fig-attrib", "Figure 15", "Each kernel of the graph run takes its module from the eager run.", "", fig_attrib(),
                      "A stretch of one layer, with simplified kernel names. In the eager run each kernel is recorded inside the profiler range of the module that launched it; in the graph run only inside the range of the whole pass. The same kernel can serve several modules, as the FP8 GEMM does here, so the match counts occurrences. Each module then belongs to one component.", wide=True),
        "flags": fig("fig-flags", "Figure 16", "A replayed CUDA graph reads the memory it saw at capture, so a tensor recreated at every step is never read.", "", fig_flags(),
                     "DiffusionGemma passes a causal flag per sequence to the attention kernel: 0 for a bidirectional denoising pass, 1 for the causal commit pass. In vLLM 0.28 the flags lived in a bool buffer converted to int32 at every step; under CUDA graphs the attention kernels kept reading the copy made at capture.", wide=True),
        "moe": fig("fig-moe", "Figure 17", "The new expert path removes three passes over memory; its two GEMMs are as fast as vLLM's.", "", fig_moe(),
                   "One MoE layer at TD batch 32 (\\(Nk = 65{,}536\\) rows, each a (token, expert) pair). Each box is one kernel, in order; the orange number is what it writes to memory, read back by the next kernel.", wide=True),
        "pair": fig("fig-pair", "Figure 18", "Each thread block of the pair holds half of the head dimension; the only traffic between the two SMs is the partial score tile.", "", fig_pair(),
                    "Both blocks process the same 128 query rows. The orange arrows are the exchange: each block writes its \\(m \\times n\\) partial scores into the partner's shared memory and waits for the partner's before the softmax."),
        "smem": fig("fig-smem", "Figure 19", "Splitting the head dimension makes room for a second pipeline stage.", "", fig_smem(),
                    "Shared memory per thread block, in KB, against the 227 KB of one SM. FA4 at head dimension 512 holds full-width tiles and fits one stage; a block of the pair holds half-width tiles and fits two stages of 64 keys plus two exchange buffers. With 80-token KV pages the fastest configuration uses the room differently, for one stage of 80-key tiles loaded by TMA (Table 18)."),
        "maps_cd": fig("fig-conc-maps", "Figure 20", "The measured gain against the bound at measured routing and empirical ceilings.", "",
                       m4([("roof", "(c) bound, measured inputs"), ("measured", "(d) measured")]),
                       "Same scale as Figure 12. Medians of three repeats; the measured gain is TD's tokens per second over AR's at equal \\(B\\) and \\(C\\).", wide=True),
        "dev": fig("fig-conc-dev", "Figure 21", "The bound is conservative up to batch 8 and optimistic from batch 16 on.", "", f'<div class="pfig-grid"><div>{dev}</div></div>',
                   "Measured gain over the bound (c), which equals \\(\\eta_{TD}/\\eta_{AR}\\) by Proposition 9. Orange where the measured gain exceeds the bound, blue where it falls short; the shade grows with the distance from 1 on its own log scale."),
        "t_ceil": t_ceilings(d), "t_neff": t_neff(d), "t_prop1": t_prop1(d), "t_stock": t_stock(d), "t_eta": t_eta(d), "t_vs2": t_vs2(d),
        "t_moe": t_moe(), "t_attn": t_attn(), "t_recover": t_recover(d),
    }
    if FIT:
        raise SystemExit("labels do not fit:\n  " + "\n  ".join(FIT))
    return out


def numbers():
    """Values quoted in the text, for the checks in bin/build-roofline.py."""
    return json.loads(DATA.read_text())


if __name__ == "__main__":
    b = blocks()
    print({k: len(v) for k, v in b.items()})
