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
           f'class="pfig-svg{" pfig-wide" if w >= 560 else ""}" style="max-width:{w}px"{extra}>' + body + "</svg>")
    # wide diagrams keep a readable size on phones and scroll sideways instead of shrinking
    return f'<div class="pfig-scroll">{out}</div>' if w >= 560 else out


def arrow_defs(i, color="currentColor"):
    return (f'<defs><marker id="pa{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker></defs>')


def fig(fid, kicker, title, lede, inner, cap, wide=False):
    return (f'  <div class="bdr-fig pfig{" breakout" if wide else ""}" id="{fid}">\n'
            f'    <div class="fig-k">{kicker}</div>\n    <p class="fig-title">{title}</p>\n'
            + (f'    <p class="fig-lede">{lede}</p>\n' if lede else "")
            + f'    {inner}\n    <p class="fig-cap">{cap}</p>\n  </div>\n')


def table(caption, head, rows, cls="pt", fold=None):
    """A table; with `fold` (the figure it backs, e.g. "Figure 25"), collapsed under a disclosure."""
    th = "".join(f'<th class="l">{h}</th>' for h in head)
    tr = "".join("<tr>" + "".join(f'<td class="l">{c}</td>' for c in r) + "</tr>" for r in rows)
    t = (f'  <div class="table-wrap nofold"><table class="{cls}">\n    <caption>{caption}</caption>\n'
         f'    <thead><tr>{th}</tr></thead>\n    <tbody>{tr}</tbody>\n  </table></div>\n')
    if not fold:
        return t
    n = re.match(r"Table (\d+)", caption).group(1)
    return f'  <details class="numbers"><summary>Table {n}: the numbers behind {fold}</summary>\n{t}  </details>\n'



def rng(xs, n=2):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return "–"
    lo, hi = f"{min(xs):.{n}f}", f"{max(xs):.{n}f}"
    return lo if lo == hi else f"{lo}–{hi}"


# ---------------------------------------------------------------- Part 0: diagrams


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
            v = None if v is None or (isinstance(v, float) and math.isnan(v)) else v
            x = ml + j * cw
            if v is None:
                s.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" fill="{RULE}" fill-opacity="0.5"/>'
                         f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" font-size="11.5" fill="{MUT}">–</text>')
                continue
            fill, a = (color or (lambda v: (TD if v >= 1 else AR, min(1, abs(math.log(v)) / scale))))(v)
            a = max(0.06, a)
            txt = "var(--bdr-bg)" if a > 0.55 else INK  # page background: white on dark cells, dark on bright cells in dark mode
            s.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" fill="{fill}" fill-opacity="{a:.2f}"><title>B = {B}, C = {kc(C)}: {(fmtv or (lambda v: f"{v:.2f}"))(v)}</title></rect>'
                     f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" font-size="11.5" style="fill:{txt};font-weight:{600 if a > 0.55 else 400}">{(fmtv or (lambda v: f"{v:.2f}"))(v)}</text>')
    s.append(f'<text x="{ml + cw * len(Cs) / 2}" y="{H - 6}" text-anchor="middle" font-size="11.5" fill="{MUT}">context C</text>')
    return svg(W, H, title, "".join(s))


def stepper(name, panels, label="Show"):
    """Panels shown one at a time, chosen by buttons (reading.js); without JavaScript all are shown."""
    btns = "".join(f'<button type="button" data-k="{i}" aria-pressed="{str(i == 0).lower()}">{t}</button>' for i, (t, _) in enumerate(panels))
    body = "".join(f'<div class="pfig-step" data-k="{i}">{h}</div>' for i, (_, h) in enumerate(panels))
    return (f'<div class="pfig-steps" data-name="{name}"><div class="pfig-step-buttons" role="group" aria-label="{label}">'
            f'<span>{label}</span>{btns}</div>{body}</div>')


ETA_COMPS = [("step", "whole step"), ("experts", "experts"), ("dense", "dense layers"), ("attnG", "global attention"),
             ("attnW", "window attention"), ("head", "head, sampler, self-cond.")]


def eta_heat(d, mode, comp, title, src="eta3"):
    """Efficiency map of one component (or the whole step) of one decoder, on a 0-1 scale;
    `src` is "eta3" (after the kernel work) or "eta2" (vLLM's own kernels)."""
    e = d[src][mode]
    g = {k: {"v": (v.get(comp) if comp != "step" else v.get("step"))} for k, v in e.items()}
    col = TD if mode == "td" else AR
    return heat({"gains": g, "Bs": d["Bs"], "Cs": d["Cs"]}, lambda r: r["v"], title, 1,
                color=lambda v: (col, min(1.0, v)), fmtv=lambda v: f"{v:.2f}")


def gain_scale(d, keys):
    vals = [r[k] for r in d["gains"].values() for k in keys if r.get(k)]
    return max(abs(math.log(v)) for v in vals)


def maps(d, specs):
    sc = gain_scale(d, [k for k, _ in specs])
    return '<div class="pfig-grid">' + "".join(f"<div>{heat(d, k, t, sc)}</div>" for k, t in specs) + "</div>"


# ---------------------------------------------------------------- Part 9: diagrams
# ---------------------------------------------------------------- Part 10: diagrams (from the kernel note)





CHW = 0.61  # IBM Plex Mono advance width, in em


def tw(text, fs):
    """Estimated rendered width of a monospace label (tspans count by their characters)."""
    return len(re.sub(r"<[^>]+>", "", text)) * CHW * fs


FIT = []  # labels that do not fit their box, reported by blocks()


def fits(text, fs, width, where):
    if tw(text, fs) > width:
        FIT.append(f"{where}: {re.sub(r'<[^>]+>', '', text)!r} needs {tw(text, fs):.0f} px, has {width}")
    return text


BOOK_URL = ""  # the map is on the book's first page, so chapter links are relative to it


def tbox(x, y, w, h, lines, col=MUT, title=None, part=None, where="", href=None):
    """A box with an optional small part label, a bold title and lines of text, left-aligned;
    with `href`, the whole box is a link."""
    r = [f'<a href="{href}" class="pfig-link">' if href else "", f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="var(--bdr-surface)" stroke="{col}" stroke-width="1.4"/>']
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
    if href:
        r.append("</a>")
    return "".join(r)




def fig_eff_roofline(d):
    ce = d["ceil"]
    W, H, ml, mr, mt, mb = 660, 360, 62, 150, 24, 50
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
        W, H, ml, mr, mt, mb = 320, 300, 40, 84, 50, 44
        X = lambda i: ml + i / (len(Bs) - 1) * (W - ml - mr)
        Y = lambda u: mt + (1 - u / 140) * (H - mt - mb)
        s = [f'<text x="{ml - 40}" y="16" font-size="12.5" font-weight="600" fill="{col}">{fits(name, 12.5, W - 10, "routing")}</text>',
             f'<text x="{ml - 36}" y="34" font-size="10.5" fill="{MUT}">distinct experts per layer, pass: U</text>']
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





def fig_eager():
    """Eager mode against CUDA-graph replay: CPU and GPU lanes, same structure for both."""
    W, H, x0, L, K, nx = 470, 214, 46, 46, 30, 306
    s = [arrow_defs(9)]
    def row(y0, title, launches, kernels, notes):
        r = [f'<text x="6" y="{y0}" font-size="12" font-weight="600" fill="{INK}">{title}</text>',
             f'<text x="{x0 - 8}" y="{y0 + 23}" text-anchor="end" font-size="11" fill="{MUT}">CPU</text>',
             f'<text x="{x0 - 8}" y="{y0 + 51}" text-anchor="end" font-size="11" fill="{MUT}">GPU</text>']
        for x, t in launches:
            r.append(f'<rect x="{x}" y="{y0 + 9}" width="{L}" height="20" rx="3" fill="none" stroke="{MUT}"/>'
                     f'<text x="{x + L / 2}" y="{y0 + 23}" text-anchor="middle" font-size="10.5" fill="{INK}">{t}</text>')
        for x, t in kernels:
            r.append(f'<rect x="{x}" y="{y0 + 37}" width="{K}" height="20" rx="3" fill="{MUT}" fill-opacity="0.35"/>'
                     f'<text x="{x + K / 2}" y="{y0 + 51}" text-anchor="middle" font-size="10.5" fill="{INK}">{t}</text>')
        for k, t in enumerate(notes):
            r.append(f'<text x="{nx}" y="{y0 + 23 + 28 * k}" font-size="11" fill="{MUT}">{fits(t, 11, W - nx - 2, "eager")}</text>')
        return "".join(r)
    s.append(row(14, "eager mode", [(x0 + i * (L + 4), "launch") for i in range(4)],
                 [(x0 + L + i * (L + 4), f"k{i + 1}") for i in range(4)], ["a launch per kernel", "GPU waits"]))
    s.append(row(106, "CUDA graph", [(x0, "replay")], [(x0 + L + i * (K + 2), f"k{i + 1}") for i in range(4)],
                 ["one launch", "back to back"]))
    s.append(f'<line x1="{x0}" y1="186" x2="{nx - 16}" y2="186" stroke="currentColor" marker-end="url(#pa9)"/>')
    s.append(f'<text x="{(x0 + nx - 16) / 2}" y="206" text-anchor="middle" font-size="11" fill="{MUT}">time</text>')
    return svg(W, H, "In eager mode the CPU launches each kernel separately and the GPU waits between short kernels; a CUDA graph is launched once and its kernels run back to back", "".join(s))


def fig_run():
    """One measured point: windows of passes in time order, and the eager run of the same point."""
    W, H = 600, 236
    s = [arrow_defs(10)]
    def box(x, y, w, h, title, lines):
        r = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="var(--bdr-surface)" stroke="{MUT}"/>']
        yy = y + 16
        if title:
            r.append(f'<text x="{x + 8}" y="{yy}" font-size="11.5" font-weight="600" fill="{INK}">{fits(title, 11.5, w - 14, "run")}</text>')
            yy += 15
        for t in lines:
            r.append(f'<text x="{x + 8}" y="{yy}" font-size="10.5" fill="var(--bdr-ink-2)">{fits(t, 10.5, w - 14, "run")}</text>')
            yy += 14
        return "".join(r)
    s.append(f'<text x="6" y="16" font-size="12" font-weight="600" fill="{INK}">graph run, three repeats</text>')
    segs = [(6, 96, "prefill", ["C tokens,", "first canvas"]), (106, 104, "warm-up", ["1 canvas,", "15 passes"]),
            (214, 188, "profiled window", ["2 canvases, 30 passes,", "profiler on: kernels"]),
            (406, 188, "timed window", ["2 canvases, 30 passes,", "profiler off: step time t"])]
    for x, w, t, lines in segs:
        s.append(box(x, 26, w, 62, t, lines))
    s.append(box(214, 96, 380, 24, None, ["expert counter: tokens per expert, layer, pass"]))
    s.append(f'<line x1="6" y1="138" x2="{W - 6}" y2="138" stroke="currentColor" marker-end="url(#pa10)"/>')
    s.append(f'<text x="6" y="156" font-size="10.5" fill="{MUT}">passes in time order</text>')
    s.append(f'<text x="6" y="190" font-size="12" font-weight="600" fill="{INK}">eager run, once</text>')
    s.append(box(6, 200, 350, 24, None, ["same point; kernels launched from their modules"]))
    return svg(W, H, "One measured point: after prefill and warm-up, a profiled window records every kernel, then a timed window without the profiler gives the step time; an expert counter covers both windows; one eager run of the same point follows", "".join(s))



def fig_decoders():
    W, H, x0, sw, gap = 600, 236, 46, 26, 3
    n = 15
    s = [arrow_defs(0)]
    s.append(f'<text x="{x0}" y="18" font-size="11" fill="{MUT}">forward passes in time order; each reads the weights once</text>')
    for row, (name, col, y) in enumerate((("AR", AR, 32), ("TD", TD, 116))):
        s.append(f'<text x="{x0 - 10}" y="{y + 21}" text-anchor="end" font-size="12" font-weight="600" fill="{col}">{name}</text>')
        for i in range(n):
            x = x0 + i * (sw + gap)
            solid = row == 0 or i == n - 1  # a denoising pass is outlined: it commits nothing
            s.append(f'<rect x="{x + 0.6}" y="{y + 0.6}" width="{sw - 1.2}" height="30.8" rx="3" fill="{col}" fill-opacity="{0.85 if solid else 0.16}"'
                     f'{"" if solid else f" stroke=\"{col}\" stroke-width=\"1.2\""}/>')
            if row == 0:
                s.append(f'<text x="{x + sw / 2}" y="{y + 20}" text-anchor="middle" font-size="10.5" style="fill:#fff">1</text>')
        xr = x0 + n * (sw + gap) + 6
        s.append(f'<text x="{xr}" y="{y + 14}" font-size="12" fill="{INK}">{"15 tokens" if row == 0 else "256 tokens"}</text>')
        s.append(f'<text x="{xr}" y="{y + 29}" font-size="11" fill="{MUT}">{"one per pass" if row == 0 else "one canvas"}</text>')
    s.append(f'<text x="{x0}" y="84" font-size="11" fill="{MUT}">each pass commits one token per sequence</text>')
    s.append(f'<path d="M{x0},154 v6 h{14 * (sw + gap) - gap} v-6" fill="none" stroke="{TD}" stroke-width="1.2"/>')
    s.append(f'<text x="{x0 + 7 * (sw + gap)}" y="176" text-anchor="middle" font-size="11" fill="{INK}">S = 14 denoising passes over the 256 positions</text>')
    xc = x0 + 14 * (sw + gap) + sw / 2
    s.append(f'<line x1="{xc}" y1="200" x2="{xc}" y2="152" stroke="{TD}" marker-end="url(#pa0)"/>')
    s.append(f'<text x="{xc}" y="216" text-anchor="middle" font-size="11" fill="{INK}">commit pass</text>')
    wide = svg(W, H, "In the same 15 passes AR commits 15 tokens and TD commits a canvas of 256", "".join(s))
    return f'<div class="v-wide">{wide}</div><div class="v-narrow">{fig_decoders_narrow()}</div>'


def fig_decoders_narrow():
    """Figure 1 for a phone: the same two rows of 15 passes, with the counts under each row."""
    W, H, x0, sw, gap, n = 360, 214, 30, 18, 3, 15
    s = [arrow_defs(30)]
    s.append(f'<text x="{x0}" y="13" font-size="11" fill="{MUT}">passes in time order; each reads the weights</text>')
    for row, (name, col, y) in enumerate((("AR", AR, 22), ("TD", TD, 92))):
        s.append(f'<text x="{x0 - 6}" y="{y + 17}" text-anchor="end" font-size="11.5" font-weight="600" fill="{col}">{name}</text>')
        for i in range(n):
            x = x0 + i * (sw + gap)
            solid = row == 0 or i == n - 1
            s.append(f'<rect x="{x + 0.6}" y="{y + 0.6}" width="{sw - 1.2}" height="24.8" rx="3" fill="{col}" fill-opacity="{0.85 if solid else 0.16}"'
                     f'{"" if solid else f" stroke=\"{col}\" stroke-width=\"1.2\""}/>')
    s.append(f'<text x="{x0}" y="66" font-size="11" fill="{INK}">15 tokens, one per pass</text>')
    s.append(f'<path d="M{x0},124 v5 h{14 * (sw + gap) - gap} v-5" fill="none" stroke="{TD}" stroke-width="1.2"/>')
    s.append(f'<text x="{x0}" y="145" font-size="11" fill="{INK}">S = 14 denoising passes</text>')
    xc = x0 + 14 * (sw + gap) + sw / 2
    s.append(f'<line x1="{xc}" y1="160" x2="{xc}" y2="122" stroke="{TD}" marker-end="url(#pa30)"/>')
    s.append(f'<text x="{xc}" y="174" text-anchor="end" font-size="11" fill="{INK}">commit pass</text>')
    s.append(f'<text x="{x0}" y="200" font-size="11" fill="{INK}">256 tokens, one canvas</text>')
    return svg(W, H, "In the same 15 passes AR commits 15 tokens and TD commits a canvas of 256", "".join(s))


def fig_map_narrow(base=BOOK_URL):
    """Figure 2 for a phone: the parts as a list of links."""
    top = [p for _, _, parts in MAP_TOP for p in parts]
    li = lambda slug, part, title, lines: (f'<li><a href="{base}{slug}/"><span class="mn-p">{part.title()}</span> '
                                           f'<b>{title}</b>{(" · " + " ".join(lines)) if lines else ""}</a></li>')
    return ('<div class="map-narrow"><div class="mn-h">On paper · Parts 1–7</div><ol>' + "".join(li(*p) for p in top) + '</ol>'
            '<div class="mn-h">On an H200 · Parts 8–11</div><ol class="mn-h200">' + "".join(li(*p) for p in MAP_BOT) + '</ol>'
            '<div class="mn-also">Also: ' + ", ".join(f'<a href="{base}{slug}/">{t}</a> ({w})' for slug, t, w in MAP_SIDE) + '</div></div>')


# The book map: one box per part, every box a link to its chapter.
MAP_TOP = [  # (x, width, [(slug, part label, title, lines)]); two entries are stacked in one column
    (10, 100, [("setup", "PART 1", "Setup", ["costs F, Q,", "penalties"])]),
    (122, 100, [("gain", "PART 2", "The gain", ["Prop. 1"])]),
    (234, 126, [("linear", "PART 3", "Linear part", []), ("attention", "PART 4", "Attention part", [])]),
    (372, 122, [("whole", "PART 5", "Whole picture", ["regimes, maps"])]),
    (506, 144, [("calculator", "PART 6", "Calculator", []), ("examples", "PART 7", "Worked examples", [])]),
]
MAP_BOT = [("effective", "PART 8", "Effective rooflines", ["measured π, β", "and experts U"]),
           ("stack", "PART 9", "Measurement stack", ["time per component"]),
           ("kernels", "PART 10", "Kernels", ["two rewrites"]),
           ("conclusion", "PART 11", "Conclusion", ["measured Γ", "against the bound"])]
MAP_SIDE = [("summary", "Summary", "the results first"), ("appendix", "Appendix", "exact accounting, sources")]


def linkbox(x, y, w, h, href, part, title, lines, col, tip):
    """One part of the book: a box that is a link to its chapter."""
    r = [f'<a href="{href}" class="pfig-link"><title>{tip}</title>'
         f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="var(--bdr-surface)" stroke="{col}" stroke-width="1.4"/>']
    yy = y + 16
    r.append(f'<text x="{x + 10}" y="{yy}" font-size="10" letter-spacing=".06em" fill="{MUT}">{fits(part, 11, w - 20, "map")}</text>')
    yy += 16
    r.append(f'<text x="{x + 10}" y="{yy}" font-size="11.5" font-weight="600" fill="{INK}">{fits(title, 11.5, w - 20, "map")}</text>')
    for t in lines:
        yy += 15
        r.append(f'<text x="{x + 10}" y="{yy}" font-size="11" fill="var(--bdr-ink-2)">{fits(t, 11, w - 20, "map")}</text>')
    r.append("</a>")
    return "".join(r)


def fig_map(base=BOOK_URL):
    W, H, y1, h1, y2, h2 = 660, 322, 30, 86, 190, 78
    s = [arrow_defs(1)]
    s.append(f'<text x="10" y="18" font-size="11" font-weight="600" letter-spacing=".06em" fill="{MUT}">ON PAPER · PARTS 1–7</text>')
    prev = None
    for x, w, parts in MAP_TOP:
        pad = 0 if len(parts) == 1 else 3  # a stacked pair sits in a frame, on which the arrows end
        if len(parts) == 1:
            slug, part, title, lines = parts[0]
            s.append(linkbox(x, y1, w, h1, f"{base}{slug}/", part, title, lines, MUT, f"Go to {part.title()}: {title}"))
        else:
            s.append(f'<rect x="{x - pad}" y="{y1 - pad}" width="{w + 2 * pad}" height="{h1 + 2 * pad}" rx="8" fill="none" stroke="{MUT}" stroke-dasharray="3 3" stroke-opacity="0.7"/>')
            hh = (h1 - 8) / 2
            for j, (slug, part, title, lines) in enumerate(parts):
                s.append(linkbox(x, y1 + j * (hh + 8), w, hh, f"{base}{slug}/", part, title, lines, MUT, f"Go to {part.title()}: {title}"))
        if prev is not None:
            s.append(f'<line x1="{prev + 1}" y1="{y1 + h1 / 2}" x2="{x - pad - 1}" y2="{y1 + h1 / 2}" stroke="currentColor" marker-end="url(#pa1)"/>')
        prev = x + w + pad
    s.append(f'<text x="{W - 6}" y="{y2 - 12}" text-anchor="end" font-size="11" font-weight="600" letter-spacing=".06em" fill="{MUT}">ON AN H200 · PARTS 8–11</text>')
    bw, gap, bx = 154, 12, 4
    X = lambda i: bx + i * (bw + gap)
    for i, (slug, part, title, lines) in enumerate(MAP_BOT):
        s.append(linkbox(X(i), y2, bw, h2, f"{base}{slug}/", part, title, lines, TD, f"Go to {part.title()}: {title}"))
        if i:
            s.append(f'<line x1="{X(i) - gap + 1}" y1="{y2 + h2 / 2}" x2="{X(i) - 1}" y2="{y2 + h2 / 2}" stroke="currentColor" marker-end="url(#pa1)"/>')
    # Part 8 measures the inputs of Part 2's proposition, and evaluates Part 5's exact accounting at them.
    ya, yb = y1 + h1 + 2, y2 - 3
    s.append(f'<line x1="128" y1="{ya}" x2="128" y2="{yb}" stroke="currentColor" marker-end="url(#pa1)"/>')
    for j, lab in enumerate(("its inputs,", "measured")):
        s.append(f'<text x="118" y="{(ya + yb) / 2 - 3 + j * 13}" font-size="10.5" fill="{MUT}" text-anchor="end">{lab}</text>')
    s.append(f'<line x1="433" y1="{ya}" x2="152" y2="{yb}" stroke="currentColor" marker-end="url(#pa1)"/>')
    s.append(f'<text x="330" y="{(ya + yb) / 2 + 8}" font-size="10.5" fill="{MUT}">exact accounting</text>')
    yc = y2 + h2 + 18
    s.append(f'<text x="10" y="{yc + 17}" font-size="11" font-weight="600" letter-spacing=".06em" fill="{MUT}">ALSO</text>')
    cx = 58
    for slug, title, what in MAP_SIDE:
        label = f"{title}: {what}"
        w = round(tw(label, 11) + 22)
        s.append(f'<a href="{base}{slug}/" class="pfig-link"><title>Go to the {title}</title><rect x="{cx}" y="{yc}" width="{w}" height="26" rx="13" fill="var(--bdr-surface)" stroke="{MUT}" stroke-width="1.2"/>'
                 f'<text x="{cx + 11}" y="{yc + 17}" font-size="11" fill="{INK}">{label}</text></a>')
        cx += w + 12
    if cx > W:
        FIT.append(f"map: the chips need {cx} px, have {W}")
    wide = svg(W, H, "Map of the book, each box a link to its chapter: Parts 1 to 7 bound the gain on paper; Parts 8 to 11 measure it on an H200 and compare", "".join(s))
    return f'<div class="v-wide">{wide}</div><div class="v-narrow">{fig_map_narrow(base)}</div>'


def fig_ladder(d):
    g = d["gains"]
    W, H, bw, gap, x0 = 664, 250, 150, 20, 4
    rungs = [("(a) book's bound", ["datasheet π, β,", "independent routing"], "roof_ind_ds", MUT),
             ("(b) ceilings", ["π, β measured", "on this H200"], "roof_ind", MUT),
             ("(c) routing", ["U counted in", "the same runs"], "roof", MUT),
             ("(d) measured", ["vLLM with fixes", "and new kernels"], "measured", TD)]
    steps = ["Part 8", "Part 8", "Parts 9–10"]
    s = [arrow_defs(2)]
    for i, (t, l, key, col) in enumerate(rungs):
        x = x0 + i * (bw + gap)
        s.append(tbox(x, 30, bw, 210, l, col, t, None, "ladder"))
        for j, B in enumerate((1, 64)):
            v = g[f"{B},1024"][key]
            y = 122 + j * 60
            s.append(f'<text x="{x + 10}" y="{y}" font-size="11" fill="{MUT}">B = {B}, C = 1k</text>')
            s.append(f'<text x="{x + 10}" y="{y + 28}" font-size="22" font-weight="600" fill="{TD if v >= 1 else AR}">Γ = {v:.2f}</text>')
        if i < 3:
            xa = x + bw + 2
            s.append(f'<line x1="{xa}" y1="135" x2="{xa + gap - 4}" y2="135" stroke="currentColor" marker-end="url(#pa2)"/>')
            s.append(f'<text x="{x + bw + gap / 2}" y="20" text-anchor="middle" font-size="11" fill="{MUT}">{steps[i]}</text>')
    return svg(W, H, "From the book's bound to the measured gain in four steps, each changing one input", "".join(s))


def fig_stack():
    W, rh, pitch, y0 = 660, 58, 78, 30
    ax, aw, bx, bw, cx, cw, bus = 6, 180, 222, 244, 500, 118, 638
    ACC = "var(--accent)"
    s = [arrow_defs(30), arrow_defs(31, ACC)]
    for x, t in ((ax, "WHAT RUNS"), (bx, "WHAT ROOFSCOPE RECORDS THERE"), (cx, "WHICH GIVES")):
        s.append(f'<text x="{x}" y="16" font-size="10.5" font-weight="600" letter-spacing=".06em" fill="{MUT}">{t}</text>')

    def box(x, y, w, title, lines, col):
        r = [f'<rect x="{x}" y="{y}" width="{w}" height="{rh}" rx="5" fill="var(--bdr-surface)" stroke="{col}" stroke-width="1.3"/>']
        yy = y + 19
        r.append(f'<text x="{x + 10}" y="{yy}" font-size="12" font-weight="600" fill="{INK}">{fits(title, 12, w - 20, "stack")}</text>')
        for t in lines:
            yy += 14
            r.append(f'<text x="{x + 10}" y="{yy}" font-size="10.5" fill="var(--bdr-ink-2)">{fits(t, 10.5, w - 20, "stack")}</text>')
        return "".join(r)

    hat = lambda ch: ch + "\u0302"  # combining circumflex
    stack = [("Workload", ["B prompts of C tokens"], "served by"),
             ("Engine: vLLM 0.28", ["one forward pass per", "step, as a CUDA graph"], "calls"),
             ("Model", ["30 layers: attention,", "dense MLP, MoE; the head"], "launches"),
             ("Kernels", ["CUTLASS, cuBLAS GEMMs,", "FlashAttention, Triton"], "run on"),
             ("Accelerator", ["one H200 SXM, 700 W cap"], None)]
    reads = [None,
             ("Wall clock of each pass", ["timed window, profiler off"], ("step time " + "<tspan font-style=\"italic\">t</tspan>", [])),
             ("Counter in each MoE layer", ["tokens per expert, in every pass", "of both windows"], ("experts " + "<tspan font-style=\"italic\">U</tspan>", ["per layer"])),
             ("Profiler trace", ["every kernel and its duration;", "its module, from an eager run"], ("time " + sub("<tspan font-style=\"italic\">t</tspan>", "c"), ["per component"])),
             ("Microbenchmarks (Part 8)", ["highest sustained FLOP rates", "and bandwidth measured"], ("ceilings", [f"{sub(hat('π'), 'c')} and {hat('β')}"]))]
    for i, (title, lines, verb) in enumerate(stack):
        y = y0 + i * pitch
        s.append(box(ax, y, aw, title, lines, MUT))
        if verb:
            xm = ax + 36
            s.append(f'<line x1="{xm}" y1="{y + rh + 1}" x2="{xm}" y2="{y + pitch - 2}" stroke="currentColor" marker-end="url(#pa30)"/>')
            s.append(f'<text x="{xm + 9}" y="{y + rh + 14}" font-size="10.5" fill="{MUT}">{verb}</text>')
        if reads[i]:
            rt, rl, (qt, ql) = reads[i]
            s.append(box(bx, y, bw, rt, rl, ACC))
            s.append(box(cx, y, cw, qt, ql, ACC))
            ym = y + rh / 2
            s.append(f'<line x1="{ax + aw + 1}" y1="{ym}" x2="{bx - 2}" y2="{ym}" stroke="{ACC}" marker-end="url(#pa31)"/>')
            s.append(f'<line x1="{bx + bw + 1}" y1="{ym}" x2="{cx - 2}" y2="{ym}" stroke="{ACC}" marker-end="url(#pa31)"/>')
            s.append(f'<line x1="{cx + cw}" y1="{ym}" x2="{bus}" y2="{ym}" stroke="{ACC}"/>')
    yr = y0 + 5 * pitch + 2
    s.append(f'<line x1="{bus}" y1="{y0 + pitch + rh / 2}" x2="{bus}" y2="{yr - 2}" stroke="{ACC}" marker-end="url(#pa31)"/>')
    rw = W - bx - 6
    s.append(f'<rect x="{bx}" y="{yr}" width="{rw}" height="68" rx="5" fill="var(--bdr-surface)" stroke="{INK}" stroke-width="1.5"/>')
    it = lambda t: f'<tspan font-style="italic">{t}</tspan>'
    tau, eta, F, Q, t = sub("τ", "c"), sub("η", "c"), sub(it("F"), "c"), sub(it("Q"), "c"), sub(it("t"), "c")
    s.append(f'<text x="{bx + 10}" y="{yr + 19}" font-size="12" font-weight="600" fill="{INK}">Definition 7, for each component c</text>')
    l1 = f'roofline time {tau} = max({F}/{sub(hat("π"), "c")}, {Q}/{hat("β")}); {F}, {Q} taken at {it("B")}, {it("C")}, {it("U")}'
    l2 = f'efficiency {eta} = {tau}/{t};  for the step, η = τ/{it("t")}'
    for j, line in enumerate((l1, l2)):
        fits(line.replace("\u0302", ""), 11, rw - 20, "stack")
        s.append(f'<text x="{bx + 10}" y="{yr + 38 + j * 17}" font-size="11" fill="{INK}">{line}</text>')
    return svg(W, yr + 76, "The serving stack from the workload down to the accelerator, what roofscope records at four of its levels, and the quantities of Definition 7 that each record gives", "".join(s))


def fig_attrib():
    W, H, x0, bw, gap = 660, 262, 106, 84, 8
    ks = ["rms_norm", "gemm_fp8", "attn_fwd", "gemm_fp8", "moe_gemm1", "moe_gemm2"]   # eager names
    kg = ["triton_red", "gemm_fp8", "attn_fwd", "gemm_fp8", "moe_gemm1", "moe_gemm2"]  # graph names
    mods = ["norm", "qkv proj.", "attention", "out proj.", "experts", "experts"]
    comps = ["outside", "dense", "attention", "dense", "experts", "experts"]
    X = lambda i: x0 + i * (bw + gap)
    s = [arrow_defs(11)]
    left = [(62, "eager run", 12, 600, INK), (77, "name, module", 10.5, 400, MUT),
            (112, "matched by", 10.5, 400, MUT), (126, "name, shape", 10.5, 400, MUT), (140, "and order", 10.5, 400, MUT),
            (184, "graph run", 12, 600, INK), (199, "name only", 10.5, 400, MUT), (238, "component", 10.5, 400, MUT)]
    for y, t, fs, fw, col in left:
        s.append(f'<text x="4" y="{y}" font-size="{fs}" font-weight="{fw}" fill="{col}">{fits(t, fs, x0 - 10, "attrib")}</text>')
    for i, (k, m, c) in enumerate(zip(ks, mods, comps)):
        x = X(i)
        s.append(f'<text x="{x + bw / 2}" y="26" text-anchor="middle" font-size="10.5" fill="{MUT}">{fits(m, 10.5, bw + gap - 4, "attrib")}</text>')
        s.append(f'<path d="M{x + 2},38 v-5 h{bw - 4} v5" fill="none" stroke="{MUT}"/>')
        s.append(f'<rect x="{x}" y="46" width="{bw}" height="34" rx="4" fill="var(--bdr-surface)" stroke="{MUT}"/>'
                 f'<text x="{x + bw / 2}" y="67" text-anchor="middle" font-size="11" fill="{INK}">{fits(k, 11, bw - 8, "attrib")}</text>')
        if kg[i] == k:
            s.append(f'<line x1="{x + bw / 2}" y1="82" x2="{x + bw / 2}" y2="164" stroke="{MUT}" stroke-dasharray="3 3" marker-end="url(#pa11)"/>')
        else:  # a compiler-fused kernel: no eager kernel of that name, so no module
            s.append(f'<text x="{x + bw / 2}" y="128" text-anchor="middle" font-size="10.5" fill="{MUT}">{fits("no match", 10.5, bw, "attrib")}</text>')
        s.append(f'<rect x="{x}" y="168" width="{bw}" height="34" rx="4" fill="var(--bdr-surface)" stroke="{MUT}"/>'
                 f'<text x="{x + bw / 2}" y="189" text-anchor="middle" font-size="11" fill="{INK}">{fits(kg[i], 11, bw - 8, "attrib")}</text>')
        s.append(f'<text x="{x + bw / 2}" y="238" text-anchor="middle" font-size="11" font-weight="600" fill="{INK}">{c}</text>')
    return svg(W, H, "Graph-run kernels carry their names but not their modules; each takes the module of the eager kernel with the same name and launch shape, in order", "".join(s))


def fig_flags():
    W, H = 660, 220
    s = [arrow_defs(4), arrow_defs(5, TD)]
    def cell(x, y, w, t, col, dash=""):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="4" fill="none" stroke="{col}" stroke-dasharray="{dash}"/>'
                f'<text x="{x + w / 2}" y="{y + 19}" text-anchor="middle" font-size="11" fill="{INK}">{fits(t, 11, w - 10, "flags")}</text>')
    s.append(f'<text x="6" y="16" font-size="12" font-weight="600" fill="{INK}">vLLM 0.28: flags copied with .to(int32) at every step</text>')
    s.append(cell(6, 54, 150, "bool buffer", MUT))
    s.append(cell(250, 30, 180, "int32 copy, capture", MUT))
    s.append(cell(250, 80, 180, "int32 copy, step n", CRIT, "5 3"))
    s.append(cell(510, 30, 144, "graph replay", TD))
    s.append(f'<line x1="156" y1="62" x2="248" y2="47" stroke="currentColor" marker-end="url(#pa4)"/>')
    s.append(f'<text x="202" y="42" text-anchor="middle" font-size="10.5" fill="{MUT}">at capture</text>')
    s.append(f'<line x1="156" y1="76" x2="248" y2="94" stroke="currentColor" marker-end="url(#pa4)"/>')
    s.append(f'<text x="202" y="106" text-anchor="middle" font-size="10.5" fill="{MUT}">every step</text>')
    s.append(f'<line x1="430" y1="45" x2="508" y2="45" stroke="{TD}" stroke-width="2" marker-end="url(#pa5)"/>')
    s.append(f'<text x="469" y="38" text-anchor="middle" font-size="10.5" fill="{TD}">reads</text>')
    s.append(f'<text x="440" y="99" font-size="10.5" fill="{CRIT}">never read</text>')
    s.append(f'<text x="6" y="152" font-size="12" font-weight="600" fill="{INK}">fix: one persistent int32 buffer</text>')
    s.append(cell(6, 170, 250, "int32 buffer, written in place", MUT))
    s.append(cell(510, 170, 144, "graph replay", TD))
    s.append(f'<line x1="256" y1="185" x2="508" y2="185" stroke="{TD}" stroke-width="2" marker-end="url(#pa5)"/>')
    s.append(f'<text x="382" y="178" text-anchor="middle" font-size="10.5" fill="{TD}">reads this step\'s flags</text>')
    return svg(W, H, "A replayed CUDA graph reads the address recorded at capture, so a flag tensor recreated at every step is never read", "".join(s))


MOE_ROWS = [("quantise x (2 kernels)", "", "quantise x, per token (1 kernel)", "23 MB"), ("sort rows by expert", "", "sort rows by expert", ""),
            ("GEMM 1 (gate, up)", "184 MB", "GEMM 1 + gelu(gate)·up", "100 MB"), ("act_and_mul", "92 MB", None, ""),
            ("quantise h (2 kernels)", "46 MB", "quantise h, per row (1 kernel)", "50 MB"),
            ("GEMM 2, a row per pair", "369 MB", "GEMM 2 + p·row, atomic add into y", "in L2"), ("moe_sum (read 369 MB)", "46 MB", None, "")]


def fig_moe():
    W, rh, top = 660, 40, 56
    H = top + rh * len(MOE_ROWS) + 10
    L0, LW, R0, RW = 6, 186, 290, 300
    s = [arrow_defs(6), f'<text x="{L0 + LW / 2}" y="20" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">vLLM Triton path</text>',
         f'<text x="{R0 + RW / 2}" y="20" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">new path (moe_ws)</text>',
         f'<text x="{L0 + LW / 2}" y="38" text-anchor="middle" font-size="10.5" fill="{MUT}">kernel · bytes written</text>',
         f'<text x="{R0 + RW / 2}" y="38" text-anchor="middle" font-size="10.5" fill="{MUT}">kernel · bytes written</text>']
    for i, (lb, lw, rb, rw) in enumerate(MOE_ROWS):
        y = top + i * rh
        s.append(f'<rect x="{L0}" y="{y}" width="{LW}" height="28" rx="4" fill="none" stroke="{MUT}"/><text x="{L0 + LW / 2}" y="{y + 18}" text-anchor="middle" font-size="11" fill="{INK}">{fits(lb, 11, LW - 8, "moe")}</text>')
        if lw:
            s.append(f'<text x="{L0 + LW + 6}" y="{y + 18}" font-size="11" fill="{TD}">{lw}</text>')
        if rb:
            s.append(f'<rect x="{R0}" y="{y}" width="{RW}" height="28" rx="4" fill="none" stroke="{MUT}"/><text x="{R0 + RW / 2}" y="{y + 18}" text-anchor="middle" font-size="11" fill="{INK}">{fits(rb, 11, RW - 8, "moe")}</text>')
            if rw:
                s.append(f'<text x="{R0 + RW + 6}" y="{y + 18}" font-size="11" fill="{TD}">{rw}</text>')
        else:
            s.append(f'<rect x="{R0}" y="{y}" width="{RW}" height="28" rx="4" fill="none" stroke="{MUT}" stroke-dasharray="4 4" opacity="0.5"/><text x="{R0 + RW / 2}" y="{y + 18}" text-anchor="middle" font-size="11" fill="{MUT}">removed (fused)</text>')
        if i:
            s.append(f'<line x1="{L0 + LW / 2}" y1="{y - 11}" x2="{L0 + LW / 2}" y2="{y - 2}" stroke="currentColor" marker-end="url(#pa6)"/><line x1="{R0 + RW / 2}" y1="{y - 11}" x2="{R0 + RW / 2}" y2="{y - 2}" stroke="currentColor" marker-end="url(#pa6)"/>')
    return svg(W, H, "vLLM's expert path runs nine kernels and writes intermediate tensors to memory; the new path runs five", "".join(s))


def fig_pair():
    W, H, bw, x1 = 660, 280, 260, 396
    s = [arrow_defs(7), arrow_defs(8, TD)]
    for r, x0 in ((0, 6), (1, x1)):
        cx = x0 + bw / 2
        s += [f'<rect x="{x0}" y="10" width="{bw}" height="256" rx="8" fill="none" stroke="{MUT}"/>',
              f'<text x="{cx}" y="32" text-anchor="middle" font-weight="600" font-size="11.5" fill="{INK}">{fits(f"CTA {r}, SM {chr(97 + r)}: columns {256 * r}–{256 * r + 255}", 11.5, bw - 12, "pair")}</text>',
              f'<text x="{cx}" y="52" text-anchor="middle" font-size="10.5" fill="var(--bdr-ink-2)">tiles {sub("Q", r)}, {sub("K", r)}, {sub("V", r)} in shared memory</text>']
        for j, (y, lab) in enumerate(((70, f'{sub("S", r)} = {sub("Q", r)} {sub("K", r)}ᵀ'), (142, f'S = {sub("S", 0)} + {sub("S", 1)}, P = softmax(S)'), (214, f'{sub("O", r)} += P {sub("V", r)}'))):
            s += [f'<rect x="{x0 + 12}" y="{y}" width="{bw - 24}" height="34" rx="4" fill="none" stroke="{MUT}"/>',
                  f'<text x="{cx}" y="{y + 22}" text-anchor="middle" font-size="11" fill="{INK}">{fits(lab, 11, bw - 32, "pair")}</text>']
            if j:
                s.append(f'<line x1="{cx}" y1="{y - 36}" x2="{cx}" y2="{y - 2}" stroke="currentColor" marker-end="url(#pa7)"/>')
    s += [f'<line x1="{6 + bw - 12}" y1="88" x2="{x1 + 12}" y2="156" stroke="{TD}" stroke-width="2" marker-end="url(#pa8)"/>',
          f'<line x1="{x1 + 12}" y1="88" x2="{6 + bw - 12}" y2="156" stroke="{TD}" stroke-width="2" marker-end="url(#pa8)"/>']
    for k, t in enumerate(("partial scores", "Tq × Tk, FP16,", "to partner")):
        s.append(f'<text x="{(6 + bw + x1) / 2}" y="{200 + 14 * k}" text-anchor="middle" font-size="10.5" fill="{TD}">{fits(t, 10.5, x1 - 6 - bw - 6, "pair")}</text>')
    return svg(W, H, "Two thread blocks on two SMs each hold half of the head dimension and exchange their partial scores", "".join(s))


def fig_smem():
    W, x0, sc = 660, 150, 1.95
    rows = [("FA4, hd 512", [("Q", 64, AR), ("K", 64, GREEN), ("V", 64, AMBER), ("P", 8, MUT)]),
            ("pair, per CTA", [("Q half", 64, AR), ("K st.1", 32, GREEN), ("K st.2", 32, GREEN), ("V st.1", 32, AMBER), ("V st.2", 32, AMBER), ("S ×2", 32, TD)])]
    s = []
    for i, (lab, segs) in enumerate(rows):
        y = 16 + i * 46
        s.append(f'<text x="{x0 - 8}" y="{y + 19}" text-anchor="end" font-size="11" fill="{INK}">{fits(lab, 11, x0 - 12, "smem")}</text>')
        x = x0
        for name, kb, col in segs:
            w = kb * sc
            s.append(f'<rect x="{x}" y="{y}" width="{w - 2}" height="28" rx="3" fill="{col}" fill-opacity="0.8"/>')
            if w > 36:
                s.append(f'<text x="{x + w / 2 - 1}" y="{y + 18}" text-anchor="middle" font-size="10.5" style="fill:#fff">{fits(name, 10.5, w - 6, "smem")}</text>')
            x += w
        s.append(f'<text x="{x0 + 227 * sc + 8}" y="{y + 19}" font-size="11" fill="{INK}">{sum(k for _, k, _ in segs)} KB</text>')
    lx = x0 + 227 * sc
    s += [f'<line x1="{lx}" y1="8" x2="{lx}" y2="108" stroke="{CRIT}" stroke-dasharray="5 4"/>',
          f'<text x="{lx}" y="126" text-anchor="middle" font-size="11" fill="{CRIT}">227 KB per SM</text>']
    return svg(W, 136, "Shared memory per thread block: FA4 at head dimension 512 fits one pipeline stage; the pair kernel fits two stages and its exchange buffers", "".join(s))


# ---------------------------------------------------------------- tables
def t_ceilings(d):
    ce = d["ceil"]
    rows = [["FP8 GEMM (TFLOP/s)", "1,979", f"{ce['pi_fp8'] / 1e12:,.0f}", f"{ce['pi_fp8'] / 1979e12:.2f}", "the model's own projections, CUDA-graph bursts of 2 ms"],
            ["BF16 GEMM (TFLOP/s)", "989", f"{ce['pi_bf16'] / 1e12:,.0f}", f"{ce['pi_bf16'] / 989e12:.2f}", "8,192³, short burst"],
            ["attention (TFLOP/s)", "989", f"{ce['pi_attn'] / 1e12:,.0f}", f"{ce['pi_attn'] / 989e12:.2f}", "FlashAttention 3, head dim. 128, 2 × 16k tokens, sustained"],
            ["HBM bandwidth (TB/s)", "4.80", f"{ce['beta'] / 1e12:.2f}", f"{ce['beta'] / 4.8e12:.2f}", "read of a 4 GB buffer, sustained"]]
    return table("Table 11. Datasheet and empirical ceilings of one H200 SXM. The empirical ceiling is the highest rate any measurement reached on random data; the ridge point for FP8 linear layers moves from 412 to "
                 f"{ce['pi_fp8'] / ce['beta']:.0f} FLOP/byte.", ["rate", "datasheet", "empirical", "ratio", "measured on"], rows, fold="Figure 23")


def t_neff(d):
    R, Bs, Cs = d["routing"], d["Bs"], d["Cs"]
    rows = []
    for B in Bs:
        r = [f"B = {B}"]
        for m in ("td", "ar"):
            for C in (1024, 65536):
                v = R[m].get(f"{B},{C}")
                neff = v and (v.get("neff") or (v.get("meas") and math.log(1 - v["meas"] / 128) / math.log(1 - 8 / 128)))  # U_{E,k}(N) = meas, E = 128, k = 8
                r.append(f"{neff:.1f}" if neff else "–")
        rows.append(r)
    return table("Table 12. Effective number of independent tokens per pass, \\(N^{\\mathrm{eff}}\\), the number of uniformly routed independent tokens that would touch the measured \\(U\\) (Definition 6). A TD pass holds \\(256B\\) positions, an AR step \\(B\\).",
                 ["", "TD, 1k", "TD, 64k", "AR, 1k", "AR, 64k"], rows, fold="Figure 24")


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
                 ["", "\\(\\Pi_{\\mathrm{lin}}\\), independent", "\\(\\Pi_{\\mathrm{lin}}\\), measured", "\\(\\Gamma\\), Prop. 1, independent", "\\(\\Gamma\\), Prop. 1, measured routing", "\\(\\Gamma\\), measured"], rows, fold="Figure 25")


COMP = [("experts", "experts"), ("dense", "dense GEMMs and router"), ("attnW", "window attention"), ("attnG", "global attention"), ("head", "vocabulary head, sampler and self-conditioning")]


def t_eta(d):
    rows = []
    # the points with batch 8 and above that both TD configurations measured
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
    return table(f"Table 17. Efficiency against the empirical roofline, \\(\\eta_c = \\tau_c/t_c\\), over the {sum(1 for k in d['eta3']['td'] if big(k))} measured points with batch 8 and above. Stock kernels: the three engine fixes of Part 9 with vLLM's kernels; new kernels: the same with the two kernels of this Part, in the same campaign and on the same prompts. AR runs vLLM's kernels throughout.",
                 ["component", "TD, stock kernels", "TD, new kernels", "AR"], rows, fold="Figure 38")


def t_stock(d):
    rows = []
    for c, name in COMP:
        rows.append([name] + [rng([v.get(c) for k, v in d["eta2"]["td"].items() if int(k.split(",")[0]) in Bset]) for Bset in ((1, 2, 4), (8, 16, 32, 64))])
    return table("Table 14. TD with the three engine fixes and vLLM's own kernels: efficiency of each component against its empirical roofline, at batch 1–4 and at batch 8–64, over all contexts: the 27 points of this run, which also fitted \\((32, 64\\text{k})\\) and \\((64, 16\\text{k})\\); Table 17 keeps the points measured with both sets of kernels.",
                 ["component", "B = 1–4", "B = 8–64"], rows, fold="Figure 32")


def t_vs2(d):
    v = d["vs2"]
    def r(m, Bp, C):
        xs = [x["ratio"] for k, x in v[m].items() if Bp(int(k.split(",")[0])) and int(k.split(",")[1]) == C]
        return rng(xs)
    rows = [[kc(C), r("td", lambda b: b == 1, C), r("td", lambda b: b >= 2, C)] for C in d["Cs"]]
    return table("Table 18. TD's throughput with the new kernels over its throughput with vLLM's kernels, at the same points and in the same campaign: the median of three repeats, each on its own prompts, against one run on the first repeat's prompts.",
                 ["context", "B = 1", "B = 2–64"], rows, fold="Figure 39")


def dumbbell(rows, lo, hi, ticks, fmtx, refs, xlabel, label_w=150):
    """Before -> after per row: rows = [(label, before, after, note)]; refs = [(x, text)] vertical guides."""
    W, rh, mt, mb, mr = 664, 34, 30, 40, 92
    H = mt + rh * len(rows) + mb
    X = lambda v: label_w + (v - lo) / (hi - lo) * (W - label_w - mr)
    s = [arrow_defs(21, TD)]
    for t in ticks:
        s.append(f'<line x1="{X(t)}" x2="{X(t)}" y1="{mt - 8}" y2="{H - mb}" stroke="{RULE}"/>'
                 f'<text x="{X(t)}" y="{H - mb + 15}" text-anchor="middle" font-size="11" fill="{MUT}">{fmtx(t)}</text>')
    s.append(f'<text x="{(label_w + W - mr) / 2}" y="{H - 8}" text-anchor="middle" font-size="11.5" fill="{INK}">{xlabel}</text>')
    for x, text in refs:
        s.append(f'<line x1="{X(x)}" x2="{X(x)}" y1="{mt - 14}" y2="{H - mb}" stroke="{INK}" stroke-dasharray="4 3"/>'
                 f'<text x="{X(x) + 4}" y="{mt - 16}" font-size="11" fill="{INK}">{text}</text>')
    for i, (lab, a, b, note) in enumerate(rows):
        y = mt + i * rh + rh / 2
        s.append(f'<text x="{label_w - 10}" y="{y + 4}" text-anchor="end" font-size="11.5" fill="{INK}">{lab}</text>')
        s.append(f'<line x1="{X(a)}" x2="{X(b)}" y1="{y}" y2="{y}" stroke="{TD}" stroke-width="2.4" stroke-opacity="0.55"/>')
        s.append(f'<circle cx="{X(a)}" cy="{y}" r="5" fill="var(--bdr-surface)" stroke="{MUT}" stroke-width="2"><title>{lab}: before {fmtx(a)}</title></circle>')
        s.append(f'<circle cx="{X(b)}" cy="{y}" r="5.5" fill="{TD}"><title>{lab}: after {fmtx(b)}</title></circle>')
        s.append(f'<text x="{X(b) + 10}" y="{y + 4}" font-size="11" fill="{MUT}">{note}</text>')
    return svg(W, H, xlabel, "".join(s))


def fig_moe_eff():
    rows = [("B = 8, C = 1k", 0.27, 0.33, "time −19%"), ("B = 32, C = 1k", 0.34, 0.47, "time −26%"),
            ("B = 32, C = 16k", 0.35, 0.48, "time −27%"), ("B = 64, C = 1k", 0.37, 0.52, "time −28%")]
    return dumbbell(rows, 0, 1, [0, 0.2, 0.4, 0.6, 0.8, 1], lambda v: f"{v:.1f}", [(0.81, "cuBLAS reference 0.81")],
                    "expert path: fraction of its empirical roofline, vLLM's kernel (open) → moe_ws (filled)")


def fig_attn_eff():
    rows = [("B = 8, C = 16k", 335, 455, "step −8.2%"), ("B = 32, C = 16k", 320, 436, "step −6.4%"), ("B = 8, C = 64k", 330, 453, "step −18%")]
    return dumbbell(rows, 0, 700, [0, 100, 200, 300, 400, 500, 600, 700], lambda v: f"{v:.0f}", [(638, "ceiling 638")],
                    "global attention, TFLOP/s: FlashAttention 4 (open) → pair kernel (filled)")


def t_moe():
    return table("Table 15. The expert path per denoising pass in the engine, vLLM's kernel against the new one, and its efficiency against the empirical roofline at the measured routing. These are dedicated runs of the expert kernel (three per arm), separate from the grid of Parts 8–11, so their efficiencies differ slightly from Table 17's.",
                 ["TD point (B, C)", "expert path per pass", "fraction of empirical roofline", "TD step"],
                 [["(8, 1k)", "15.03 → 12.23 ms (−19%)", "0.27 → 0.33", "43.56 → 40.07 ms (−8.0%)"],
                  ["(32, 1k)", "46.60 → 34.26 ms (−26%)", "0.34 → 0.47", "158.75 → 147.81 ms (−6.9%)"],
                  ["(32, 16k)", "45.59 → 33.37 ms (−27%)", "0.35 → 0.48", "227.44 → 217.54 ms (−4.4%)"],
                  ["(64, 1k)", "85.78 → 61.61 ms (−28%)", "0.37 → 0.52", "326.30 → 299.23 ms (−8.3%)"]], fold="Figure 34")


def t_attn():
    return table("Table 16. Global attention in the engine, TFLOP/s and fraction of the 638 TFLOP/s attention ceiling, one run each; the last column is the TD step with FA4 at 64-token pages against the pair kernel at 80-token pages.",
                 ["(B, C)", "FA4, page 64", "pair kernel, page 80", "TD step"],
                 [["(8, 16k)", "335 (0.52)", "455 (0.71)", "58.05 → 53.27 ms (−8.2%)"],
                  ["(32, 16k)", "320 (0.50)", "436 (0.68)", "222.37 → 208.24 ms (−6.4%)"],
                  ["(8, 64k)", "330 (0.52)", "453 (0.71)", "107.92 → 88.47 ms (−18.0%)"]], fold="Figure 37")


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
                 ["", "context 1k", "context 64k"], rows, fold="Figure 40")


# ---------------------------------------------------------------- figures that show what the tables list
def fig_prop1(d, C=1024):
    """Proposition 1 at independent and at measured routing against the measured gain, by batch (Table 13)."""
    p, g, Bs = d["prop1"]["fp8_empirical"]["pts"], d["gains"], [B for B in d["Bs"] if f"{B},{C}" in d["prop1"]["fp8_empirical"]["pts"]]
    W, H, ml, mr, mt, mb = 600, 300, 46, 182, 16, 44
    X = lambda i: ml + i * (W - ml - mr) / (len(Bs) - 1)
    ymax = 8
    Y = lambda v: H - mb - v / ymax * (H - mb - mt)
    s = []
    for v in range(0, ymax + 1, 2):
        s.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W - mr}" y2="{Y(v):.1f}" stroke="{RULE}"/><text x="{ml - 7}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUT}">{v}</text>')
    s.append(f'<line x1="{ml}" y1="{Y(1):.1f}" x2="{W - mr}" y2="{Y(1):.1f}" stroke="{MUT}" stroke-dasharray="3 4"/><text x="{ml + 4}" y="{Y(1) + 14:.1f}" font-size="10.5" fill="{MUT}">parity, Γ = 1</text>')
    for i, B in enumerate(Bs):
        s.append(f'<text x="{X(i):.1f}" y="{H - mb + 16}" text-anchor="middle" font-size="11" fill="{MUT}">{B}</text>')
    s.append(f'<text x="{(ml + W - mr) / 2}" y="{H - 8}" text-anchor="middle" font-size="11.5" fill="{INK}">batch B, at context {kc(C)}</text>')
    s.append(f'<text x="13" y="{(mt + H - mb) / 2}" text-anchor="middle" font-size="11.5" fill="{INK}" transform="rotate(-90 13 {(mt + H - mb) / 2})">gain Γ</text>')
    series = [("Prop. 1, routing of A2", [p[f"{B},{C}"]["gain_ind"] for B in Bs], MUT, ' stroke-dasharray="6 4"', 1.8),
              ("Prop. 1, measured routing", [p[f"{B},{C}"]["gain_meas"] for B in Bs], INK, "", 2),
              ("measured gain", [g[f"{B},{C}"]["measured"] for B in Bs], TD, "", 2.6)]
    ends = spread([Y(vals[-1]) for _, vals, _, _, _ in series])
    for (name, vals, col, dash, sw), ye in zip(series, ends):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round"{dash}/>')
        for i, v in enumerate(vals):
            s.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.2" fill="{col}"><title>{name}, B = {Bs[i]}: {v:.2f}</title></circle>')
        lab_col, bold = ("var(--bdr-ink-2)" if col == MUT else col), (' font-weight="600"' if col == TD else "")
        s.append(f'<text x="{W - mr + 10}" y="{ye + 4:.1f}" font-size="11" fill="{lab_col}"{bold}>{fits(name, 11, mr - 12, "prop1")}</text>')
    return svg(W, H, "Gain against batch at context 1k: Proposition 1 under independent routing, Proposition 1 at the measured routing, and the measured gain", "".join(s))


def cell_grid(rows, cols, val, fmtv, scale, col=TD, label_w=176, cw=52, ch=28, head=""):
    """A grid of values with one row per (key, name) of `rows` and one column per (key, name) of `cols`."""
    ml, mt = label_w, 30
    W, H = ml + cw * len(cols) + 6, mt + ch * len(rows) + 6
    s = [f'<text x="{ml - 8}" y="{mt - 10}" text-anchor="end" font-size="11" fill="{MUT}">{head}</text>']
    for j, (_, cn) in enumerate(cols):
        s.append(f'<text x="{ml + j * cw + cw / 2}" y="{mt - 10}" text-anchor="middle" font-size="11.5" fill="{MUT}">{cn}</text>')
    for i, (rk, rn) in enumerate(rows):
        y = mt + i * ch
        s.append(f'<text x="{ml - 8}" y="{y + ch / 2 + 4}" text-anchor="end" font-size="11.5" fill="{INK}">{fits(rn, 11.5, ml - 12, "grid")}</text>')
        for j, (ck, cn) in enumerate(cols):
            v, x = val(rk, ck), ml + j * cw
            if v is None:
                s.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" fill="{RULE}" fill-opacity="0.5"/><text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" font-size="11.5" fill="{MUT}">–</text>')
                continue
            a = max(0.05, min(1, v / scale))
            txt = "var(--bdr-bg)" if a > 0.55 else INK
            s.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" fill="{col}" fill-opacity="{a:.2f}"><title>{rn}, {cn}: {fmtv(v)}</title></rect>'
                     f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" font-size="11.5" style="fill:{txt};font-weight:{600 if a > 0.55 else 400}">{fmtv(v)}</text>')
    return svg(W, H, head or "grid of values", "".join(s))


RECOVER_ROWS = [("experts", "experts"), ("dense", "dense layers"), ("attnG", "global attention"), ("attnW", "window attention"),
                ("head", "head, sampler, self-cond."), ("outside", "outside the model"), ("idle", "idle")]


def fig_recover(d, C):
    """Recoverable share of TD's step by component and batch, at one context (Table 19 lists the two largest)."""
    rec = d["recoverable"]["td"]
    cols = [(B, f"B = {B}" if B == d["Bs"][0] else str(B)) for B in d["Bs"]]
    return cell_grid(RECOVER_ROWS, cols, lambda c, B: (rec.get(f"{B},{C}") or {}).get(c), lambda v: f"{100 * v:.0f}%", 0.2, label_w=190,
                     head=f"context {kc(C)}")


def fig_vs2(d):
    """TD's throughput with the new kernels over its throughput with vLLM's kernels (Table 18 gives the ranges)."""
    v = d["vs2"]["td"]
    g = {k: {"v": x["ratio"]} for k, x in v.items()}
    return heat({"gains": g, "Bs": d["Bs"], "Cs": d["Cs"]}, lambda r: r["v"], "new kernels / vLLM's kernels", 1,
                color=lambda x: (TD, min(1, max(0, x - 1) / 0.25)))


# ---------------------------------------------------------------- assembly
def blocks():
    d = json.loads(DATA.read_text())
    sc_dev = max(abs(math.log(r["measured"] / r["roof"])) for r in d["gains"].values())
    dev = heat(d, lambda r: r["measured"] / r["roof"], "(e) measured / bound (c)", sc_dev,
               color=lambda v: (TD if v >= 1 else AR, min(1, abs(math.log(v)) / sc_dev)))
    sc4 = gain_scale(d, ["roof_ind_ds", "roof_ind", "roof", "measured"])
    m4 = lambda specs: '<div class="pfig-grid">' + "".join(f"<div>{heat(d, k, t, sc4)}</div>" for k, t in specs) + "</div>"
    out = {
        "decoders": fig("fig-decoders", "Figure 1", "In the same number of passes, AR commits one token per pass and TD a whole canvas.",
                        "", fig_decoders(),
                        "One sequence, DiffusionGemma's setting: a canvas of \\(b = 256\\) positions, \\(S = 14\\) denoising passes and one commit pass, so \\(\\omega = 15\\) passes per canvas and \\(\\mathrm{TPF} = b/\\omega = 17.1\\) tokens per forward pass. Each TD pass processes 256 positions instead of one, so it does more FLOPs and its positions touch more experts; Parts 1–5 weigh these costs against the shared read of the weights."),
        "map": fig("fig-map", "Figure 2 · interactive", "The book in two halves: bounds on paper, then measurements on one H200.", "", fig_map(),
                   "Parts 1–7 derive the gain \\(\\Gamma\\) as a ratio of roofline bounds and map it over batch and context. Parts 8–11 take the inputs of those bounds from measurements, time every kernel of both decoders in vLLM, rewrite the two kernels that lag most, and compare the measured gain with the bound. Select a box to open its chapter."),
        "ladder": fig("fig-ladder", "Figure 3 · interactive", "From the book's bound to the measured gain in four steps, each changing one input.", "Choose a context; hover over or tap a batch to read the four values.",
                      '<div data-mfig="ladder" data-series="a,b,c,d"></div>',
                      "DiffusionGemma (TD) against Gemma 4 26B-A4B (AR) on one H200, FP8 weights, at the 25 measured points. (a) is the book's exact accounting, vocabulary included, with the kernel-by-kernel timing rule of Part 5, at FP8, with datasheet rates and independent routing, evaluated at the context of the measured windows. (b) uses the ceilings measured on this H200, (c) also the expert counts measured in the same runs, and (d) is the measured gain. Part 8 explains (a) to (c), and Part 11 compares (c) with (d)."),
        "roofline": fig("fig-eff-roofline", "Figure 23", "The measured roofline sits below the datasheet's, more in compute than in bandwidth.", "", fig_eff_roofline(d),
                        "Attainable throughput \\(\\min(\\pi, I\\beta)\\) against arithmetic intensity, log–log, for FP8 linear layers. Dashed: datasheet (1,979 TFLOP/s, 4.8 TB/s). Solid orange: empirical ceilings of Table 11. Blue: the attention ceiling, which applies to attention kernels whatever their intensity. The ridge point, where a kernel turns compute-bound, moves from 412 to 340 FLOP/byte."),
        "routing": fig("fig-eff-routing", "Figure 24 · interactive", "Both decoders touch fewer experts than independent routing predicts, for different reasons.", "Choose the decoder and the context; hover over or tap a batch to read the counts.",
                       '<div data-mfig="routing"></div>',
                       "Distinct experts per MoE layer and forward pass \\(U\\), averaged over the 30 layers and the timed passes, against batch size: the median of three repeats with different prompts, and the band spanning them. Dotted: independent tokens with uniform routing, \\(U_{E,k}(N)\\) of Definition 1. Dashed: independent tokens with each expert's measured popularity (Definition 6). AR follows the dashed curve; TD lies below it."),
        "maps_abc": fig("fig-eff-maps", "Figure 26 · interactive", "The gain bound moves with each measured input, and routing moves it most.", "Choose a context; hover over or tap a batch to read the three bounds.",
                        '<div data-mfig="ladder" data-series="a,b,c"></div>',
                        "Roofline bound on \\(\\Gamma\\) against batch, with FP8 weights, at the measured points; each curve changes one input of the previous one. (a) the book's exact accounting with datasheet rates and independent routing (A2); (b) the empirical ceilings of Table 11; (c) the expert counts measured in the same runs. Below the parity line AR is faster. Missing points were not measured: the grid keeps \\(BC \\le 2^{21}\\) tokens, and TD's batches at \\((32, 64\\text{k})\\) and \\((64, 16\\text{k})\\) did not fit in memory."),
        "eager": fig("fig-eager", "Figure 27", "In eager mode the GPU waits for each launch; a CUDA graph runs the kernels of a pass back to back.", "", fig_eager(),
                     "Four kernels of one forward pass, as in Part 8's small GEMMs: launching a kernel costs the CPU a few µs, and with 64 tokens or fewer a projection runs for 5–12 µs, so in eager mode the GPU waits a large share of the time. A CUDA graph replays the recorded sequence of kernels with one launch."),
        "stack": fig("fig-stack", "Figure 28", "Each term of a component's efficiency is read at a different level of the serving stack.", "", fig_stack(),
                     "Left: what runs when the engine serves a point \\((B, C)\\), from the prompts down to the accelerator. Middle: the four records roofscope takes, each at the level where the quantity exists. The engine's wall clock gives the step time \\(t\\); a counter in the model gives the experts touched, \\(U\\); the profiler's trace of kernels gives the time \\(t_c\\) of each component; and the microbenchmarks of Part 8 give the ceilings \\(\\hat\\pi_c\\), one per kind of kernel, and \\(\\hat\\beta\\) of the accelerator. Definition 7 combines them: the FLOPs \\(F_c\\) and bytes \\(Q_c\\) of a component, counted by the cost model of Parts 1–7 at the point \\((B, C)\\) and at the measured \\(U\\), and the ceilings give its roofline time \\(\\tau_c\\), which is compared with \\(t_c\\). Each \\(\\tau_c\\) uses the expert count of the profiled window, where \\(t_c\\) is read, and the step's \\(\\tau\\) that of the timed window, where \\(t\\) is read. The rest of this section describes the two windows of a run (Figure 29) and the labelling of kernels (Figure 30)."),
        "run": fig("fig-run", "Figure 29", "One measured point: a profiled window for the kernels, a timed window for the step time.", "", fig_run(),
                   "TD at one batch size \\(B\\) and context \\(C\\): the windows are the third to sixth canvases of the answer; the first is produced during the prefill. The profiler slows the passes down, so the step time comes from the second window, where it is off. For AR the windows are 48 steps each, after 976 warm-up steps, which put AR's context at the middle of the windows at TD's."),
        "attrib": fig("fig-attrib", "Figure 30", "Each kernel of the graph run takes its module from the eager run.", "", fig_attrib(),
                      "A stretch of one layer, with simplified kernel names. In the eager run each kernel is recorded inside the profiler range of the module that launched it; in the graph run only inside the range of the whole pass. The same kernel can serve several modules, as the FP8 GEMM does here, so the match counts occurrences. Each module then belongs to one component. The compiler fuses the norm with its neighbours into a kernel of another name, which has no eager match and counts as outside the model."),
        "flags": fig("fig-flags", "Figure 31", "A replayed CUDA graph reads the memory it saw at capture, so a tensor recreated at every step is never read.", "", fig_flags(),
                     "DiffusionGemma passes a causal flag per sequence to the attention kernel: 0 for a bidirectional denoising pass, 1 for the causal commit pass. In vLLM 0.28 the flags lived in a bool buffer converted to int32 at every step; under CUDA graphs the attention kernels kept reading the copy made at capture."),
        "moe": fig("fig-moe", "Figure 33", "The new expert path runs five kernels where vLLM's runs nine; its two GEMMs are as fast as vLLM's.", "", fig_moe(),
                   "One MoE layer at TD batch 32 (\\(Nk = 65{,}536\\) rows, each a (token, expert) pair). Each box is one step of the path, one kernel unless marked, in order; the orange number is what it writes to memory, read back by the next step."),
        "pair": fig("fig-pair", "Figure 35", "Each thread block of the pair holds half of the head dimension; the only traffic between the two SMs is the partial score tile.", "", fig_pair(),
                    "Both blocks process the same 128 query rows. The orange arrows are the exchange: each block writes its \\(T_q \\times T_k\\) partial scores into the partner's shared memory and waits for the partner's before the softmax."),
        "smem": fig("fig-smem", "Figure 36", "Splitting the head dimension makes room for a second pipeline stage.", "", fig_smem(),
                    "Shared memory per thread block, in KB, against the 227 KB of one SM. FA4 at head dimension 512 holds full-width tiles and fits one stage; a block of the pair holds half-width tiles and fits two stages of 64 keys plus two exchange buffers. With 80-token KV pages the fastest configuration uses the room differently, for one stage of 80-key tiles loaded by TMA (Table 16)."),
        "maps_cd": fig("fig-conc-maps", "Figure 41 · interactive", "The measured gain against the bound at measured routing and empirical ceilings.", "Choose a context; hover over or tap a batch to read both.",
                       '<div data-mfig="ladder" data-series="c,d"></div>',
                       "The bound (c) of Figure 26 and the measured gain (d), TD's tokens per second over AR's at equal \\(B\\) and \\(C\\), medians of three repeats. The measured gain lies above the bound up to batch 4 and below it from batch 16 on; Proposition 9 says why."),
        "dev": fig("fig-conc-dev", "Figure 42 · interactive", "The bound is conservative while TD runs closer to its roofline than AR, up to batch 8, and optimistic from batch 16 on.", "Choose a context; hover over or tap a batch to read the two efficiencies and their ratio.",
                   '<div data-mfig="eff"></div>',
                   "Step efficiency \\(\\eta_X = \\tau_X/t_X\\) of each decoder, its empirical roofline time over its measured step time, medians of three repeats. By Proposition 9 the measured gain over the bound is \\(\\eta_{TD}/\\eta_{AR}\\): the band between the curves is orange where the measured gain exceeds the bound and blue where it falls short. AR's efficiency rises with the batch and TD's barely moves."),
        "eta_maps": fig("fig-eta-maps", "Figure 38 · interactive", "After the kernel work, every component of TD's step runs at half its roofline or more from batch 8 on, the experts aside.", "Choose the decoder, highlight a component and choose a context; hover over or tap a batch to read every component.",
                        '<div data-mfig="comp" data-src="eta3"></div>',
                        "Efficiency \\(\\eta_c = \\tau_c/t_c\\) of each component, its empirical roofline time over its measured kernel time, medians of three repeats; dashed, the whole step \\(\\eta_X = \\tau_X/t_X\\), which also counts time outside the model and idle time. The choice TD with vLLM kernels repeats Figure 32 for comparison."),
        "prop1": fig("fig-prop1", "Figure 25 · interactive", "Proposition 1 underestimates the measured gain at batch 1–2 and overestimates it from batch 4 on, at either routing.", "Choose a context; hover over or tap a batch to read the penalties and gains.",
                     '<div data-mfig="prop1"></div>',
                     "Gain against batch with FP8 weights and the empirical ceilings: the values of Table 13 at context 1k. Dashed: Proposition 1 under A2, independent uniform routing. Black: Proposition 1 with the measured expert counts of both decoders. Orange: the measured gain, median of three repeats. At small batch the measured routing raises the bound, since a canvas touches fewer experts than A2 predicts, and at batch 1–2 the measurement is higher still (Part 11). From batch 8 on it lowers the bound, since AR's count falls further below A2's prediction than TD's, and Proposition 1 still overestimates the measured gain, because it leaves out the vocabulary head."),
        "eta_stock": fig("fig-eta-stock", "Figure 32 · interactive", "With vLLM's own kernels, the experts and global attention are the furthest from their rooflines from batch 8 on.", "Highlight a component and choose a context; hover over or tap a batch to read every component.",
                         '<div data-mfig="comp" data-src="eta2"></div>',
                         "Efficiency \\(\\eta_c = \\tau_c/t_c\\) of each component of TD's step with the three engine fixes and vLLM's own kernels, one run per point; dashed, the whole step, \\(\\eta = \\tau/t\\), which also counts time outside the model and idle time. Table 14 gives the ranges, and Figure 38 the same curves after the kernel work of Part 10."),
        "vs2": fig("fig-vs2", "Figure 39 · interactive", "The two kernels raise TD's throughput at every measured point, by 4% to 22%, most at long context.", "Hover over or tap a batch to read the four contexts.",
                   '<div data-mfig="vs2"></div>',
                   "TD's tokens per second with the kernels of this chapter over its tokens per second with vLLM's kernels, at the same batch and context: the median of three repeats, each on its own prompts, over one run on the first repeat's prompts. Table 18 gives the ranges."),
        "recover": fig("fig-recover", "Figure 40 · interactive", "The recoverable time that remains sits mostly in the experts up to batch 8, and outside the model from batch 16 on.", "Choose a context and a batch.",
                       '<div data-mfig="recover"></div>',
                       "Each component's share of TD's step after the kernel work, split into the time its target kernel would keep and the recoverable rest (Definition 7); time outside the model and idle time count in full. At batch 1 and context 64k global attention leads, at 17% of the step. Table 19 lists the two largest entries at contexts 1k and 64k."),
        "moe_eff": fig("fig-moe-eff", "Figure 34", "The new expert path raises the experts to 0.33–0.52 of their roofline at every point, short of the 0.81 that cuBLAS reaches.", "",
                       fig_moe_eff(),
                       "Efficiency of the expert path against its empirical roofline at the measured routing, in dedicated runs of the kernel, three per arm and separate from the grid of Parts 8–11; the label gives the change in time per pass. The dashed line is the stacked cuBLAS reference at batch 32, the target of Definition 7. Table 15 gives the times."),
        "attn_eff": fig("fig-attn-eff", "Figure 37", "The pair kernel raises global attention from about half to about 0.7 of the attention ceiling.", "",
                        fig_attn_eff(),
                        "Global attention in the engine at three points, one run each: FlashAttention 4 with 64-token pages against the pair kernel with 80-token pages; the label gives the change in TD's step time. The ceiling is FlashAttention 3's sustained rate (Table 11). Table 16 gives the numbers."),
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
