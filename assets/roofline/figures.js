// Interactive figures for the Block Diffusion Rooflines page. Every number comes from the note's own
// cost model (window.BDModel) and helpers (window.BDR, exposed by note.js).
//
// The figures of Part 2 on weights, the KV cache and sparse attention (figWeights ... figSparse)
// use the model as the propositions do: no vocabulary, independent uniform routing (A3) and
// independent uniform selections in sparse layers.
// Their toy grids (experts of one layer, cached tokens of one sequence) show one random draw; the
// readout next to each gives the expectation that the propositions use.
//
// Accountings:
//   - intensities are always those of Proposition 1 (Definition 3, no vocabulary): prop1();
//   - gains labelled "exact" use the full accounting at a fixed context: steady();
//   - regimes quoted next to an exact gain are stated as shares of the peak compute (util()),
//     never as Proposition-1 intensities, so the two accountings are not mixed in one statement.
(function () {
  'use strict';
  const M = window.BDModel, X = window.BDR;
  if (!M || !X) return;
  const { HWBYID, REF, omega, fmt, css, comma, kTok } = X;

  // ---------------------------------------------------------------- helpers
  const hwOf = (id, kernels = 'whole') => { const h = HWBYID[id]; return { peak: h.tflops * 1e12, peakFp8: 2 * h.tflops * 1e12, bw: h.tbps * 1e12, mem: h.gb * 1e9, fp8: false, kernels }; };
  const ridge = (hw) => hw.peak / hw.bw;
  const DG = M.gemma4_26b(), LL = M.MODELS.llama31_8b.build();
  const W0 = REF.w, O0 = REF.o, OM = omega(W0);
  const steady = (m, hw, B, C, o = O0, w = {}) => M.steady(m, { ...W0, ...w, B }, o, hw, C);
  const fits = (m, hw, B, C, o = O0) => M.footprint(m, { ...W0, B, P: C, G: W0.b }, o, 'td').total <= hw.mem;
  // Proposition 1: intensities of Definition 3 (no vocabulary, no activations), and the gain.
  function prop1(m, hw, B, C, w = {}) {
    const ww = { ...W0, ...w, B };
    const c = M.compact({ ...m, V: 0, fSC: 0 }, ww, { ...O0, ba: 0, bl: 0 }, hw);
    const IAR = c.phi(C) / (c.WAR(B) / B + c.qkv(C)), ITD = (ww.b * c.phi(C)) / (c.WTD(B) / B + c.qkvTD(C));
    const Is = c.Istar, speed = Math.min(ITD, Is) / Math.min(IAR, Is);
    return { IAR, ITD, Is, om: c.om, speed, gain: speed / c.om, Pi: c.Pi(B), c };
  }
  // Share of the peak compute that a pass uses: time its FLOPs need at peak over the time it takes.
  const util = (t) => t.tc / t.t;
  const g2 = (x) => (x >= 10 ? x.toFixed(1) : x.toFixed(2));
  const kctx = (C) => (C >= 1 << 20 ? `${fmt(C / (1 << 20))}M` : `${fmt(C / 1024)}k`);
  const pct = (x) => (x >= 0.995 ? '100%' : x >= 0.1 ? `${Math.round(x * 100)}%` : `${fmt(x * 100, 2)}%`);
  // A trailing * on a base is typeset as a superscript star. The decoder labels AR and TD go up, as in the
  // text (I^TD, W^AR), and a star then goes down: sub('B*', 'TD') is B^TD_⋆.
  const STAR = '<tspan baseline-shift="super" font-size="0.75em">⋆</tspan>', HIS = '<i>I</i><sup>⋆</sup>';
  const DEC = (s) => s === 'AR' || s === 'TD';
  const sub = (base, s) => {
    const star = base.endsWith('*'), b = star ? base.slice(0, -1) : base;
    if (DEC(s)) return `${b}<tspan baseline-shift="super" font-size="0.8em">${s}</tspan>${star ? '<tspan baseline-shift="sub" font-size="0.8em">⋆</tspan>' : ''}`;
    return `${star ? b + STAR : b}<tspan baseline-shift="sub" font-size="0.8em">${s}</tspan>`;
  };
  // I^AR_attn and I^TD_attn: the decoder as a superscript, attn as a subscript.
  const IATT = (d) => `<span class="q-${d.toLowerCase()}"><i>I</i><sup>${d}</sup><sub>attn</sub></span>`;
  const hsub = (base, s) => {
    const star = base.endsWith('*'), b = star ? base.slice(0, -1) : base;
    if (!DEC(s)) return star ? `<i>${b}</i><sup>⋆</sup><sub>${s}</sub>` : `<i>${b}</i><sub>${s}</sub>`;
    return `<span class="q-${s.toLowerCase()}"><i>${b}</i><sup>${s}</sup>${star ? '<sub>⋆</sub>' : ''}</span>`;
  };
  const colors = () => ({
    ar: css('--bdr-ar'), td: css('--bdr-td'), ink: css('--ink'), muted: css('--bdr-muted'), rule: css('--rule'),
    bg: css('--bdr-bg'), surface: css('--bdr-surface'), hi: css('--bdr-hi'), soft: css('--bdr-accent-soft'), bad: css('--bdr-critical'),
  });
  function scale(d0, d1, r0, r1, log = true) {
    const f = log ? Math.log10 : (v) => v, a = f(d0), b = f(d1);
    const s = (v) => r0 + ((f(v) - a) / (b - a)) * (r1 - r0);
    s.inv = (p) => { const t = a + ((p - r0) / (r1 - r0)) * (b - a); return log ? 10 ** t : t; };
    return s;
  }
  function svgPoint(svg, ev) {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  const nearestIndex = (arr, v) => arr.reduce((best, x, i) => (Math.abs(Math.log(x / v)) < Math.abs(Math.log(arr[best] / v)) ? i : best), 0);
  function el(html) { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; }
  function slider(label, values, i, show) {
    const lab = el(`<label><span>${label} <output></output></span><input type="range" min="0" max="${values.length - 1}" step="1" value="${i}"></label>`);
    const input = lab.querySelector('input'), out = lab.querySelector('output');
    const sync = () => { out.textContent = show(values[+input.value]); };
    input.addEventListener('input', sync); sync();
    return { lab, input, get value() { return values[+input.value]; }, get index() { return +input.value; }, set(v) { input.value = nearestIndex(values, v); sync(); } };
  }
  function seg(options, current, onPick, name = '') {
    const box = el(`<div class="seg" role="group"${name ? ` aria-label="${name}"` : ''}></div>`);
    for (const [value, text] of options) {
      const b = el(`<button type="button" aria-pressed="${value === current}">${text}</button>`);
      b.addEventListener('click', () => { box.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); onPick(value); });
      box.appendChild(b);
    }
    return box;
  }
  // A captioned button group: role=group named by its caption (a <label> would forward clicks to its first button).
  let groupId = 0;
  function groupOf(text, s, cls = 'wide') {
    const id = `bdr-grp-${++groupId}`;
    const g = el(`<div class="ctl ${cls}" role="group" aria-labelledby="${id}"><span id="${id}">${text}</span><span class="segbox"></span></div>`);
    s.removeAttribute('role'); s.removeAttribute('aria-label');
    g.querySelector('.segbox').appendChild(s);
    return g;
  }
  // Drag a vertical marker along a log axis (pointer, touch, arrow keys). The handlers sit on the
  // container, which survives redraws; the SVG and its .drag group are rebuilt on every move.
  function draggable(box, getXs, values, getIndex, setIndex, axis = 'x') {
    let on = false;
    const move = (ev) => { const p = svgPoint(box.querySelector('svg'), ev); setIndex(nearestIndex(values, getXs().inv(p[axis]))); };
    box.addEventListener('pointerdown', (ev) => { if (!ev.target.closest('.drag')) return; on = true; box.setPointerCapture(ev.pointerId); move(ev); });
    box.addEventListener('pointermove', (ev) => { if (on) move(ev); });
    box.addEventListener('pointerup', () => { on = false; });
    box.addEventListener('pointercancel', () => { on = false; });
    box.addEventListener('keydown', (ev) => {
      if (!ev.target.closest('.drag')) return;
      const n = values.length - 1, d = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -4, PageUp: 4, Home: -n, End: n }[ev.key];
      if (!d) return;
      ev.preventDefault();
      setIndex(Math.max(0, Math.min(n, getIndex() + d)));
      box.querySelector('.drag').focus({ preventScroll: true });
    });
  }
  function logGrid(ys, x0, x1, lo, hi, fmtY, c, marks = [1, 2, 5]) {
    let s = '';
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const k of marks) {
        const v = k * 10 ** e; if (v < lo * 0.999 || v > hi * 1.001) continue;
        s += `<line x1="${x0}" x2="${x1}" y1="${ys(v)}" y2="${ys(v)}" stroke="${c.rule}"/><text x="${x0 - 6}" y="${ys(v) + 4}" text-anchor="end">${fmtY(v)}</text>`;
      }
    }
    return s;
  }
  // Polyline drawn solid where `ok(point)` holds and dashed elsewhere.
  function dashedLine(pts, xy, ok, attrs) {
    let s = '', run = [], dash = null;
    const flush = () => { if (run.length > 1) s += `<polyline fill="none" ${attrs} ${dash ? 'stroke-dasharray="5 4"' : ''} points="${run.join(' ')}"/>`; };
    for (const p of pts) {
      const d = !ok(p);
      if (dash === null) dash = d;
      if (d !== dash) { run.push(xy(p)); flush(); run = [xy(p)]; dash = d; } else run.push(xy(p));
    }
    flush();
    return s;
  }
  // SVGs are drawn at the width they are shown at, so that 12px text stays 12px on a phone.
  const widthOf = (box, lo = 330, hi = 1000) => Math.max(lo, Math.min(hi, Math.round(box.clientWidth || 640)));
  function watchWidth(box, draw) {
    let last = box.clientWidth;
    new ResizeObserver(() => { const w = box.clientWidth; if (Math.abs(w - last) > 4) { last = w; draw(); } }).observe(box);
  }
  const rerenders = [];
  const onTheme = (f) => rerenders.push(f);

  // The accelerator is shared by Figures 6, 12 and 7: choosing one in a figure sets it in all.
  const ACCEL = ['h20', 'rtx5090', 'a100', 'h200', 'v5e', 'b200', 'mi355x', 'h100pcie', 'spark', 'v6e'];
  const SHORT = { h20: 'H20', rtx5090: 'RTX 5090', a100: 'A100', h200: 'H200', v5e: 'TPU v5e', b200: 'B200', mi355x: 'MI355X', h100pcie: 'H100 PCIe', spark: 'DGX Spark', v6e: 'TPU v6e' };
  // labels of Figure 7's dots on narrow screens
  const NARROW = { rtx5090: 'RTX5090', v5e: 'v5e', h100pcie: 'H100P', spark: 'Spark', v6e: 'v6e' };
  // The choice is shared within the page only: every page opens on the H200, the accelerator the text quotes.
  const shared = { hw: 'h200', subs: [] };
  function setHw(id) {
    if (!HWBYID[id] || id === shared.hw) return;
    shared.hw = id;
    shared.subs.forEach((f) => f(id));
  }
  // A line under a figure's controls, shown while the shared accelerator is not the H200.
  function hwNote() {
    const box = el('<div class="hw-note" aria-live="polite" hidden><span></span><button type="button" class="fig-btn">Show the H200</button></div>');
    const sync = (id) => { box.hidden = id === 'h200'; box.querySelector('span').textContent = `Showing the ${SHORT[id] || HWBYID[id].label}; the text quotes the H200.`; };
    box.querySelector('button').addEventListener('click', () => setHw('h200'));
    shared.subs.push(sync); sync(shared.hw);
    return box;
  }
  function hwControl() {
    const lab = el(`<label class="sel">accelerator<select>${ACCEL.map((id) => `<option value="${id}"${id === shared.hw ? ' selected' : ''}>${HWBYID[id].label}</option>`).join('')}</select></label>`);
    const s = lab.querySelector('select');
    s.addEventListener('change', () => setHw(s.value));
    shared.subs.push((id) => { s.value = id; });
    return lab;
  }
  const BATCHES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
  const CONTEXTS = Array.from({ length: 11 }, (_, i) => 1024 * 2 ** i);

  // ---------------------------------------------------------------- roofline renderer (Figure 6)
  // One accelerator's roofline with an AR dot and a TD dot at their intensities; the bracket on the
  // right is the ratio of their attainable FLOP rates, min(I_TD, I*) / min(I_AR, I*).
  function rooflineSvg({ hw, IAR, ITD, W = 640, H = 300, mini = false }) {
    const c = colors(), Is = ridge(hw), pk = hw.peak / 1e12, rate = (I) => Math.min(pk, (I * hw.bw) / 1e12);
    const m = { l: mini ? 40 : 54, r: mini ? 58 : 92, t: 28, b: mini ? 30 : 40 };
    const xlo = 0.3, xhi = 3e4, xs = scale(xlo, xhi, m.l, W - m.r);
    const ylo = rate(xlo) / 1.6, yhi = pk * 1.9, ys = scale(ylo, yhi, H - m.b, m.t);
    const ra = rate(IAR), rt = rate(ITD), cx = (I) => xs(Math.min(Math.max(I, xlo), xhi));
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Roofline: AR at ${fmt(ra)} TFLOP/s, TD at ${fmt(rt)} TFLOP/s">`;
    s += `<defs><marker id="rf-ah${mini ? 'm' : ''}" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${c.ink}"/></marker></defs>`;
    s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => (v >= 1 ? fmt(v) : v.toPrecision(1)), c, [1]);
    for (let e = 0; e <= 4; e++) s += `<text x="${xs(10 ** e)}" y="${H - m.b + 15}" text-anchor="middle">${10 ** e >= 1000 ? `${10 ** (e - 3)}k` : 10 ** e}</text>`;
    if (!mini) {
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">arithmetic intensity, FLOP per byte</text>`;
      s += `<text x="4" y="13" class="ink">attainable FLOP rate, TFLOP/s</text>`;
    }
    s += `<line x1="${xs(Is)}" x2="${xs(Is)}" y1="${ys(pk)}" y2="${H - m.b}" stroke="${c.muted}" stroke-dasharray="3 4"/>`;
    s += `<text x="${xs(Is) + 5}" y="${H - m.b - 6}">${mini ? `I${STAR} ` : `ridge I${STAR} = `}${fmt(Is)}</text>`;
    s += `<polyline fill="none" stroke="${c.ink}" stroke-width="2" points="${[xlo, Is, xhi].map((I) => `${xs(I)},${ys(rate(I))}`).join(' ')}"/>`;
    if (!mini) s += `<text x="${xs(Is * 3)}" y="${ys(pk) - 6}" class="ink">peak π = ${fmt(pk)} TFLOP/s</text>`;
    const bx = W - m.r + (mini ? 14 : 22);
    s += `<line x1="${cx(IAR)}" x2="${bx}" y1="${ys(ra)}" y2="${ys(ra)}" stroke="${c.ar}" stroke-dasharray="3 3"/>`;
    s += `<line x1="${cx(ITD)}" x2="${bx}" y1="${ys(rt)}" y2="${ys(rt)}" stroke="${c.td}" stroke-dasharray="3 3"/>`;
    if (Math.abs(ys(ra) - ys(rt)) > 10) s += `<line x1="${bx}" x2="${bx}" y1="${Math.min(ys(ra), ys(rt)) + 4}" y2="${Math.max(ys(ra), ys(rt)) - 4}" stroke="${c.ink}" stroke-width="1.5" marker-start="url(#rf-ah${mini ? 'm' : ''})" marker-end="url(#rf-ah${mini ? 'm' : ''})"/>`;
    s += `<text class="strong" x="${bx + 6}" y="${(ys(ra) + ys(rt)) / 2 + 4}">×${fmt(rt / ra, 3)}</text>`;
    const lab = (I, r, cls, txt, below) => `<text class="${cls} halo" x="${cx(I) + (below ? 9 : -9)}" y="${ys(r) + (below ? 16 : -10)}" text-anchor="${below ? 'start' : 'end'}">${txt}</text>`;
    s += lab(IAR, ra, 'ar', mini ? 'AR' : 'AR step', true);
    s += lab(ITD, rt, 'td', mini ? 'TD' : 'TD pass', false);
    s += `<circle cx="${cx(IAR)}" cy="${ys(ra)}" r="${mini ? 5 : 6.5}" fill="${c.ar}" stroke="${c.bg}" stroke-width="2"/>`;
    s += `<circle cx="${cx(ITD)}" cy="${ys(rt)}" r="${mini ? 5 : 6.5}" fill="${c.td}" stroke="${c.bg}" stroke-width="2"/>`;
    return `${s}</svg>`;
  }

  // ---------------------------------------------------------------- Figure 5: byte reuse (dense model)
  function figReuse(root) {
    const st = { b: 256, S: 14, t: 0, timer: 0 };
    const ctl = el('<div class="fig-controls"></div>');
    const bSeg = groupOf('canvas length <i>b</i>', seg([16, 32, 64, 128, 256].map((v) => [v, String(v)]), st.b, (v) => { st.b = v; reset(); }));
    const sS = slider('denoising passes <i>S</i>', Array.from({ length: 32 }, (_, i) => i + 1), st.S - 1, (v) => `${v}  (ω = ${v + 1})`);
    sS.input.addEventListener('input', () => { st.S = sS.value; reset(); });
    const btns = el('<div class="seg" role="group" aria-label="animation"><button type="button" class="fig-btn primary" data-a="play">Play</button><button type="button" class="fig-btn" data-a="step">Step</button><button type="button" class="fig-btn" data-a="reset">Reset</button></div>');
    ctl.append(bSeg, sS.lab, btns);
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    const om = () => st.S + 1;
    const stop = () => { clearInterval(st.timer); st.timer = 0; btns.querySelector('[data-a=play]').textContent = st.t >= om() ? 'Replay' : 'Play'; };
    const step = () => { if (st.t < om()) st.t += 1; draw(); if (st.t >= om()) stop(); };
    function reset() { stop(); st.t = 0; btns.querySelector('[data-a=play]').textContent = 'Play'; draw(); }
    btns.addEventListener('click', (ev) => {
      const a = ev.target.dataset && ev.target.dataset.a; if (!a) return;
      if (a === 'reset') reset();
      if (a === 'step') { stop(); step(); }
      if (a === 'play') {
        if (st.timer) { stop(); return; }
        if (st.t >= om()) st.t = 0;
        const dt = matchMedia('(prefers-reduced-motion: reduce)').matches ? 700 : 380;
        st.timer = setInterval(step, dt); ev.target.textContent = 'Pause'; step();
      }
    });
    function draw() {
      const c = colors(), W = widthOf(pic), x0 = W < 480 ? 136 : 160, x1 = W - 16, w = x1 - x0, n = om(), t = st.t, b = st.b;
      const cell = w / Math.max(b, n), committed = t >= n;
      let s = `<svg viewBox="0 0 ${W} 198" role="img" aria-label="After ${t} of ${n} passes: AR has ${t} tokens, TD ${committed ? b : 0}">`;
      // shared time axis: one slot per pass
      s += `<text x="0" y="27" class="ink">forward passes</text>${W < 480 ? '' : '<text x="0" y="41">one weight read each</text>'}`;
      const sw = w / n;
      for (let j = 0; j < n; j++) {
        const done = j < t, last = j === n - 1;
        s += `<rect x="${x0 + j * sw + 1}" y="18" width="${Math.max(2, sw - 2)}" height="22" rx="2" fill="${done ? c.ink : 'none'}" fill-opacity="${done ? 0.82 : 0}" stroke="${c.muted}" stroke-opacity="0.6"/>`;
        if (sw > 22) s += `<text x="${x0 + j * sw + sw / 2}" y="33" text-anchor="middle" style="fill:${done ? c.bg : c.muted};font-size:12px">${last ? (sw >= 50 ? 'commit' : '+1') : j + 1}</text>`;
      }
      // AR: one token per pass
      s += `<text x="0" y="84" class="ar">AR</text><text x="0" y="98">1 token per pass</text>`;
      for (let j = 0; j < n; j++) s += `<rect x="${x0 + j * cell}" y="70" width="${Math.max(1, cell - (cell > 4 ? 1.5 : 0.4))}" height="26" fill="${j < t ? c.ar : 'none'}" stroke="${c.ar}" stroke-opacity="${j < t ? 1 : 0.3}" stroke-width="${cell > 4 ? 1 : 0.4}"/>`;
      // TD: a canvas of b positions, all updated by every pass, committed by the last one
      s += `<text x="0" y="142" class="td">TD</text><text x="0" y="156">b positions</text>`;
      const shade = committed ? 1 : Math.min(t, st.S) / st.S * 0.75;
      for (let j = 0; j < b; j++) s += `<rect x="${x0 + j * cell}" y="128" width="${Math.max(1, cell - (cell > 4 ? 1.5 : 0.4))}" height="26" fill="${c.td}" fill-opacity="${committed ? 1 : shade * 0.85}" stroke="${c.td}" stroke-opacity="${committed ? 1 : 0.35}" stroke-width="${cell > 4 ? 1 : 0.4}"/>`;
      if (!committed && t > 0) s += `<text x="${x0}" y="174">denoising: every position refined by pass ${Math.min(t, st.S)} of ${st.S}</text>`;
      if (committed) s += `<text x="${x0}" y="174" class="td">committed: ${b} tokens</text>`;
      s += '</svg>';
      pic.innerHTML = s;
      const tpf = b / n;
      out.innerHTML = t < n
        ? `After <b>${t}</b> of <b>${n}</b> passes: <span class="ar">AR</span> has committed ${t} token${t === 1 ? '' : 's'}, <span class="td">TD</span> none yet.`
        : `<span class="big">In ${n} passes, <span class="ar">AR</span> commits ${n} tokens and <span class="td">TD</span> commits ${b}: TPF = ${b}/${n} = <b>${fmt(tpf)}</b>.</span>`;
    }
    reset();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 6: live roofline (Proposition 1)
  function figRoofline(root) {
    const ctl = el('<div class="fig-controls"></div>');
    const sB = slider('batch <i>B</i>', BATCHES, 0, (v) => String(v));
    const sC = slider('context <i>C</i>', CONTEXTS, 0, kctx);
    ctl.append(hwControl(), sB.lab, sC.lab);
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, hwNote(), pic, out);
    sB.input.addEventListener('input', draw);
    sC.input.addEventListener('input', draw);
    function draw() {
      const hw = hwOf(shared.hw), B = sB.value, C = sC.value, p = prop1(DG, hw, B, C);
      pic.innerHTML = rooflineSvg({ hw, IAR: p.IAR, ITD: p.ITD, W: widthOf(pic), H: widthOf(pic) < 480 ? 260 : 300 });
      const bound = (I) => (I >= p.Is ? 'compute-bound' : 'memory-bound');

    }
    shared.subs.push(draw);
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- best-batch figure (no longer placed in the text)
  const BSCAN = [...new Set([...Array.from({ length: 40 }, (_, i) => i + 1), ...Array.from({ length: 60 }, (_, i) => Math.round(40 * 1.056 ** (i + 1))), 64, 128, 256, 512, 1024])].filter((B) => B <= 1024).sort((a, b) => a - b);
  const curveCache = new Map();
  function gainCurve(hwId, rule) {
    const key = `${hwId}|${rule}`;
    if (curveCache.has(key)) return curveCache.get(key);
    const hw = hwOf(hwId, rule);
    const pts = BSCAN.map((B) => { const r = steady(DG, hw, B, 1024); return { B, g: r.gain, td: r.tdCompute, uAR: util(r.ar), uTD: util(r.d1), Pi: prop1(DG, hw, B, 1024).Pi, fit: fits(DG, hw, B, 1024) }; });
    const best = pts.reduce((a, p) => (p.g > a.g ? p : a));
    const bstar = (pts.find((p) => p.td) || {}).B;
    const k = pts.findIndex((p) => p.g < 1), parity = k > 0 ? pts[k - 1].B + (1 - pts[k - 1].g) / (pts[k].g - pts[k - 1].g) * (pts[k].B - pts[k - 1].B) : null;
    const res = { pts, best, bstar, parity };
    curveCache.set(key, res);
    return res;
  }
  function openCalculator(hwId, B, rule) {
    const set = (id, v) => { const e = document.getElementById(id); if (!e) return; e.value = v; e.dispatchEvent(new Event('change', { bubbles: true })); };
    set('c-hw', hwId); set('c-kern', rule); set('c-B', B);
    const calc = document.getElementById('calc');
    if (calc) calc.scrollIntoView();
    else location.href = calcHref(hwId, B, rule);
  }
  // The calculator is a page of its own; its settings travel in the address (hw, kern, B).
  function calcHref(hwId, B, rule) {
    let url = '#calc';
    try { const c = JSON.parse(document.getElementById('bdr-book').textContent).chapters.find((x) => x.slug === 'calculator'); if (c) url = c.url; } catch (e) { /* single page */ }
    if (url === '#calc') return url;
    return `${url}?${new URLSearchParams({ hw: hwId, kern: rule, B: String(B) })}#calc`;
  }
  // ---------------------------------------------------------------- Figure 12: both intensities become independent of the batch (Proposition 1)
  const CSCAN = Array.from({ length: 41 }, (_, i) => Math.round(1024 * 2 ** (i / 4)));
  const CBATCH = [1, 4, 16, 64];
  function figContext(root) {
    const st = { i: CSCAN.indexOf(1 << 20) };
    const ctl = el('<div class="fig-controls"></div>');
    ctl.append(hwControl());
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, hwNote(), pic, out);
    // The intensities do not depend on the accelerator; only the ridge and the memory limit do.
    const curves = CBATCH.map((B) => CSCAN.map((C) => { const p = prop1(DG, hwOf('h200'), B, C); return { B, C, IAR: p.IAR, ITD: p.ITD }; }));
    draggable(pic, () => st.xs, CSCAN, () => st.i, (i) => { st.i = i; draw(); });
    function draw() {
      const c = colors(), W = widthOf(pic), H = 344, m = { l: 48, r: 18, t: 30, b: 40 }, C = CSCAN[st.i];
      const hw = hwOf(shared.hw), Is = ridge(hw), ok = (p) => fits(DG, hw, p.B, p.C);
      const xs = scale(1024, 1 << 20, m.l, W - m.r), ylo = 0.7, yhi = 6000, ys = scale(ylo, yhi, H - m.b, m.t);
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Intensities of an AR step and of a TD pass against context, for batches ${CBATCH.join(', ')}">`;
      s += `<rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${ys(Is) - m.t}" fill="${c.soft}" opacity="0.7"/>`;
      s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => (v >= 1000 ? `${v / 1000}k` : v), c, [1]);
      for (const C2 of [1024, 4096, 16384, 65536, 262144, 1 << 20]) s += `<text x="${xs(C2)}" y="${H - m.b + 15}" text-anchor="middle">${kctx(C2)}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">${W < 480 ? 'context C, tokens' : 'context C, tokens in the KV cache'}</text>`;
      s += `<text x="4" y="13" class="ink">arithmetic intensity, FLOP per byte</text>`;
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(Is)}" y2="${ys(Is)}" stroke="${c.ink}" stroke-width="1.5"/>`;
      s += `<text class="ink halo" x="${W - m.r - 4}" y="${ys(Is) - 6}" text-anchor="end">ridge I${STAR} = ${fmt(Is)}: compute-bound above</text>`;
      const op = [0.4, 0.6, 0.8, 1];
      // AR curves start close together: lift each label at least 15px above the one of the batch below.
      const arY = [];
      curves.forEach((pts, k) => { const y = ys(pts[0].IAR) - 6; arY.push(k ? Math.min(y, arY[k - 1] - 15) : y); });
      curves.forEach((pts, k) => {
        s += dashedLine(pts, (p) => `${xs(p.C)},${ys(p.ITD)}`, ok, `stroke="${c.td}" stroke-opacity="${op[k]}" stroke-width="2.2"`);
        s += dashedLine(pts, (p) => `${xs(p.C)},${ys(p.IAR)}`, ok, `stroke="${c.ar}" stroke-opacity="${op[k]}" stroke-width="2.2"`);
        const yl = ys(pts[0].ITD) - 6, onRidge = Math.abs(yl - 4 - ys(Is)) < 9;
        s += `<text class="td halo" x="${xs(1024) + 6}" y="${onRidge ? yl + 20 : yl}" style="opacity:${0.55 + 0.45 * op[k]}">B = ${CBATCH[k]}</text>`;
        s += `<text class="ar halo" x="${xs(1024) + 6}" y="${arY[k]}" style="opacity:${0.55 + 0.45 * op[k]}">B = ${CBATCH[k]}</text>`;
      });
      const mid = CSCAN[18];
      s += `<text class="td halo" x="${xs(mid)}" y="${ys(curves[3][18].ITD) - 10}" text-anchor="middle">TD pass, ${sub('I', 'TD')}</text>`;
      s += `<text class="ar halo" x="${xs(mid)}" y="${ys(curves[0][18].IAR) + 20}" text-anchor="middle">AR step, ${sub('I', 'AR')}</text>`;
      s += `<g class="drag" tabindex="0" role="slider" aria-label="Context" aria-valuemin="1024" aria-valuemax="${1 << 20}" aria-valuenow="${C}" aria-valuetext="context ${kctx(C)} tokens"><rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${H - m.b - m.t}" fill="transparent"/>`;
      s += `<line x1="${xs(C)}" x2="${xs(C)}" y1="${m.t}" y2="${H - m.b}" stroke="${c.hi}" stroke-width="2"/>`;
      s += `<path d="M${xs(C) - 7},${H - m.b} l7,-11 l7,11 z" fill="${c.hi}"/>`;
      curves.forEach((pts, k) => {
        s += `<circle cx="${xs(C)}" cy="${ys(pts[st.i].ITD)}" r="4" fill="${c.td}" fill-opacity="${op[k]}" stroke="${c.bg}" stroke-width="1.5"/>`;
        s += `<circle cx="${xs(C)}" cy="${ys(pts[st.i].IAR)}" r="4" fill="${c.ar}" fill-opacity="${op[k]}" stroke="${c.bg}" stroke-width="1.5"/>`;
      });
      s += '</g></svg>';
      pic.innerHTML = s;
      st.xs = xs;
      const at = curves.map((pts) => pts[st.i]), ar = at.map((p) => p.IAR), td = at.map((p) => p.ITD);
      const gains = at.map((p) => Math.min(p.ITD, Is) / (OM * Math.min(p.IAR, Is)));
      const above = td.filter((v) => v >= Is).length, noFit = at.filter((p) => !ok(p)).map((p) => p.B);

    }
    shared.subs.push(draw);
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 7: the gain is a function of the ridge point (exact)
  // Scaling peak and bandwidth together scales every pass time alike, so the gain depends on the
  // accelerator only through I* = peak / bandwidth. The curve is drawn for a chip with unlimited memory.
  function figAccel(root) {
    const RIDGES = Object.fromEntries(ACCEL.map((id) => [id, ridge(hwOf(id))]));
    const ISCAN = [...new Set([...Array.from({ length: 31 }, (_, i) => 25 * 32 ** (i / 30)), ...Object.values(RIDGES)])].sort((a, b) => a - b);
    const st = { i: ISCAN.indexOf(RIDGES[shared.hw]) };
    const pic = el('<div></div>');
    root.append(hwNote(), pic);
    let data = null;
    function compute() {
      const Ikv = M.compact(DG, W0, O0, hwOf('h200')).Ikv;
      const pts = ISCAN.map((Is) => {
        const hw = { peak: Is * 1e12, peakFp8: 2 * Is * 1e12, bw: 1e12, mem: Infinity, fp8: false, kernels: 'whole' };
        return { Is, long: steady(DG, hw, 1, 1 << 20).gain };
      });
      const dots = ACCEL.map((id) => ({ id, p: pts[ISCAN.indexOf(RIDGES[id])], fitLong: fits(DG, hwOf(id), 1, 1 << 20) }));
      data = { pts, dots, Ikv };
    }
    draggable(pic, () => st.xs, ISCAN, () => st.i, (i) => { st.i = i; const hit = ACCEL.find((id) => RIDGES[id] === ISCAN[i]); if (hit) setHw(hit); draw(); });
    function draw() {
      if (!data) compute();
      const c = colors(), W = widthOf(pic), H = 330, m = { l: 48, r: 22, t: 28, b: 42 };
      const xs = scale(25, 800, m.l, W - m.r), ylo = 0.25, yhi = 14, ys = scale(ylo, yhi, H - m.b, m.t);
      const val = (p) => p.long, P = data.pts[st.i];
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Roofline bound Γ against ridge point">`;
      s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => `${v}×`, c);
      for (const v of [25, 50, 100, 200, 400, 800]) s += `<text x="${xs(v)}" y="${H - m.b + 15}" text-anchor="middle">${v}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">ridge point I${STAR} = π/β, FLOP per byte</text>`;
      s += `<text x="4" y="13" class="ink">roofline bound Γ at batch 1, context 1M</text>`;
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(1)}" y2="${ys(1)}" stroke="${c.muted}" stroke-width="1.5"/><text x="${W - m.r - 4}" y="${ys(1) + 15}" text-anchor="end">parity, Γ = 1</text>`;
      const f = (Is) => Is / (OM * data.Ikv), xa = Math.max(25, ylo * OM * data.Ikv);  // the limit C → ∞, I*/(ω I_attn): dashed
      s += `<line x1="${xs(xa)}" x2="${xs(800)}" y1="${ys(f(xa))}" y2="${ys(f(800))}" stroke="${c.muted}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
      s += `<polyline fill="none" stroke="${c.ink}" stroke-width="2.2" stroke-linejoin="round" points="${data.pts.map((p) => `${xs(p.Is)},${ys(val(p))}`).join(' ')}"/>`;
      // marker, under the dots so that a dot stays clickable
      s += `<g class="drag" tabindex="0" role="slider" aria-label="Ridge point" aria-valuemin="25" aria-valuemax="800" aria-valuenow="${Math.round(P.Is)}" aria-valuetext="ridge point ${fmt(P.Is)}, gain ${g2(val(P))}"><rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${H - m.b - m.t}" fill="transparent"/>`;
      s += `<line x1="${xs(P.Is)}" x2="${xs(P.Is)}" y1="${m.t}" y2="${H - m.b}" stroke="${c.hi}" stroke-width="2"/><path d="M${xs(P.Is) - 7},${H - m.b} l7,-11 l7,11 z" fill="${c.hi}"/></g>`;
      // every dot is labelled: in ridge order, each label takes the first free slot above or below its dot,
      // with a leader line when it is not the nearest slot
      const placed = [], narrowDots = W < 560;
      const name = (id) => (narrowDots ? NARROW[id] || SHORT[id] : SHORT[id]);
      const slots = [-13, 22, -28, 37, -43, 52];
      const free = (b) => b.x0 >= 2 && b.x1 <= W - 2 && b.y0 >= m.t - 14 && b.y1 <= H - m.b && placed.every((q) => b.x1 < q.x0 - 3 || b.x0 > q.x1 + 3 || b.y1 < q.y0 - 1 || b.y0 > q.y1 + 1);
      const order = data.dots.map((d, k) => k).sort((a, b) => data.dots[a].p.Is - data.dots[b].p.Is);
      data.dots.forEach((d) => { const x = xs(d.p.Is), y = ys(val(d.p)); placed.push({ x0: x - 8, x1: x + 8, y0: y - 8, y1: y + 8 }); });
      const labels = [];
      for (const k of order) {
        const d = data.dots[k], x = xs(d.p.Is), y = ys(val(d.p)), on = d.id === shared.hw && RIDGES[d.id] === P.Is;
        const w = name(d.id).length * 7.4 * (on ? 1.05 : 1);
        let pick = null;
        for (const dy of slots) { const b = { x0: x - w / 2, x1: x + w / 2, y0: y + dy - 11, y1: y + dy + 2 }; if (free(b)) { pick = { dy, b }; break; } }
        if (!pick) pick = { dy: slots[0], b: { x0: x - w / 2, x1: x + w / 2, y0: y - 24, y1: y - 11 } };
        placed.push(pick.b);
        labels[k] = { x, y, dy: pick.dy, on };
      }
      data.dots.forEach((d, k) => {
        const fit = d.fitLong, { x, y, dy, on } = labels[k];
        s += `<g class="dot" data-id="${d.id}" tabindex="0" role="button" aria-pressed="${on}" aria-label="${HWBYID[d.id].label}" style="cursor:pointer">`;
        s += `<circle cx="${x}" cy="${y}" r="14" fill="transparent"/>`;
        if (Math.abs(dy) > 22) s += `<line x1="${x}" x2="${x}" y1="${y + (dy < 0 ? -7 : 7)}" y2="${y + (dy < 0 ? dy + 3 : dy - 12)}" stroke="${c.muted}" stroke-width="0.8"/>`;
        if (on) s += `<circle cx="${x}" cy="${y}" r="10.5" fill="none" stroke="${c.hi}" stroke-width="2"/>`;
        s += `<circle cx="${x}" cy="${y}" r="${on ? 6.5 : 5.5}" fill="${fit ? c.ink : c.bg}" stroke="${c.ink}" stroke-width="2"/>`;
        s += `<text class="${on ? 'strong' : 'ink'} halo" x="${x}" y="${y + dy}" text-anchor="middle">${name(d.id)}</text>`;
        s += '</g>';
      });
      s += `<circle cx="${xs(P.Is)}" cy="${ys(val(P))}" r="3.5" fill="${c.hi}" pointer-events="none"/>`;
      pic.innerHTML = `${s}</svg>`;
      st.xs = xs;
      pic.querySelectorAll('.dot').forEach((g) => {
        const pick = () => { st.i = ISCAN.indexOf(RIDGES[g.dataset.id]); if (g.dataset.id === shared.hw) draw(); else setHw(g.dataset.id); pic.querySelector(`.dot[data-id="${g.dataset.id}"]`).focus({ preventScroll: true }); };
        g.addEventListener('click', pick);
        g.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { pick(); ev.preventDefault(); } });
      });
    }
    shared.subs.push((id) => { st.i = ISCAN.indexOf(RIDGES[id]); draw(); });
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- helpers of the figures of Parts 2-3 below
  const gx = (x) => (x >= 100 ? String(Math.round(x)) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
  const gbs = (x) => (x / 1e9 >= 10 ? (x / 1e9).toFixed(1) : (x / 1e9).toFixed(2));
  const segLabel = (text, s) => groupOf(text, s);
  const marker = (xs, v, y0, y1, c) => `<line x1="${xs(v)}" x2="${xs(v)}" y1="${y0}" y2="${y1}" stroke="${c.hi}" stroke-width="2"/><path d="M${xs(v) - 7},${y1} l7,-11 l7,11 z" fill="${c.hi}"/>`;
  // A small deterministic generator, so that a random draw can be repeated and redrawn on resize.
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  // k distinct values of 0..n-1, uniformly.
  function pick(rand, n, k) { const s = new Set(); while (s.size < k) s.add(Math.floor(rand() * n)); return s; }

  // ---------------------------------------------------------------- the model, as the propositions use it
  // The compact model of the propositions: no vocabulary, no activations (as prop1 above).
  const bare = (m, w = {}) => M.compact({ ...m, V: 0, fSC: 0 }, { ...W0, ...w }, { ...O0, ba: 0, bl: 0 }, REF.hw);
  const C0 = bare(DG), NLAYERS = DG.layers.length, WE = O0.bw * M.expertParams(DG); // bytes of one expert
  const expertsOf = (N) => M.distinctExperts(DG, N); // U_{E,k}(N)
  // Weights an AR step and a TD pass read at batch B, split into the part every token reads and the experts touched.
  function weights(B, b) {
    const c = b === W0.b ? C0 : bare(DG, { b });
    const WAR = c.WAR(B), WTD = c.WTD(B), eAR = NLAYERS * expertsOf(B) * WE, eTD = NLAYERS * expertsOf(B * b) * WE;
    return { WAR, WTD, eAR, eTD, dense: WAR - eAR, PiE: expertsOf(B * b) / expertsOf(B), thetaE: eAR / WAR, Pi: WTD / WAR, Wtot: c.Wtot };
  }

  // The batch and the canvas are shared by the two figures on the experts.
  const lin = { B: 1, b: 256, subs: [] };
  const LB = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
  const setLin = (patch) => { Object.assign(lin, patch); lin.subs.forEach((f) => f()); };
  function linControls(extra) {
    const ctl = el('<div class="fig-controls"></div>');
    const sB = slider('batch <i>B</i>', LB, LB.indexOf(lin.B), (v) => String(v));
    sB.input.addEventListener('input', () => setLin({ B: sB.value }));
    const bs = seg([8, 32, 256].map((v) => [v, String(v)]), lin.b, (v) => setLin({ b: v }));
    ctl.append(sB.lab, segLabel('canvas length <i>b</i>', bs));
    if (extra) ctl.append(extra);
    lin.subs.push(() => { sB.set(lin.B); bs.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x.textContent === String(lin.b)))); });
    return ctl;
  }

  // ---------------------------------------------------------------- weights read by a pass (Proposition 3)
  function figWeights(root) {
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(linControls(), pic, out);
    function draw() {
      const c = colors(), W = widthOf(pic), { B, b } = lin, w = weights(B, b), narrow = W < 480;
      const m = { l: narrow ? 8 : 92, r: 16 }, xs = scale(0, 52e9, m.l, W - m.r, false), bh = 30;
      const rowsY = narrow ? [44, 136] : [30, 92], H = rowsY[1] + bh + 46;
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Weights read by one pass at batch ${B}: AR step ${gbs(w.WAR)} GB, TD pass ${gbs(w.WTD)} GB">`;
      for (const v of [0, 10, 20, 30, 40, 50]) s += `<line x1="${xs(v * 1e9)}" x2="${xs(v * 1e9)}" y1="${rowsY[0] - 10}" y2="${rowsY[1] + bh + 8}" stroke="${c.rule}"/><text x="${xs(v * 1e9)}" y="${rowsY[1] + bh + 22}" text-anchor="middle">${v}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">weights read by one pass, GB${narrow ? '' : ' (BF16, no vocabulary)'}</text>`;
      s += `<line x1="${xs(w.Wtot)}" x2="${xs(w.Wtot)}" y1="${rowsY[0] - 14}" y2="${rowsY[1] + bh + 8}" stroke="${c.muted}" stroke-dasharray="3 4"/><text x="${xs(w.Wtot) - 4}" y="${rowsY[0] - 16}" text-anchor="end">every weight: ${gbs(w.Wtot)} GB</text>`;
      const row = (y, name, cls, col, experts, total) => {
        let g = narrow ? `<text x="${m.l}" y="${y - 8}" class="${cls}">${name}</text>` : `<text x="${m.l - 8}" y="${y + bh / 2 + 4}" text-anchor="end" class="${cls}">${name}</text>`;
        g += `<rect x="${xs(0)}" y="${y}" width="${xs(w.dense) - xs(0)}" height="${bh}" fill="${c.muted}" fill-opacity="0.55"/>`;
        g += `<rect x="${xs(w.dense)}" y="${y}" width="${Math.max(1, xs(w.dense + experts) - xs(w.dense))}" height="${bh}" fill="${col}"/>`;
        const xe = xs(total), inside = xe > W - m.r - 62;
        g += `<text x="${inside ? xe - 6 : xe + 6}" y="${y + bh / 2 + 4}" text-anchor="${inside ? 'end' : 'start'}" class="strong"${inside ? ` style="fill:${c.bg}"` : ''}>${gbs(total)} GB</text>`;
        return g;
      };
      s += row(rowsY[0], 'AR step', 'ar', c.ar, w.eAR, w.WAR);
      s += row(rowsY[1], 'TD pass', 'td', c.td, w.eTD, w.WTD);
      const ly = (rowsY[0] + bh + rowsY[1]) / 2 + (narrow ? -12 : 4);
      s += `<rect x="${xs(0)}" y="${ly - 9}" width="10" height="10" fill="${c.muted}" fill-opacity="0.55"/><text x="${xs(0) + 15}" y="${ly}">read by every token: ${gbs(w.dense)} GB</text>`;
      const ex = narrow ? [xs(0), ly + 15] : [xs(0) + 250, ly];
      s += `<rect x="${ex[0]}" y="${ex[1] - 9}" width="10" height="10" fill="${c.ar}"/><rect x="${ex[0] + 12}" y="${ex[1] - 9}" width="10" height="10" fill="${c.td}"/><text x="${ex[0] + 27}" y="${ex[1]}">experts touched</text>`;
      pic.innerHTML = `${s}</svg>`;
      out.innerHTML = `At batch ${B}, both passes read the ${gbs(w.dense)} GB that every token uses (attention and dense layers, routers). An <span class="ar">AR step</span> adds ${gbs(w.eAR)} GB of experts and a <span class="td">TD pass</span> of ${b} positions per sequence ${gbs(w.eTD)} GB, ${hsub('Π', 'E')} = ${gx(w.PiE)} times as much.<br>`
        + `<span class="big">${hsub('Π', 'lin')} = ${hsub('W', 'TD')}/${hsub('W', 'AR')} = ${gbs(w.WTD)}/${gbs(w.WAR)} = <b>${gx(w.Pi)}</b>.</span> ${w.Pi < 1.1 ? 'The batch alone touches almost every expert, and the penalty is below 1.1.' : `The shared ${gbs(w.dense)} GB dilute the expert ratio ${gx(w.PiE)} to ${gx(w.Pi)}.`}`;
    }
    lin.subs.push(draw);
    draw(); onTheme(draw); watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- where the weights dominate (Proposition 5)
  // C_lin(B, x): the context up to which the weights are at least a share x of a pass's bytes.
  const clin = (WB, x) => (((1 - x) / x) * WB - C0.nuS * C0.r) / C0.nuG;
  const BS = Array.from({ length: 65 }, (_, i) => 2 ** (i / 8)); // batch 1 to 256
  const BMARK = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  function figLinMap(root) {
    const st = { i: 3 };
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(pic, out);
    const ar = (B, x) => clin(C0.WAR(B) / B, x), td = (B, x) => clin(C0.WTD(B) / B, x);
    // Context on the horizontal axis, as in the maps of Part 3; the batch marker is dragged vertically.
    draggable(pic, () => st.ys, BMARK, () => st.i, (i) => { st.i = i; draw(); }, 'y');
    function draw() {
      const c = colors(), W = widthOf(pic), H = 354, m = { l: 50, r: 16, t: 28, b: 42 }, B = BMARK[st.i];
      const clo = 1024, chi = 2 ** 22, xs = scale(clo, chi, m.l, W - m.r), ys = scale(1, 256, H - m.b, m.t);
      const xc = (v) => xs(Math.max(clo, Math.min(chi, v)));
      const path = (f) => BS.map((b2) => `${xc(f(b2))},${ys(b2)}`).join(' ');
      const area = (f) => `${xs(clo)},${ys(1)} ${path(f)} ${xs(clo)},${ys(256)}`;
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Context up to which the weights are at least 90% and 50% of a pass's bytes, for each batch">`;
      s += `<polygon points="${area((b2) => ar(b2, 0.5))}" fill="${c.ar}" fill-opacity="0.13"/><polygon points="${area((b2) => ar(b2, 0.9))}" fill="${c.ar}" fill-opacity="0.2"/>`;
      for (const v of [1024, 4096, 16384, 65536, 262144, 1 << 20, 1 << 22]) s += `<line x1="${xs(v)}" x2="${xs(v)}" y1="${m.t}" y2="${H - m.b}" stroke="${c.rule}"/><text x="${xs(v)}" y="${H - m.b + 15}" text-anchor="middle">${kctx(v)}</text>`;
      for (const v of BMARK) s += `<text x="${m.l - 6}" y="${ys(v) + 4}" text-anchor="end">${v}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">context C, tokens (log scale)</text>`;
      s += `<text x="4" y="13" class="ink">batch B (log scale)</text>`;
      s += `<polyline fill="none" stroke="${c.td}" stroke-width="2" stroke-dasharray="6 4" points="${path((b2) => td(b2, 0.5))}"/>`;
      s += `<polyline fill="none" stroke="${c.ar}" stroke-width="2.2" points="${path((b2) => ar(b2, 0.5))}"/><polyline fill="none" stroke="${c.ar}" stroke-width="1.4" points="${path((b2) => ar(b2, 0.9))}"/>`;
      const small = W < 480;
      s += `<text class="ink halo" x="${xs(1150)}" y="${ys(1.55)}">weights ≥ 90%</text>`;
      s += `<text class="ink halo" x="${xs(small ? 1150 : 26000)}" y="${ys(small ? 3.2 : 1.55)}">weights ≥ 50%</text>`;
      s += `<text class="ink halo" x="${W - m.r - 6}" y="${ys(180)}" text-anchor="end">${small ? 'KV cache > 50% of AR bytes' : "the KV cache is most of an AR step's bytes"}</text>`;
      s += `<text class="td halo" x="${xc(td(16, 0.5)) + 8}" y="${ys(16) + 4}">${small ? 'TD, 50%' : 'TD pass: weights ≥ 50%'}</text>`;
      const y = ys(B);
      s += `<g class="drag" tabindex="0" role="slider" aria-orientation="vertical" aria-label="Batch" aria-valuemin="1" aria-valuemax="256" aria-valuenow="${B}" aria-valuetext="batch ${B}"><rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${H - m.b - m.t}" fill="transparent"/>`;
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" stroke="${c.hi}" stroke-width="2"/><path d="M${m.l},${y - 7} l11,7 l-11,7 z" fill="${c.hi}"/>`;
      for (const [f, col] of [[ar(B, 0.9), c.ar], [ar(B, 0.5), c.ar], [td(B, 0.5), c.td]]) if (f >= clo && f <= chi) s += `<circle cx="${xs(f)}" cy="${y}" r="4.5" fill="${col}" stroke="${c.bg}" stroke-width="1.5"/>`;
      pic.innerHTML = `${s}</g></svg>`;
      st.ys = ys;
      // "up to C = ..." or, when the cache of a sequence with full windows already exceeds the complementary share, no such context
      const upTo = (v, rest) => (v <= C0.r ? `at no context with full windows (for <i>C</i> ≥ <i>w</i> the cache already exceeds ${rest}% of the bytes)` : `up to <b>C = ${comma(v)}</b> tokens`);
      out.innerHTML = `At <b>batch ${B}</b>, the weights are at least 90% of an <span class="ar">AR step</span>'s bytes ${upTo(ar(B, 0.9), 10)} and at least half ${upTo(ar(B, 0.5), 50)}. A <span class="td">TD pass</span> reads ${hsub('Π', 'lin')}(${B}) = ${gx(C0.Pi(B))} times the weights of an AR step, and they are at least half of its bytes ${upTo(td(B, 0.5), 50)}.`;
    }
    draw(); onTheme(draw); watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- what a pass reads of one sequence's cache (Table 1)
  // A toy cache: columns are the cached tokens of one sequence, rows are the queries of a pass. A filled cell
  // is a (query, key) pair that the layer scores, so FLOPs count cells; bytes count the columns that hold one.
  function figAttnRead(root) {
    const TOY = { C: 40, w: 10, m: 5 }, st = { kind: 'dense', b: 6, seed: 3 };
    const kinds = seg([['dense', 'dense'], ['window', 'sliding window'], ['sparse', 'sparse (DSA)']], st.kind, (v) => { st.kind = v; draw(); });
    const sb = slider('canvas queries <i>b</i> (toy)', [2, 3, 4, 5, 6, 7, 8], st.b - 2, (v) => String(v));
    sb.input.addEventListener('input', () => { st.b = sb.value; draw(); });
    const again = el('<button type="button" class="fig-btn">New selections</button>');
    again.addEventListener('click', () => { st.seed += 1; draw(); });
    const ctl = el('<div class="fig-controls"></div>'), tools = el('<div class="seg"></div>');
    tools.appendChild(again);
    ctl.append(segLabel('layer', kinds), sb.lab, tools);
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    function draw() {
      const c = colors(), W = widthOf(pic), { C, w, m } = TOY, { kind, b } = st, sparse = kind === 'sparse';
      again.hidden = !sparse;
      const narrow = W < 520, x0 = narrow ? 6 : 140, cw = Math.floor((W - x0 - 6) / C), ch = 13, sh = 8;
      const rand = rng(st.seed * 104729 + 5);
      const sel = Array.from({ length: 8 }, () => pick(rand, C, m)).slice(0, b); // the same first selections whatever b
      const reads = (q) => (kind === 'dense' ? () => true : kind === 'window' ? (s) => s >= C - w : (s) => sel[q].has(s));
      const block = (y, name, cls, col, nq, title2) => {
        let g = '', yy = y;
        if (narrow) { g += `<text x="${x0}" y="${yy + 10}" class="${cls}">${name}: ${title2}</text>`; yy += 16; }
        else g += `<text x="${x0 - 8}" y="${yy + 10}" text-anchor="end" class="${cls}">${name}, ${title2}</text>`;
        const union = new Set();
        for (let q = 0; q < nq; q++) {
          const f = reads(q);
          for (let s2 = 0; s2 < C; s2++) {
            const on = f(s2); if (on) union.add(s2);
            g += `<rect x="${x0 + s2 * cw + 0.5}" y="${yy + q * ch + 0.5}" width="${cw - 1}" height="${ch - 1}" fill="${on ? col : sparse ? col : 'none'}" fill-opacity="${on ? 1 : sparse ? 0.16 : 0}" stroke="${c.muted}" stroke-opacity="0.35" stroke-width="0.6"/>`;
          }
        }
        yy += nq * ch + 6;
        if (sparse) {
          g += narrow ? '' : `<text x="${x0 - 8}" y="${yy + 7}" text-anchor="end">index keys read</text>`;
          for (let s2 = 0; s2 < C; s2++) g += `<rect x="${x0 + s2 * cw + 0.5}" y="${yy}" width="${cw - 1}" height="${sh - 3}" fill="${col}" fill-opacity="0.35"/>`;
          yy += sh + 2;
        }
        g += narrow ? '' : `<text x="${x0 - 8}" y="${yy + 8}" text-anchor="end">keys, values read</text>`;
        for (let s2 = 0; s2 < C; s2++) g += `<rect x="${x0 + s2 * cw + 0.5}" y="${yy}" width="${cw - 1}" height="${sh}" fill="${union.has(s2) ? c.ink : 'none'}" fill-opacity="0.8" stroke="${c.muted}" stroke-opacity="0.35" stroke-width="0.6"/>`;
        yy += sh;
        if (narrow) { g += `<text x="${x0}" y="${yy + 13}">${sparse ? 'bars: index keys, then keys and values' : 'bar: keys and values read'}</text>`; yy += 16; }
        return { g, bottom: yy, cells: nq * (kind === 'dense' ? C : kind === 'window' ? w : m), cols: union.size };
      };
      let s = `<text x="${x0}" y="12">${narrow ? '' : 'the '}${C} cached tokens of one sequence${narrow ? '' : ', oldest first'}</text>`;
      const a = block(22, 'AR step', 'ar', c.ar, 1, '1 query');
      const t = block(a.bottom + 20, 'TD pass', 'td', c.td, b, `${b} queries`);
      const H = t.bottom + 8;
      pic.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cached tokens read by an AR step and by a TD pass in a ${kind} layer: ${a.cols} and ${t.cols} of ${C}">${s}${a.g}${t.g}</svg>`;
      const expU = C * (1 - (1 - m / C) ** b);
      const law = kind === 'dense'
        ? `In a dense layer every query scores every cached token. The canvas scores ${b} times as many pairs and reads the same ${C} tokens, once: ${hsub('Π', 'attn')} = 1, and its intensity is <b>${b} times</b> that of an AR step (<i>b</i> times in general).`
        : kind === 'window'
          ? `In a sliding-window layer every query scores the last <i>w</i> = ${w} cached tokens. The canvas reads the same ${w} tokens, once: ${hsub('Π', 'attn')} = 1, and its intensity is <b>${b} times</b> that of an AR step. (Drawn as Table 1 counts it: the window of a later canvas position starts later and ends inside the canvas, and reads no other cached token.)`
          : `In a sparse layer every query scores the index keys of all ${C} cached tokens (pale cells) and attends to its own <i>m</i> = ${m} selected tokens (full cells). The canvas reads the index once but the union of its ${b} selections, ${t.cols} tokens in this draw and <i>U</i><sub>C,m</sub>(${b}) = ${expU.toFixed(2)} in expectation, where an AR step reads ${m}: ${hsub('Π', 'attn')} &gt; 1, and its intensity is <b>less than ${b} times</b> that of an AR step.`;
      out.innerHTML = `<span class="ar">AR step</span>: ${a.cells} scored pairs${sparse ? ' on the selected tokens' : ''}, ${a.cols} tokens' keys and values read. <span class="td">TD pass</span>: ${t.cells} pairs, ${t.cols} tokens read.<br>${law}`;
    }
    draw(); onTheme(draw); watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- the penalty of a sparse layer (Proposition 6)
  // A model whose attention is one sparse layer type, as in Table 2: constants of DiffusionGemma's global layers.
  // The shaded contexts are those where Corollary 3 (attention dominates both passes) gives a gain below 1.
  const SP = bare(M.MODELS.dgemma_sparse.build());
  const CSP = Array.from({ length: 73 }, (_, i) => Math.round(1024 * 2 ** (i / 4))); // 1k to 268M
  const CFINE = Array.from({ length: 289 }, (_, i) => 1024 * 2 ** (i / 16)); // the same range, for the shading
  const MSP = [256, 512, 1024, 2048, 4096, 8192, 16384], BSP = [8, 16, 32, 64, 128, 256, 512, 1024];
  function figSparse(root) {
    const st = { i: CSP.indexOf(1 << 17) };
    const sm = slider('selected tokens per query <i>m</i>', MSP, MSP.indexOf(SP.msel), comma);
    const sb = slider('canvas length <i>b</i>', BSP, BSP.indexOf(256), comma);
    sm.input.addEventListener('input', () => draw());
    sb.input.addEventListener('input', () => draw());
    const ctl = el('<div class="fig-controls"></div>');
    ctl.append(sm.lab, sb.lab);
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    const Is = SP.Istar, om = OM;
    draggable(pic, () => st.xs, CSP, () => st.i, (i) => { st.i = i; draw(); });
    function draw() {
      const m = sm.value, b = sb.value, tpf = b / om;
      const Un = (C) => (C <= m ? C : C * (1 - (1 - m / C) ** b));
      const qAR = (C) => SP.nuI * C + SP.nuSp * Math.min(C, m), qTD = (C) => SP.nuI * C + SP.nuSp * Un(C);
      const IAR = (C) => (SP.aI * C + SP.aSp * Math.min(C, m)) / qAR(C);
      const Pi = (C) => qTD(C) / qAR(C);
      // Corollary 3: Γ = min(TPF/Π_attn, I*/(ω I^AR_attn))
      const gain = (C) => Math.min(tpf / Pi(C), Is / (om * IAR(C)));
      const c = colors(), W = widthOf(pic), H = 334, mg = { l: 46, r: 18, t: 30, b: 42 }, C = CSP[st.i];
      const xs = scale(1024, 2 ** 28, mg.l, W - mg.r), ys = scale(0.4, 100, H - mg.b, mg.t);
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Attention penalty of a sparse layer against context, m = ${m} selected tokens, canvas of ${b}; shaded where the gain is below 1">`;
      // runs of contexts with Γ < 1, bounded half a grid step either side of the sampled points
      const runs = [], half = 2 ** (1 / 32);
      CFINE.forEach((x, k) => {
        if (!(gain(x) < 1)) return;
        const lo = k === 0 ? x : x / half, hi = k === CFINE.length - 1 ? x : x * half, last = runs[runs.length - 1];
        if (last && last.k === k - 1) { last.hi = hi; last.k = k; } else runs.push({ lo, hi, k });
      });
      for (const r of runs) s += `<rect x="${xs(r.lo)}" y="${mg.t}" width="${xs(r.hi) - xs(r.lo)}" height="${H - mg.b - mg.t}" fill="${c.bad}" fill-opacity="0.12"/>`;
      for (const v of [1, 10, 100]) s += `<line x1="${mg.l}" x2="${W - mg.r}" y1="${ys(v)}" y2="${ys(v)}" stroke="${c.rule}"/><text x="${mg.l - 6}" y="${ys(v) + 4}" text-anchor="end">${v}</text>`;
      for (const v of [1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26]) s += `<text x="${xs(v)}" y="${H - mg.b + 15}" text-anchor="middle">${kctx(v)}</text>`;
      s += `<text x="${(mg.l + W - mg.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">${W < 480 ? 'context C, tokens (log scale)' : 'context C, tokens in the KV cache (log scale)'}</text>`;
      s += `<text x="4" y="13" class="ink">attention penalty (log scale)</text>`;
      for (const [v, t, anchor] of [[m, 'm', 'end'], [b * m, 'bm', 'start']]) s += `<line x1="${xs(v)}" x2="${xs(v)}" y1="${mg.t}" y2="${H - mg.b}" stroke="${c.muted}" stroke-dasharray="3 4"/><text class="ink halo" x="${xs(v) + (anchor === 'end' ? -4 : 4)}" y="${H - mg.b - 6}" text-anchor="${anchor}">C = ${t}</text>`;
      if (tpf >= 0.4) s += `<line x1="${mg.l}" x2="${W - mg.r}" y1="${ys(tpf)}" y2="${ys(tpf)}" stroke="${c.muted}" stroke-dasharray="3 4"/><text class="halo" x="${W - mg.r - 4}" y="${ys(tpf) - 5}" text-anchor="end">TPF = b/ω = ${fmt(tpf)}</text>`;
      if (runs.length) {
        const wide = runs.reduce((a, r) => (xs(r.hi) - xs(r.lo) > xs(a.hi) - xs(a.lo) ? r : a));
        const xm = Math.min(Math.max((xs(wide.lo) + xs(wide.hi)) / 2, mg.l + 24), W - mg.r - 24);
        s += `<text class="halo" x="${xm}" y="${mg.t + 26}" text-anchor="middle" fill="${c.bad}" style="fill:${c.bad}">Γ &lt; 1</text>`;
      }
      s += `<polyline fill="none" stroke="${c.td}" stroke-width="2.4" points="${CSP.map((x) => `${xs(x)},${ys(Pi(x))}`).join(' ')}"/>`;
      const peak = CSP.reduce((best, x) => (Pi(x) > Pi(best) ? x : best), CSP[0]);
      const lx = Math.min(xs(peak) + 10, W - mg.r - (W < 480 ? 40 : 220));
      s += `<text class="td halo" x="${lx}" y="${ys(Pi(peak)) - 9}">${sub('Π', 'attn')}${W < 480 ? '' : ": with the scan's index cache"}</text>`;
      s += `<g class="drag" tabindex="0" role="slider" aria-label="Context" aria-valuemin="1024" aria-valuemax="${2 ** 28}" aria-valuenow="${C}" aria-valuetext="context ${kctx(C)} tokens"><rect x="${mg.l}" y="${mg.t}" width="${W - mg.r - mg.l}" height="${H - mg.b - mg.t}" fill="transparent"/>${marker(xs, C, mg.t, H - mg.b, c)}`;
      s += `<circle cx="${xs(C)}" cy="${ys(Pi(C))}" r="4.5" fill="${c.td}" stroke="${c.bg}" stroke-width="1.5"/>`;
      pic.innerHTML = `${s}</g></svg>`;
      st.xs = xs;
      const P = Pi(C), ia = IAR(C), mem = tpf / P <= Is / (om * ia), G = gain(C);
      const tok = (x) => (x < 1024 ? comma(x) : kctx(x));
      out.innerHTML = `At <b>C = ${kctx(C)}</b>, a canvas of ${comma(b)} queries reads <i>U</i><sub>C,m</sub>(${b}) = ${tok(Un(C))} selected tokens where one query reads ${tok(Math.min(C, m))}, and with the index cache <b>${hsub('Π', 'attn')} = ${gx(P)}</b>.<br>
         If attention dominated both decoders, Corollary 3 gives Γ = ${mem ? `TPF/${hsub('Π', 'attn')} = ${fmt(tpf)}/${gx(P)}` : `${HIS}/(ω ${IATT('AR')}) = ${fmt(Is)}/(${om} × ${gx(ia)})`} = <b>${gx(G)}</b>${G < 1 ? ': TD is slower than AR at this context (shaded)' : ''}.`;
    }
    draw(); onTheme(draw); watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- expandable TL;DR cards
  // Each card with a .card-more panel opens in place, across the full row.
  function initCards() {
    document.querySelectorAll('.bdr .cards').forEach((grid) => {
      const cards = [...grid.querySelectorAll('.bdr-card')].filter((c) => c.querySelector('.card-more'));
      if (!cards.length) return;
      const setOpen = (card, on) => {
        card.classList.toggle('open', on);
        card.querySelector('.card-more').hidden = !on;
        card.querySelector('.card-toggle').setAttribute('aria-expanded', String(on));
        const t = card.querySelector('.card-toggle');
        t.textContent = on ? 'less −' : 'more +';
        t.setAttribute('aria-label', `${on ? 'less' : 'more'} on ${t.dataset.subject}`);
      };
      for (const card of cards) {
        card.classList.add('has-more');
        const k = card.querySelector('.card-k'), subject = (k ? k.textContent.split(',')[0].trim() : 'this card').toLowerCase();
        const t = el(`<button type="button" class="card-toggle" aria-expanded="false" data-subject="${subject}" aria-label="more on ${subject}">more +</button>`);
        card.insertBefore(t, card.querySelector('.card-more'));
        card.addEventListener('click', (ev) => {
          if (ev.target.closest('.card-more') || ev.target.closest('a, input, select')) return;
          setOpen(card, !card.classList.contains('open'));
        });
      }
      const tools = el('<div class="cards-tools"><button type="button" class="card-toggle">open all</button></div>');
      const all = tools.querySelector('button');
      all.addEventListener('click', () => { const on = !cards.every((c) => c.classList.contains('open')); cards.forEach((c) => setOpen(c, on)); all.textContent = on ? 'close all' : 'open all'; });
      grid.parentNode.insertBefore(tools, grid);
    });
  }
  initCards();

  // Figure 14 has one title per tab (bin/build-roofline.py writes all of them): show the selected tab's.
  document.querySelectorAll('.bdr .tabs-title').forEach((title) => {
    const spans = title.querySelectorAll('[data-tab-title]'), tabs = title.nextElementSibling;
    if (!spans.length || !tabs || !tabs.classList.contains('tabs')) return;
    const sync = () => {
      const on = tabs.querySelector('[aria-selected="true"]'), id = on ? on.dataset.tab : spans[0].dataset.tabTitle;
      spans.forEach((sp) => { sp.hidden = sp.dataset.tabTitle !== id; });
    };
    new MutationObserver(sync).observe(tabs, { subtree: true, attributes: true, attributeFilter: ['aria-selected'] });
    sync();
  });


  // ---------------------------------------------------------------- the logit traffic λ of the exact accounting
  // One output position: each arrow moves a vector of V numbers between a kernel and HBM. AR: the head writes the
  // logits, the sampler reads them (λ = 2). DiffusionGemma: the next pass's self-conditioning reads the logits again,
  // writes their softmax and reads it back to form the soft embedding (λ = 5).
  function figLogits(root) {
    const V = DG.V, MB = (V * 4) / 1e6;
    const lE = '<tspan font-style="italic">E</tspan><tspan baseline-shift="super" font-size="0.75em">⊤</tspan>';
    const K = {
      head: { name: 'head', f: `ℓ = <tspan font-style="italic">h</tspan>${lE}` },
      samp: { name: 'sampler', f: 'draws from ℓ', fn: 'x ∼ ℓ' },
      soft: { name: 'softmax', f: 'p = softmax(ℓ)', fn: 'p ∝ exp ℓ' },
      emb: { name: 'soft embedding', short: 'embed', f: `<tspan font-style="italic">e</tspan> = p<tspan font-style="italic">E</tspan>` },
    };
    const MODES = {
      ar: { col: 'ar', who: 'AR step', groups: [['one step', ['head', 'samp']]],
        moves: [['head', 'w', 'ℓ', 'The head applies the vocabulary matrix to the last hidden state and writes the logits ℓ of the position to HBM.'],
                ['samp', 'r', 'ℓ', 'The sampler reads ℓ and draws the next token.']] },
      td0: { col: 'td', who: 'denoising pass, no self-conditioning', groups: [['pass j', ['head', 'samp']]],
        moves: [['head', 'w', 'ℓ', 'The head writes the logits ℓ of the canvas position to HBM.'],
                ['samp', 'r', 'ℓ', 'The sampler reads ℓ and updates the token at this position.']] },
      td: { col: 'td', who: 'denoising pass with self-conditioning', groups: [['pass j', ['head', 'samp']], ['pass j + 1', ['soft', 'emb']]],
        moves: [['head', 'w', 'ℓ', 'The head writes the logits ℓ of the canvas position to HBM.'],
                ['samp', 'r', 'ℓ', 'The sampler reads ℓ and updates the token at this position.'],
                ['soft', 'r', 'ℓ', 'Self-conditioning: the next pass reads ℓ again to take its softmax.'],
                ['soft', 'w', 'p', 'It writes the probabilities p, again V numbers, to HBM.'],
                ['emb', 'r', 'p', 'The soft embedding reads p and multiplies it by the embedding matrix, which gives the d numbers added to the input of the next pass.']] },
    };
    const st = { mode: 'td', t: 0, timer: 0 };
    const btns = el('<div class="seg" role="group" aria-label="animation"><button type="button" class="fig-btn primary" data-a="play">Play</button><button type="button" class="fig-btn" data-a="step">Step</button><button type="button" class="fig-btn" data-a="reset">Reset</button></div>');
    const modes = segLabel('decoder', seg([['ar', 'AR'], ['td0', 'TD, no self-conditioning'], ['td', 'TD (DiffusionGemma)']], st.mode, (v) => { st.mode = v; reset(); }));
    const ctl = el('<div class="fig-controls"></div>');
    ctl.append(modes, btns);
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    const n = () => MODES[st.mode].moves.length;
    const stop = () => { clearInterval(st.timer); st.timer = 0; btns.querySelector('[data-a=play]').textContent = st.t >= n() ? 'Replay' : 'Play'; };
    const step = () => { if (st.t < n()) st.t += 1; draw(); if (st.t >= n()) stop(); };
    function reset() { stop(); st.t = 0; btns.querySelector('[data-a=play]').textContent = 'Play'; draw(); }
    btns.addEventListener('click', (ev) => {
      const a = ev.target.dataset && ev.target.dataset.a; if (!a) return;
      if (a === 'reset') reset();
      if (a === 'step') { stop(); step(); }
      if (a === 'play') {
        if (st.timer) { stop(); return; }
        if (st.t >= n()) st.t = 0;
        const dt = matchMedia('(prefers-reduced-motion: reduce)').matches ? 1600 : 1100;
        st.timer = setInterval(step, dt); ev.target.textContent = 'Pause'; step();
      }
    });
    function draw() {
      const c = colors(), W = widthOf(pic), md = MODES[st.mode], col = c[md.col], t = st.t;
      const narrow = W < 520, x0 = 6, x1 = W - 6;
      const kernels = md.groups.flatMap(([, ks]) => ks);
      const nk = 4, gap = narrow ? 8 : 16, bw = (x1 - x0 - (nk - 1) * gap) / nk; // always four slots, so the boxes keep their size
      const off = (W - (kernels.length * bw + (kernels.length - 1) * gap)) / 2;
      const kx = (i) => off + i * (bw + gap);
      const yG = 14, yK = 30, hK = 50, yM = yK + hK, yH = 160, hH = 34;
      let s = '';
      // pass brackets
      let i0 = 0;
      for (const [label, ks] of md.groups) {
        const a = kx(i0), b = kx(i0 + ks.length - 1) + bw;
        s += `<text x="${(a + b) / 2}" y="${yG}" text-anchor="middle">${label}</text>`;
        s += `<path d="M${a} ${yG + 9}V${yG + 5}H${b}V${yG + 9}" fill="none" stroke="${c.muted}" stroke-width="1"/>`;
        i0 += ks.length;
      }
      // kernels
      kernels.forEach((k, i) => {
        const x = kx(i), used = md.moves.some((m, j) => m[0] === k && j < t);
        s += `<rect x="${x}" y="${yK}" width="${bw}" height="${hK}" rx="5" fill="${c.surface}" stroke="${used ? col : c.muted}" stroke-width="${used ? 1.6 : 1}"/>`;
        s += `<text x="${x + bw / 2}" y="${yK + 20}" text-anchor="middle" class="ink" font-weight="600">${narrow && K[k].short ? K[k].short : K[k].name}</text>`;
        s += `<text x="${x + bw / 2}" y="${yK + 38}" text-anchor="middle">${narrow && K[k].fn ? K[k].fn : K[k].f}</text>`;
      });
      // HBM
      s += `<rect x="${x0}" y="${yH}" width="${x1 - x0}" height="${hH}" rx="4" fill="${c.soft}" stroke="${c.muted}" stroke-width="1"/>`;
      s += `<text x="${x0 + 8}" y="${yH + hH - 9}" class="ink" font-weight="600">HBM</text>`;
      // moves: one vertical arrow per move, under its kernel (two moves of one kernel side by side)
      md.moves.forEach(([k, dir, ten], j) => {
        const i = kernels.indexOf(k), same = md.moves.filter((m) => m[0] === k), r = same.indexOf(md.moves[j]);
        const x = kx(i) + bw * (same.length === 1 ? 0.5 : r === 0 ? 0.3 : 0.7);
        const on = j < t, cur = j === t - 1, stroke = on ? col : c.muted;
        const ya = yM + 2, yb = yH - 2, tip = dir === 'w' ? yb : ya, sgn = dir === 'w' ? -1 : 1;
        s += `<line x1="${x}" y1="${ya}" x2="${x}" y2="${yb}" stroke="${stroke}" stroke-width="${cur ? 2.6 : on ? 1.8 : 1}" ${on ? '' : 'stroke-dasharray="3 3"'} opacity="${on ? 1 : 0.6}"/>`;
        s += `<path d="M${x - 5} ${tip + sgn * 8}L${x} ${tip}L${x + 5} ${tip + sgn * 8}Z" fill="${stroke}" opacity="${on ? 1 : 0.6}"/>`;
        s += `<text x="${x + 6}" y="${(ya + yb) / 2 + 4}" font-weight="${cur ? 600 : 400}" ${on ? `style="fill:${col}"` : ''}>${j + 1}</text>`;
        s += `<text x="${x}" y="${yH + 14}" text-anchor="middle" font-style="italic" ${on ? 'class="ink"' : ''}>${ten}</text>`;
      });
      const H = yH + hH + 4;
      pic.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${md.who}: ${t} of ${n()} moves of a vocabulary-sized vector between a kernel and HBM">${s}</svg>`;
      const lam = `<i>λ</i><sup>${md.col === 'ar' ? 'AR' : 'TD'}</sup>`;
      const per = `${fmt(n() * MB)} MB`;
      const tail = st.mode === 'ar'
        ? `${lam} = <b>2</b>: ${per} per generated token.`
        : st.mode === 'td0'
          ? `${lam} = <b>2</b> for a TD twin without self-conditioning: ${per} per position and pass.`
          : `${lam} = <b>5</b>: ${per} per position and pass, i.e., ${fmt((n() * MB * W0.b) / 1e3)} GB per sequence for a canvas of ${W0.b} positions, about as much as the ${fmt((DG.V * DG.d * 2) / 1e9)} GB of the vocabulary matrix that the pass reads once for the whole batch.`;
      out.innerHTML = t === 0
        ? `<span class="${md.col}">${md.who[0].toUpperCase() + md.who.slice(1)}</span>, one output position. Each arrow moves <i>V</i> = ${comma(V)} numbers, ${fmt(MB)} MB in FP32: down into HBM is a write, up out of it a read. Press Play or Step.`
        : `<b>Move ${t} of ${n()}, ${md.moves[t - 1][1] === 'w' ? 'write' : 'read'} <i>${md.moves[t - 1][2]}</i>.</b> ${md.moves[t - 1][3]}${t === n() ? `<br>${tail}` : ''}`;
    }
    reset(); onTheme(draw); watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- mount
  const FIGS = {
    reuse: figReuse, roofline: figRoofline, context: figContext, accel: figAccel,
    weights: figWeights, linmap: figLinMap, attnread: figAttnRead, sparse: figSparse, logits: figLogits,
  };
  document.querySelectorAll('[data-fig]').forEach((root) => {
    try { FIGS[root.dataset.fig](root); } catch (e) { root.textContent = 'This figure failed to load.'; console.error(e); }
  });
  const redraw = () => rerenders.forEach((f) => f());
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
