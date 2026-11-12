// Interactive figures for the Block Diffusion Rooflines page. Every number comes from the note's own
// cost model (window.BDModel) and helpers (window.BDR, exposed by note.js).
//
// Accountings (see the table built by figAccounting):
//   - intensities are always those of Proposition 1 (Definition 3, no vocabulary): prop1();
//   - gains labelled "exact" use the full accounting at a fixed context: steady();
//   - regimes quoted next to an exact gain are stated as shares of the peak compute (util()),
//     never as Proposition-1 intensities, so the two accountings are not mixed in one statement.
(function () {
  'use strict';
  const M = window.BDModel, X = window.BDR;
  if (!M || !X) return;
  const { HWBYID, REF, omega, fmt, css, comma } = X;

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
  const sub = (base, s) => `${base}<tspan baseline-shift="sub" font-size="8.5">${s}</tspan>`;
  const hsub = (base, s) => { const t = `<i>${base}</i><sub>${s}</sub>`; return s === 'AR' || s === 'TD' ? `<span class="q-${s.toLowerCase()}">${t}</span>` : t; };
  const colors = () => ({
    ar: css('--bdr-ar'), td: css('--bdr-td'), ink: css('--ink'), muted: css('--bdr-muted'), rule: css('--rule'),
    bg: css('--bdr-bg'), surface: css('--bdr-surface'), accent: css('--accent'), soft: css('--bdr-accent-soft'),
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
  function seg(options, current, onPick) {
    const box = el('<div class="seg" role="group"></div>');
    for (const [value, text] of options) {
      const b = el(`<button type="button" aria-pressed="${value === current}">${text}</button>`);
      b.addEventListener('click', () => { box.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); onPick(value); });
      box.appendChild(b);
    }
    return box;
  }
  // Drag a vertical marker along a log axis (pointer, touch, arrow keys). The handlers sit on the
  // container, which survives redraws; the SVG and its .drag group are rebuilt on every move.
  function draggable(box, getXs, values, getIndex, setIndex) {
    let on = false;
    const move = (ev) => { const p = svgPoint(box.querySelector('svg'), ev); setIndex(nearestIndex(values, getXs().inv(p.x))); };
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
  // SVGs are drawn at the width they are shown at, so that 11px text stays 11px on a phone.
  const widthOf = (box, lo = 330, hi = 1000) => Math.max(lo, Math.min(hi, Math.round(box.clientWidth || 640)));
  function watchWidth(box, draw) {
    let last = box.clientWidth;
    new ResizeObserver(() => { const w = box.clientWidth; if (Math.abs(w - last) > 4) { last = w; draw(); } }).observe(box);
  }
  const rerenders = [];
  const onTheme = (f) => rerenders.push(f);

  // The accelerator is shared by Figures 2, 3, 4 and 6: choosing one in a figure sets it in all.
  const ACCEL = ['h20', 'rtx5090', 'a100', 'h200', 'v5e', 'b200', 'mi355x', 'h100pcie', 'spark', 'v6e'];
  const SHORT = { h20: 'H20', rtx5090: 'RTX 5090', a100: 'A100', h200: 'H200', v5e: 'TPU v5e', b200: 'B200', mi355x: 'MI355X', h100pcie: 'H100 PCIe', spark: 'DGX Spark', v6e: 'TPU v6e' };
  const shared = { hw: 'h200', subs: [] };
  function setHw(id) { if (!HWBYID[id] || id === shared.hw) return; shared.hw = id; shared.subs.forEach((f) => f(id)); }
  function hwControl() {
    const lab = el(`<label class="sel">accelerator<select>${ACCEL.map((id) => `<option value="${id}"${id === shared.hw ? ' selected' : ''}>${HWBYID[id].label}</option>`).join('')}</select></label>`);
    const s = lab.querySelector('select');
    s.addEventListener('change', () => setHw(s.value));
    shared.subs.push((id) => { s.value = id; });
    return lab;
  }
  const BATCHES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
  const CONTEXTS = Array.from({ length: 11 }, (_, i) => 1024 * 2 ** i);

  // ---------------------------------------------------------------- the three accountings
  function figAccounting(root) {
    const hw = hwOf('h200');
    const rows = [
      { name: 'Proposition 1', what: 'closed form of §2: no vocabulary, context fixed at 1k (at zero context, as in the closed forms of §3 and Table 8, TD is ahead up to batch 109)', used: 'Figures 2 and 4, the rooflines of the quiz, Table 3', g: (B) => prop1(DG, hw, B, 1024).gain },
      { name: 'Exact, fixed context', what: 'every FLOP and byte of the appendix, vocabulary included, context fixed at 1k', used: 'Figures 3 and 6, the quiz verdicts, the maps', g: (B) => steady(DG, hw, B, 1024).gain },
      { name: 'Exact, whole generation', what: 'the same accounting summed over a 1k prompt and 8k generated tokens: the context grows from 1k to 9k', used: 'the headline tiles above, the calculator', g: (B) => M.generate(DG, { ...W0, B }, O0, hw).gainDecode, slow: true },
    ];
    const table = el(`<div class="table-wrap nofold"><table class="acct"><thead><tr><th class="l">Accounting</th><th>Γ, batch 1</th><th>Γ, batch 5</th><th>Γ, batch 64</th><th>TD ahead up to batch</th></tr></thead><tbody>${rows.map((r) => `<tr><td class="l"><b>${r.name}</b><br><span class="muted">${r.what}. Used by ${r.used}.</span></td><td class="num">…</td><td class="num">…</td><td class="num">…</td><td class="num">…</td></tr>`).join('')}</tbody></table></div>`);
    root.appendChild(table);
    const tick = () => new Promise((res) => setTimeout(res, 0));
    rows.forEach(async (r, k) => {
      const cells = table.querySelectorAll('tbody tr')[k].querySelectorAll('td.num');
      const g = async (B) => { if (r.slow) await tick(); return r.g(B); };
      for (const [j, B] of [1, 5, 64].entries()) cells[j].textContent = g2(await g(B));
      let lo = 64, hi = 512; // the gain decreases with the batch on this range
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if ((await g(mid)) >= 1) lo = mid; else hi = mid; }
      cells[3].textContent = String(lo);
    });
  }

  // ---------------------------------------------------------------- roofline renderer (Figure 2, quiz)
  // One accelerator's roofline with an AR dot and a TD dot at their intensities; the bracket on the
  // right is the ratio of their attainable FLOP rates, min(I_TD, I*) / min(I_AR, I*).
  function rooflineSvg({ hw, IAR, ITD, W = 640, H = 300, mini = false }) {
    const c = colors(), Is = ridge(hw), pk = hw.peak / 1e12, rate = (I) => Math.min(pk, (I * hw.bw) / 1e12);
    const m = { l: mini ? 40 : 54, r: mini ? 58 : 92, t: 14, b: mini ? 30 : 40 };
    const xlo = 0.3, xhi = 3e4, xs = scale(xlo, xhi, m.l, W - m.r);
    const ylo = rate(xlo) / 1.6, yhi = pk * 1.9, ys = scale(ylo, yhi, H - m.b, m.t);
    const ra = rate(IAR), rt = rate(ITD), cx = (I) => xs(Math.min(Math.max(I, xlo), xhi));
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Roofline: AR at ${fmt(ra)} TFLOP/s, TD at ${fmt(rt)} TFLOP/s">`;
    s += `<defs><marker id="rf-ah${mini ? 'm' : ''}" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${c.ink}"/></marker></defs>`;
    s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => (v >= 1 ? fmt(v) : v.toPrecision(1)), c, [1]);
    for (let e = 0; e <= 4; e++) s += `<text x="${xs(10 ** e)}" y="${H - m.b + 15}" text-anchor="middle">${10 ** e >= 1000 ? `${10 ** (e - 3)}k` : 10 ** e}</text>`;
    if (!mini) {
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">arithmetic intensity, FLOP per byte</text>`;
      s += `<text x="12" y="${(m.t + H - m.b) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(m.t + H - m.b) / 2})" class="ink">TFLOP/s</text>`;
    }
    s += `<line x1="${xs(Is)}" x2="${xs(Is)}" y1="${ys(pk)}" y2="${H - m.b}" stroke="${c.muted}" stroke-dasharray="3 4"/>`;
    s += `<text x="${xs(Is) + 5}" y="${H - m.b - 6}">${mini ? 'I* ' : 'ridge I* = '}${fmt(Is)}</text>`;
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

  // ---------------------------------------------------------------- Figure 1: byte reuse and the MoE penalty
  function figReuse(root) {
    const st = { b: 64, S: 14, t: 0, timer: 0, model: 'dense' };
    const ctl = el('<div class="fig-controls"></div>');
    const mSeg = el('<label class="wide"><span>model</span><span class="segbox"></span></label>');
    mSeg.querySelector('.segbox').appendChild(seg([['dense', 'dense'], ['moe', 'MoE: DiffusionGemma']], st.model, (v) => { st.model = v; reset(); }));
    const bSeg = el('<label class="wide"><span>canvas length <i>b</i></span><span class="segbox"></span></label>');
    bSeg.querySelector('.segbox').appendChild(seg([16, 32, 64, 128, 256].map((v) => [v, String(v)]), st.b, (v) => { st.b = v; reset(); }));
    const sS = slider('denoising passes <i>S</i>', Array.from({ length: 32 }, (_, i) => i + 1), st.S - 1, (v) => `${v}  (ω = ${v + 1})`);
    sS.input.addEventListener('input', () => { st.S = sS.value; reset(); });
    const btns = el('<div class="seg"><button type="button" class="fig-btn primary" data-a="play">Play</button><button type="button" class="fig-btn" data-a="step">Step</button><button type="button" class="fig-btn" data-a="reset">Reset</button></div>');
    ctl.append(mSeg, bSeg, sS.lab, btns);
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    const om = () => st.S + 1;
    // Weights a TD pass reads over those an AR step reads, at batch 1 (Proposition 3); 1 for a dense model.
    const penalty = () => (st.model === 'moe' ? prop1(DG, hwOf('h200'), 1, 0, { b: st.b, S: st.S }).Pi : 1);
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
      const c = colors(), W = widthOf(pic), x0 = W < 480 ? 118 : 150, x1 = W - 16, w = x1 - x0, n = om(), t = st.t, b = st.b, Pi = penalty();
      const cell = w / Math.max(b, n), committed = t >= n;
      let s = `<svg viewBox="0 0 ${W} 240" role="img" aria-label="After ${t} of ${n} passes: AR has ${t} tokens, TD ${committed ? b : 0}">`;
      // shared time axis: one slot per pass
      s += `<text x="0" y="27" class="ink">forward passes</text><text x="0" y="41">one weight read each</text>`;
      const sw = w / n;
      for (let j = 0; j < n; j++) {
        const done = j < t, last = j === n - 1;
        s += `<rect x="${x0 + j * sw + 1}" y="18" width="${Math.max(2, sw - 2)}" height="22" rx="2" fill="${done ? c.ink : 'none'}" fill-opacity="${done ? 0.82 : 0}" stroke="${c.muted}" stroke-opacity="0.6"/>`;
        if (sw > 22) s += `<text x="${x0 + j * sw + sw / 2}" y="33" text-anchor="middle" style="fill:${done ? c.bg : c.muted};font-size:10px">${last ? (sw >= 46 ? 'commit' : '+1') : j + 1}</text>`;
      }
      // bytes read by one pass: the TD bar is Π_lin times the AR bar
      const unit = Math.min(60, w / 8.2);
      s += `<text x="${x0}" y="56">weights read by one pass</text>`;
      s += `<text x="0" y="71" class="ar">AR step: ${sub('W', 'AR')}</text><rect x="${x0}" y="63" width="${unit}" height="8" fill="${c.ar}"/>`;
      s += `<text x="0" y="91" class="td">TD pass: ${Pi === 1 ? '' : `${fmt(Pi)} `}${sub('W', 'AR')}</text><rect x="${x0}" y="83" width="${unit * Pi}" height="8" fill="${c.td}"/>`;
      // AR: one token per pass
      s += `<text x="0" y="126" class="ar">AR</text><text x="0" y="140">1 token per pass</text>`;
      for (let j = 0; j < n; j++) s += `<rect x="${x0 + j * cell}" y="112" width="${Math.max(1, cell - (cell > 4 ? 1.5 : 0.4))}" height="26" fill="${j < t ? c.ar : 'none'}" stroke="${c.ar}" stroke-opacity="${j < t ? 1 : 0.3}" stroke-width="${cell > 4 ? 1 : 0.4}"/>`;
      // TD: a canvas of b positions, all updated by every pass, committed by the last one
      s += `<text x="0" y="184" class="td">TD</text><text x="0" y="198">b positions</text>`;
      const shade = committed ? 1 : Math.min(t, st.S) / st.S * 0.75;
      for (let j = 0; j < b; j++) s += `<rect x="${x0 + j * cell}" y="170" width="${Math.max(1, cell - (cell > 4 ? 1.5 : 0.4))}" height="26" fill="${c.td}" fill-opacity="${committed ? 1 : shade * 0.85}" stroke="${c.td}" stroke-opacity="${committed ? 1 : 0.35}" stroke-width="${cell > 4 ? 1 : 0.4}"/>`;
      if (!committed && t > 0) s += `<text x="${x0}" y="216">denoising: every position refined by pass ${Math.min(t, st.S)} of ${st.S}</text>`;
      if (committed) s += `<text x="${x0}" y="216" class="td">committed: ${b} tokens</text>`;
      s += '</svg>';
      pic.innerHTML = s;
      const tpf = b / n, gain = tpf / Pi;
      const moe = st.model === 'moe' ? ` A canvas of ${b} positions touches ${fmt(M.distinctExperts(DG, b))} of 128 experts per layer and one token touches 8, so a TD pass reads ${hsub('Π', 'lin')} = ${fmt(Pi)} times the weights of an AR step.` : '';
      out.innerHTML = t < n
        ? `After <b>${t}</b> of <b>${n}</b> passes: <span class="ar">AR</span> has committed ${t} token${t === 1 ? '' : 's'}, <span class="td">TD</span> none yet.${moe}`
        : `<span class="big">In ${n} passes, <span class="ar">AR</span> commits ${n} tokens and <span class="td">TD</span> commits ${b}: TPF = ${b}/${n} = <b>${fmt(tpf)}</b>.</span><br>${moe} Per token, TD reads ${Pi === 1 ? `1/TPF = ${fmt(1 / tpf)}` : `${hsub('Π', 'lin')}/TPF = ${fmt(Pi / tpf)}`} times the weight bytes of AR and performs ${n} times its FLOPs. If both passes are memory-bound, <b>Γ = ${Pi === 1 ? `TPF = ${fmt(gain)}` : `TPF/${hsub('Π', 'lin')} = ${fmt(tpf)}/${fmt(Pi)} = ${fmt(gain)}`}</b>${gain < 1 ? ': the canvas is too short to pay for what it reads, and TD loses' : ''}.${Pi === 1 && b > 206 ? ` A dense TD pass over ${b} positions has an intensity of about ${b}, above the ridge of an H200 (206): there it is compute-bound, and the gain is capped at <i>I</i>*/(ω ${hsub('I', 'AR')}) (Corollary 1, Figure 2).` : ''}`;
    }
    reset();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 2: live roofline (Proposition 1)
  function figRoofline(root) {
    const ctl = el('<div class="fig-controls"></div>');
    const sB = slider('batch <i>B</i>', BATCHES, 0, (v) => String(v));
    const sC = slider('context <i>C</i>', CONTEXTS, 0, kctx);
    const presets = seg([['r1', 'batch 1, context 1k'], ['r2', 'batch 8, context 1k'], ['r6', 'batch 4, context 1M']], 'r1', (p) => {
      const [B, C] = { r1: [1, 1024], r2: [8, 1024], r6: [4, 1 << 20] }[p]; sB.set(B); sC.set(C); draw();
    });
    ctl.append(hwControl(), sB.lab, sC.lab, presets);
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    const clearPresets = () => presets.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
    sB.input.addEventListener('input', () => { clearPresets(); draw(); });
    sC.input.addEventListener('input', () => { clearPresets(); draw(); });
    function draw() {
      const hw = hwOf(shared.hw), B = sB.value, C = sC.value, p = prop1(DG, hw, B, C);
      pic.innerHTML = rooflineSvg({ hw, IAR: p.IAR, ITD: p.ITD, W: widthOf(pic), H: widthOf(pic) < 480 ? 260 : 300 });
      const bound = (I) => (I >= p.Is ? 'compute-bound' : 'memory-bound');
      out.innerHTML = `${hsub('I', 'AR')} = ${fmt(p.IAR)} (<span class="ar">AR ${bound(p.IAR)}</span>), ${hsub('I', 'TD')} = ${fmt(p.ITD)} (<span class="td">TD ${bound(p.ITD)}</span>), <i>I</i>* = ${fmt(p.Is)}.<br>`
        + `<span class="big">TD's FLOPs run ×${fmt(p.speed)} faster; a canvas costs ω = ${p.om} passes; <b>Γ = ${fmt(p.speed)}/${p.om} = ${g2(p.gain)}</b> (Proposition 1).</span> Exact accounting at this context: Γ = ${g2(steady(DG, hw, B, C).gain)}.${fits(DG, hw, B, C) ? '' : ` At this batch and context the model and its KV cache do not fit in ${HWBYID[shared.hw].gb} GB.`}`;
    }
    shared.subs.push(draw);
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 3: find the best batch (exact, fixed context)
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
    const calc = document.getElementById('calc'); if (calc) calc.scrollIntoView();
  }
  function figBestBatch(root) {
    const st = { i: nearestIndex(BSCAN, 64), revealed: false, rule: 'whole' };
    const ctl = el('<div class="fig-controls"></div>');
    const rev = el('<div class="seg"><button type="button" class="fig-btn primary">Reveal the curve</button></div>');
    const rule = el('<label class="wide"><span>timing of a pass</span><span class="segbox"></span></label>');
    rule.querySelector('.segbox').appendChild(seg([['whole', 'one overlapped kernel (A1)'], ['sum', 'kernels one after another']], st.rule, (v) => { st.rule = v; draw(); }));
    ctl.append(hwControl(), rule, rev);
    const pic = el('<div></div>'), out = el('<div class="readout"></div>');
    root.append(ctl, pic, out);
    rev.querySelector('button').addEventListener('click', (ev) => { st.revealed = !st.revealed; ev.target.textContent = st.revealed ? 'Hide the curve' : 'Reveal the curve'; ev.target.classList.toggle('primary', !st.revealed); draw(); });
    out.addEventListener('click', (ev) => { if (ev.target.closest('[data-calc]')) { ev.preventDefault(); openCalculator(shared.hw, BSCAN[st.i], st.rule); } });
    function draw() {
      const c = colors(), W = widthOf(pic), H = 300, m = { l: 48, r: 18, t: 18, b: 40 }, sum = st.rule === 'sum';
      const cur = gainCurve(shared.hw, st.rule), pts = cur.pts, B = BSCAN[st.i], P = pts[st.i];
      const xs = scale(1, 1024, m.l, W - m.r);
      const ylo = Math.min(0.25, ...pts.map((p) => p.g)) * 0.9, yhi = Math.max(8, ...pts.map((p) => p.g)) * 1.25, ys = scale(ylo, yhi, H - m.b, m.t);
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Gain against batch">`;
      if (st.revealed && cur.bstar) s += `<rect x="${m.l}" y="${m.t}" width="${xs(cur.bstar) - m.l}" height="${H - m.b - m.t}" fill="${c.soft}" opacity="0.8"/>`;
      s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => `${v}×`, c);
      for (let e = 0; e <= 10; e += 1) s += `<text x="${xs(2 ** e)}" y="${H - m.b + 15}" text-anchor="middle">${2 ** e}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">batch B</text>`;
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(1)}" y2="${ys(1)}" stroke="${c.muted}" stroke-width="1.5"/><text x="${m.l + 6}" y="${ys(1) + 15}">parity, Γ = 1: below it AR is faster</text>`;
      if (st.revealed) {
        if (cur.bstar) {
          s += `<line x1="${xs(cur.bstar)}" x2="${xs(cur.bstar)}" y1="${m.t}" y2="${H - m.b}" stroke="${c.td}" stroke-dasharray="4 3"/>`;
          s += `<text x="${xs(cur.bstar) - 6}" y="${m.t + 14}" text-anchor="end" class="ink">${sum ? 'some TD kernels memory-bound' : 'TD memory-bound'}</text><text x="${xs(cur.bstar) + 6}" y="${m.t + 14}" class="ink">${sum ? 'all compute-bound' : 'TD compute-bound'}</text>`;
          if (!sum) s += `<text x="${xs(cur.bstar) - 6}" y="${H - m.b - 8}" text-anchor="end" class="td halo">${sub('B*', 'TD')} = ${cur.bstar}</text>`;
        }
        s += dashedLine(pts, (p) => `${xs(p.B)},${ys(p.g)}`, (p) => p.fit, `stroke="${c.ink}" stroke-width="2.2" stroke-linejoin="round"`);
        s += `<circle cx="${xs(cur.best.B)}" cy="${ys(cur.best.g)}" r="6" fill="${c.td}" stroke="${c.bg}" stroke-width="2"/>`;
        s += `<text class="strong halo" x="${xs(cur.best.B) + (xs(cur.best.B) < m.l + 190 ? 10 : -10)}" y="${ys(cur.best.g) - 12}" text-anchor="${xs(cur.best.B) < m.l + 190 ? 'start' : 'end'}">best: ${g2(cur.best.g)}× at B = ${cur.best.B}</text>`;
        if (cur.parity) s += `<circle cx="${xs(cur.parity)}" cy="${ys(1)}" r="4.5" fill="${c.bg}" stroke="${c.ink}" stroke-width="2"/><text class="halo" x="${xs(cur.parity) - 8}" y="${ys(1) + 17}" text-anchor="end">${sub('B', '×')} ≈ ${Math.round(cur.parity)}</text>`;
        s += `<circle cx="${xs(B)}" cy="${ys(P.g)}" r="4" fill="${c.accent}"/>`;
      } else {
        s += `<text x="${m.l + 8}" y="${m.t + 14}" class="ink">the curve Γ(B) is hidden</text>`;
      }
      // the marker
      s += `<g class="drag" tabindex="0" role="slider" aria-label="Batch" aria-valuemin="1" aria-valuemax="1024" aria-valuenow="${B}" aria-valuetext="batch ${B}${st.revealed ? `, gain ${g2(P.g)}` : ''}">`;
      s += `<rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${H - m.b - m.t}" fill="transparent"/>`;
      s += `<line x1="${xs(B)}" x2="${xs(B)}" y1="${m.t}" y2="${H - m.b}" stroke="${c.accent}" stroke-width="2"/>`;
      s += `<path d="M${xs(B) - 7},${H - m.b} l7,-11 l7,11 z" fill="${c.accent}"/>`;
      s += `<text x="${xs(B)}" y="${H - m.b - 16}" text-anchor="${xs(B) > W - 120 ? 'end' : 'middle'}" class="halo" style="fill:${c.accent}">${st.revealed ? 'B' : 'your guess: B'} = ${B}</text></g>`;
      pic.innerHTML = `${s}</svg>`;
      st.xs = xs;
      if (!st.revealed) {
        out.innerHTML = `Your guess: <b>B = ${B}</b>. Reveal the curve to compare.`;
      } else {
        const off = B === cur.best.B ? `${g2(P.g)}×, the best` : `${g2(P.g)}× (${pct(P.g / cur.best.g)} of the best)`;
        const lastFit = [...pts].reverse().find((p) => p.fit);
        const turn = sum
          ? `every kernel of a TD pass is compute-bound only from batch ${cur.bstar}, and the maximum falls before it`
          : `a TD pass is compute-bound from ${hsub('B*', 'TD')} = ${cur.bstar}`;
        out.innerHTML = `<span class="big">Best roofline gain: <b>${g2(cur.best.g)}× at B = ${cur.best.B}</b>; ${turn}${cur.parity ? `; AR catches up near ${hsub('B', '×')} ≈ ${Math.round(cur.parity)}` : '; AR does not catch up by batch 1024'}.</span><br>`
          + `${lastFit ? (lastFit.B < 1024 ? `Dashed: batches above ${lastFit.B} do not fit in ${HWBYID[shared.hw].gb} GB. ` : '') : `The model does not fit in ${HWBYID[shared.hw].gb} GB at any batch. `}`
          + `At the marker, B = ${B}: Γ = ${off}; an <span class="ar">AR step uses ${pct(P.uAR)}</span> of the peak compute and a <span class="td">TD pass ${pct(P.uTD)}</span>; MoE penalty ${hsub('Π', 'lin')} = ${fmt(P.Pi)}. <a href="#calc" data-calc>Open this setting in the calculator</a>.`;
      }
    }
    draggable(pic, () => st.xs, BSCAN, () => st.i, (i) => { st.i = i; draw(); });
    shared.subs.push(draw);
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 4: both intensities forget the batch (Proposition 1)
  const CSCAN = Array.from({ length: 41 }, (_, i) => Math.round(1024 * 2 ** (i / 4)));
  const CBATCH = [1, 4, 16, 64];
  function figContext(root) {
    const st = { i: CSCAN.indexOf(1 << 20) };
    const ctl = el('<div class="fig-controls"></div>');
    ctl.append(hwControl());
    const pic = el('<div></div>'), out = el('<div class="readout"></div>');
    root.append(ctl, pic, out);
    // The intensities do not depend on the accelerator; only the ridge and the memory limit do.
    const curves = CBATCH.map((B) => CSCAN.map((C) => { const p = prop1(DG, hwOf('h200'), B, C); return { B, C, IAR: p.IAR, ITD: p.ITD }; }));
    draggable(pic, () => st.xs, CSCAN, () => st.i, (i) => { st.i = i; draw(); });
    function draw() {
      const c = colors(), W = widthOf(pic), H = 330, m = { l: 48, r: 18, t: 16, b: 40 }, C = CSCAN[st.i];
      const hw = hwOf(shared.hw), Is = ridge(hw), ok = (p) => fits(DG, hw, p.B, p.C);
      const xs = scale(1024, 1 << 20, m.l, W - m.r), ylo = 0.7, yhi = 6000, ys = scale(ylo, yhi, H - m.b, m.t);
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Intensities of an AR step and of a TD pass against context, for batches ${CBATCH.join(', ')}">`;
      s += `<rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${ys(Is) - m.t}" fill="${c.soft}" opacity="0.7"/>`;
      s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => (v >= 1000 ? `${v / 1000}k` : v), c, [1]);
      for (const C2 of [1024, 4096, 16384, 65536, 262144, 1 << 20]) s += `<text x="${xs(C2)}" y="${H - m.b + 15}" text-anchor="middle">${kctx(C2)}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">context C, tokens in the KV cache</text>`;
      s += `<text x="12" y="${(m.t + H - m.b) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(m.t + H - m.b) / 2})" class="ink">FLOP per byte</text>`;
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(Is)}" y2="${ys(Is)}" stroke="${c.ink}" stroke-width="1.5"/>`;
      s += `<text class="ink halo" x="${W - m.r - 4}" y="${ys(Is) - 6}" text-anchor="end">ridge I* = ${fmt(Is)}: compute-bound above</text>`;
      const op = [0.4, 0.6, 0.8, 1];
      curves.forEach((pts, k) => {
        s += dashedLine(pts, (p) => `${xs(p.C)},${ys(p.ITD)}`, ok, `stroke="${c.td}" stroke-opacity="${op[k]}" stroke-width="2.2"`);
        s += dashedLine(pts, (p) => `${xs(p.C)},${ys(p.IAR)}`, ok, `stroke="${c.ar}" stroke-opacity="${op[k]}" stroke-width="2.2"`);
        const yl = ys(pts[0].ITD) - 6, onRidge = Math.abs(yl - 4 - ys(Is)) < 9;
        s += `<text class="td halo" x="${xs(1024) + 6}" y="${onRidge ? yl + 20 : yl}" style="opacity:${0.55 + 0.45 * op[k]}">B = ${CBATCH[k]}</text>`;
        s += `<text class="ar halo" x="${xs(1024) + 6}" y="${ys(pts[0].IAR) - 6}" style="opacity:${0.55 + 0.45 * op[k]}">B = ${CBATCH[k]}</text>`;
      });
      const mid = CSCAN[18];
      s += `<text class="td halo" x="${xs(mid)}" y="${ys(curves[3][18].ITD) - 10}" text-anchor="middle">TD pass, ${sub('I', 'TD')}</text>`;
      s += `<text class="ar halo" x="${xs(mid)}" y="${ys(curves[0][18].IAR) + 20}" text-anchor="middle">AR step, ${sub('I', 'AR')}</text>`;
      s += `<g class="drag" tabindex="0" role="slider" aria-label="Context" aria-valuemin="1024" aria-valuemax="${1 << 20}" aria-valuenow="${C}" aria-valuetext="context ${kctx(C)} tokens"><rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${H - m.b - m.t}" fill="transparent"/>`;
      s += `<line x1="${xs(C)}" x2="${xs(C)}" y1="${m.t}" y2="${H - m.b}" stroke="${c.ink}" stroke-opacity="0.55" stroke-width="1.5"/>`;
      s += `<path d="M${xs(C) - 7},${H - m.b} l7,-11 l7,11 z" fill="${c.ink}"/>`;
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
      out.innerHTML = `At <b>C = ${kctx(C)}</b>, across batches ${CBATCH.join(', ')}: ${hsub('I', 'AR')} = ${ar.map((v) => fmt(v)).join(', ')} (a factor ${fmt(Math.max(...ar) / Math.min(...ar))} apart); ${hsub('I', 'TD')} = ${td.map((v) => fmt(v)).join(', ')}, ${above === at.length ? 'all above the ridge' : above ? `${above} of ${at.length} above the ridge` : 'none above the ridge'}.<br>`
        + `Proposition 1 then gives Γ = ${gains.map((g) => g2(g)).join(', ')}.${noFit.length ? ` Dashed: batch${noFit.length > 1 ? 'es' : ''} ${noFit.join(', ')} ${noFit.length > 1 ? 'do' : 'does'} not fit in ${HWBYID[shared.hw].gb} GB at this context.` : ''}`;
    }
    shared.subs.push(draw);
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 6: the gain is a function of the ridge point (exact)
  // Scaling peak and bandwidth together scales every pass time alike, so the gain depends on the
  // accelerator only through I* = peak / bandwidth. The curve is drawn for a chip with unlimited memory.
  function figAccel(root) {
    const RIDGES = Object.fromEntries(ACCEL.map((id) => [id, ridge(hwOf(id))]));
    const ISCAN = [...new Set([...Array.from({ length: 31 }, (_, i) => 25 * 32 ** (i / 30)), ...Object.values(RIDGES)])].sort((a, b) => a - b);
    const st = { mode: 'short', i: ISCAN.indexOf(RIDGES[shared.hw]) };
    const ctl = el('<div class="fig-controls"></div>');
    ctl.append(seg([['short', 'short context, best batch'], ['long', 'long context, C = 1M, batch 1']], st.mode, (v) => { st.mode = v; draw(); }));
    const pic = el('<div></div>'), out = el('<div class="readout"></div>');
    root.append(ctl, pic, out);
    let data = null;
    function compute() {
      const Ikv = M.compact(DG, W0, O0, hwOf('h200')).Ikv;
      const pts = ISCAN.map((Is) => {
        const hw = { peak: Is * 1e12, peakFp8: 2 * Is * 1e12, bw: 1e12, mem: Infinity, fp8: false, kernels: 'whole' };
        let best = { g: 0, B: 1, uAR: 0 }, tdc = null;
        for (let B = 1; B <= 40; B++) { const r = steady(DG, hw, B, 1024); if (r.gain > best.g) best = { g: r.gain, B, uAR: util(r.ar) }; if (tdc === null && r.tdCompute) tdc = B; }
        return { Is, best, tdc, long: steady(DG, hw, 1, 1 << 20).gain, limit: Is / (OM * Ikv) };
      });
      const dots = ACCEL.map((id) => { const hw = hwOf(id), p = pts[ISCAN.indexOf(RIDGES[id])]; return { id, p, fitShort: fits(DG, hw, p.best.B, 1024), fitLong: fits(DG, hw, 1, 1 << 20) }; });
      data = { pts, dots, Ikv };
    }
    draggable(pic, () => st.xs, ISCAN, () => st.i, (i) => { st.i = i; const hit = ACCEL.find((id) => RIDGES[id] === ISCAN[i]); if (hit) setHw(hit); draw(); });
    function draw() {
      if (!data) compute();
      const c = colors(), W = widthOf(pic), H = 320, m = { l: 48, r: 22, t: 18, b: 42 }, short = st.mode === 'short';
      const xs = scale(25, 800, m.l, W - m.r), ylo = 0.25, yhi = 14, ys = scale(ylo, yhi, H - m.b, m.t);
      const val = (p) => (short ? p.best.g : p.long), P = data.pts[st.i];
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Gain against ridge point">`;
      s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => `${v}×`, c);
      for (const v of [25, 50, 100, 200, 400, 800]) s += `<text x="${xs(v)}" y="${H - m.b + 15}" text-anchor="middle">${v}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">ridge point I* = π/β, FLOP per byte</text>`;
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(1)}" y2="${ys(1)}" stroke="${c.muted}" stroke-width="1.5"/><text x="${W - m.r - 4}" y="${ys(1) + 15}" text-anchor="end">parity, Γ = 1</text>`;
      if (!short) {
        const f = (Is) => Is / (OM * data.Ikv), xa = Math.max(25, ylo * OM * data.Ikv);
        s += `<line x1="${xs(xa)}" x2="${xs(800)}" y1="${ys(f(xa))}" y2="${ys(f(800))}" stroke="${c.td}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
        s += `<text class="td halo" x="${xs(640)}" y="${ys(f(640)) + 18}" text-anchor="end">C → ∞: I*/(ω ${sub('I', 'attn')})</text>`;
      }
      s += `<polyline fill="none" stroke="${c.ink}" stroke-width="2.2" stroke-linejoin="round" points="${data.pts.map((p) => `${xs(p.Is)},${ys(val(p))}`).join(' ')}"/>`;
      // marker, under the dots so that a dot stays clickable
      s += `<g class="drag" tabindex="0" role="slider" aria-label="Ridge point" aria-valuemin="25" aria-valuemax="800" aria-valuenow="${Math.round(P.Is)}" aria-valuetext="ridge point ${fmt(P.Is)}, gain ${g2(val(P))}"><rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${H - m.b - m.t}" fill="transparent"/>`;
      s += `<line x1="${xs(P.Is)}" x2="${xs(P.Is)}" y1="${m.t}" y2="${H - m.b}" stroke="${c.accent}" stroke-width="2"/><path d="M${xs(P.Is) - 7},${H - m.b} l7,-11 l7,11 z" fill="${c.accent}"/></g>`;
      data.dots.forEach((d, k) => {
        const fit = short ? d.fitShort : d.fitLong, on = d.id === shared.hw && RIDGES[d.id] === P.Is;
        const x = xs(d.p.Is), y = ys(val(d.p)), up = k % 2 === 0;
        s += `<g class="dot" data-id="${d.id}" tabindex="0" role="button" aria-pressed="${on}" aria-label="${HWBYID[d.id].label}" style="cursor:pointer">`;
        s += `<circle cx="${x}" cy="${y}" r="14" fill="transparent"/>`;
        s += `<circle cx="${x}" cy="${y}" r="${on ? 7.5 : 6}" fill="${fit ? c.td : c.bg}" stroke="${on ? c.ink : c.td}" stroke-width="${on ? 2.5 : 2}"/>`;
        if (W >= 560 || on || ['h20', 'h200', 'spark'].includes(d.id)) s += `<text class="${on ? 'strong' : ''} halo" x="${x}" y="${up ? y - 13 : y + 21}" text-anchor="middle">${SHORT[d.id]}</text>`;
        s += '</g>';
      });
      s += `<circle cx="${xs(P.Is)}" cy="${ys(val(P))}" r="3.5" fill="${c.accent}" pointer-events="none"/>`;
      pic.innerHTML = `${s}</svg>`;
      st.xs = xs;
      pic.querySelectorAll('.dot').forEach((g) => {
        const pick = () => { st.i = ISCAN.indexOf(RIDGES[g.dataset.id]); if (g.dataset.id === shared.hw) draw(); else setHw(g.dataset.id); pic.querySelector(`.dot[data-id="${g.dataset.id}"]`).focus({ preventScroll: true }); };
        g.addEventListener('click', pick);
        g.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { pick(); ev.preventDefault(); } });
      });
      const dot = data.dots.find((d) => d.id === shared.hw && RIDGES[d.id] === P.Is), h = dot && HWBYID[dot.id];
      const who = dot ? `<b>${h.label}</b>, <i>I</i>* = ${fmt(P.Is)} (${fmt(h.tflops)} TFLOP/s over ${fmt(h.tbps)} TB/s)` : `<b>A chip with <i>I</i>* = ${fmt(P.Is)}</b>`;
      out.innerHTML = short
        ? `${who}: best roofline gain <b>Γ = ${g2(P.best.g)}× at B = ${P.best.B}</b>; a TD pass is compute-bound from batch ${P.tdc}, and at the best batch an AR step uses ${pct(P.best.uAR)} of the peak compute.${dot && !dot.fitShort ? ` The model does not fit in ${h.gb} GB.` : ''}`
        : `${who}: at 1M context and batch 1, <b>Γ = ${g2(P.long)}×</b>; as the context grows it tends to <i>I</i>*/(ω ${hsub('I', 'attn')}) = ${fmt(P.Is)}/(${OM} × ${fmt(data.Ikv)}) = ${g2(P.limit)}.${P.limit < 1 ? ' The ridge is so low that AR wins at long context.' : ''}${dot && !dot.fitLong ? ` The model and a 1M-token cache do not fit in ${h.gb} GB.` : ''}`;
    }
    shared.subs.push((id) => { st.i = ISCAN.indexOf(RIDGES[id]); draw(); });
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- quiz
  // Settings on which Proposition 1 and the exact accounting agree, and which fit in memory.
  const QUIZ = [
    { hw: 'h200', m: DG, model: 'DiffusionGemma 26B-A4B', B: 1, C: 1024,
      note: 'Both decoders are limited by the bytes they read, and per token TD reads fewer: Γ = TPF/Π = 17.1/7.7, with Π the overall penalty at this context.' },
    { hw: 'h20', m: DG, model: 'DiffusionGemma 26B-A4B', B: 16, C: 1024,
      note: 'The ridge of the H20 is low, I* = 37, so TD has little idle compute to spend: under the exact accounting a TD pass is compute-bound from batch 1, and AR catches up by batch 5.' },
    { hw: 'h200', m: LL, model: 'Llama 3.1 8B and its TD twin', B: 32, C: 1024,
      note: 'All the weights of a dense model are shared by the batch, so the intensity of AR grows in proportion to it: at batch 32 an AR step already uses 12% of the peak, more than the 1/ω = 6.7% that leaves room for TD.' },
    { hw: 'h200', m: LL, model: 'Llama 3.1 8B and its TD twin', B: 1, C: 1024,
      note: 'A dense model pays no MoE penalty: a canvas reads the same weights as one token, so a TD pass is past the ridge at batch 1 while AR uses 0.5% of the peak. Γ is then the compute cap I*/(ω I_AR).' },
    { hw: 'h200', m: DG, model: 'DiffusionGemma 26B-A4B', B: 4, C: 1 << 20,
      note: 'At 1M context the KV cache is 85% of the bytes of an AR step. TD reads it once for 256 queries and passes the ridge; AR reads it for one query and stays near I_AR = 7.' },
    { hw: 'h20', m: DG, model: 'DiffusionGemma 26B-A4B', B: 1, C: 1 << 20,
      note: 'A long context does not rescue a low ridge: Γ tends to I*/(ω I_attn) = 37/(15 × 8) = 0.31, below 1 at any batch.' },
  ];
  function figQuiz(root) {
    const st = { k: 0, answers: [] };
    const data = QUIZ.map((q) => {
      const hw = hwOf(q.hw), p = prop1(q.m, hw, q.B, q.C);
      return { ...q, hw, hwLabel: HWBYID[q.hw].label, gain: steady(q.m, hw, q.B, q.C).gain, p };
    });
    const box = el('<div></div>');
    root.appendChild(box);
    const tex = (t) => t.replaceAll('I_AR', 'I<sub>AR</sub>').replaceAll('I_attn', 'I<sub>attn</sub>');
    function draw() {
      const q = data[st.k], ans = st.answers[st.k], done = ans !== undefined, td = q.gain > 1, p = q.p;
      const answered = st.answers.filter((a) => a !== undefined).length, score = st.answers.filter((a, i) => a === (data[i].gain > 1)).length;
      const bound = (I) => (I >= p.Is ? 'compute' : 'memory');
      // vLLM reaches 84% of the TD bound at batch 1 and 34% at batch 64 (appendix); the AR bound is not measured there.
      const thin = !td ? '' : q.B === 1
        ? (q.m === DG && q.C === 1024 && q.hwLabel.startsWith('H200') ? ' For this model, accelerator and batch, vLLM reaches 84% of the TD bound (appendix).' : '')
        : (q.gain < 3 ? ' This lead is thin: in vLLM, TD reaches 84% of its bound at batch 1 and 34% at batch 64 (appendix, which does not measure AR). If TD ran at 34% of its bound and AR at its own, a roofline lead below about ×3 would be lost.' : '');
      box.innerHTML = `
        <div class="quiz-head"><span>Setting ${st.k + 1} of ${data.length}</span><span>${answered ? `${score} of ${answered} right so far` : ''}</span></div>
        <div class="quiz-case">
          <div><div class="k">model</div><div class="v">${q.model}</div></div>
          <div><div class="k">accelerator</div><div class="v">${q.hwLabel}</div></div>
          <div><div class="k">batch</div><div class="v">${q.B}</div></div>
          <div><div class="k">context</div><div class="v">${kctx(q.C)} tokens</div></div>
        </div>
        <p class="quiz-q">Canvas <i>b</i> = 256, <i>S</i> = 14 denoising passes. Which decoder does the roofline model make faster?</p>
        <div class="quiz-choices">
          <button type="button" data-a="td">TD is faster<small>block diffusion</small></button>
          <button type="button" data-a="ar">AR is faster<small>autoregressive decoding</small></button>
        </div>
        <div class="quiz-reveal" ${done ? '' : 'hidden'}>
          <div>
            <div class="quiz-verdict ${ans === td ? 'ok' : 'no'}">${ans === td ? 'Right' : 'Not quite'}: under the roofline ${td ? 'TD' : 'AR'} is faster, Γ = ${g2(q.gain)}.</div>
            <p class="quiz-why">Proposition 1: <span class="ar">AR</span> is ${bound(p.IAR)}-bound (I<sub>AR</sub> = ${fmt(p.IAR)}) and <span class="td">TD</span> is ${bound(p.ITD)}-bound (I<sub>TD</sub> = ${fmt(p.ITD)}), with the ridge at I* = ${fmt(p.Is)}. The FLOPs of TD run ×${fmt(p.speed)} faster and it does ω = ${p.om} times as many, so Γ = ${g2(p.gain)}; the exact accounting, with the vocabulary${q.m.selfCond ? ' and self-conditioning' : ''}, gives ${g2(q.gain)}. ${tex(q.note)}${thin}</p>
          </div>
          <div class="quiz-mini">${done ? rooflineSvg({ hw: q.hw, IAR: p.IAR, ITD: p.ITD, W: Math.min(340, widthOf(box)), H: 220, mini: true }) : ''}</div>
        </div>
        <div class="quiz-nav">
          <button type="button" class="fig-btn" data-n="-1" ${st.k ? '' : 'disabled'}>Previous</button>
          <button type="button" class="fig-btn ${done ? 'primary' : ''}" data-n="1" ${st.k < data.length - 1 ? '' : 'disabled'}>Next</button>
          <span class="quiz-dots">${data.map((d, i) => `<span class="${st.answers[i] === undefined ? '' : st.answers[i] === (d.gain > 1) ? 'ok' : 'no'} ${i === st.k ? 'cur' : ''}"></span>`).join('')}</span>
          ${answered === data.length ? '<button type="button" class="fig-btn" data-r="1">Start again</button>' : ''}
        </div>`;
      box.querySelectorAll('.quiz-choices button').forEach((b) => {
        const mine = b.dataset.a === 'td';
        if (done) { b.disabled = true; if (mine === td) b.classList.add('right'); else if (mine === ans) b.classList.add('wrong'); }
        b.addEventListener('click', () => { st.answers[st.k] = mine; draw(); });
      });
      box.querySelectorAll('[data-n]').forEach((b) => b.addEventListener('click', () => { st.k = Math.max(0, Math.min(data.length - 1, st.k + +b.dataset.n)); draw(); }));
      const r = box.querySelector('[data-r]'); if (r) r.addEventListener('click', () => { st.k = 0; st.answers = []; draw(); });
    }
    draw();
    onTheme(draw);
    watchWidth(box, draw);
  }

  // ---------------------------------------------------------------- expandable TL;DR cards
  // Each card with a .card-more panel opens in place, across the full row. Its slider, built on first
  // opening, moves a dot along a small curve of the quantity and updates a one-sentence readout.
  const CTX4 = [...CONTEXTS, 2 << 20, 4 << 20];
  const H200 = hwOf('h200');
  const bestOverBatch = (b) => { let tdc = null, best = { g: 0, B: 1 }; for (let B = 1; B <= 256; B++) { const r = steady(DG, H200, B, 1024, O0, { b }); if (tdc === null && r.tdCompute) tdc = B; if (r.gain > best.g) best = { g: r.gain, B }; } return { tdc, best }; };
  const memo = (f) => { const c = new Map(); return (v) => { if (!c.has(v)) c.set(v, f(v)); return c.get(v); }; };
  const MICRO = {
    omega: () => {
      const f = memo((S) => [1, 8].map((B) => prop1(DG, H200, B, 1024, { S }).gain));
      return { label: 'denoising passes <i>S</i>', values: Array.from({ length: 32 }, (_, i) => i + 1), i: 13, show: String, ylab: 'Γ at batch 1', y: (S) => f(S)[0],
        read: (S) => `ω = ${S + 1}; Proposition 1 on an H200: Γ = <b>${g2(f(S)[0])}</b> at batch 1, <b>${g2(f(S)[1])}</b> at batch 8.` };
    },
    canvas: () => {
      const f = memo(bestOverBatch);
      return { label: 'canvas length <i>b</i>', values: [32, 64, 128, 256, 512, 1024], i: 3, show: String, ylab: 'best Γ', y: (b) => f(b).best.g,
        read: (b) => `TPF = ${b}/15 = <b>${fmt(b / 15)}</b>; a TD pass is compute-bound from batch <b>${f(b).tdc ?? '> 256'}</b>; best Γ = <b>${g2(f(b).best.g)}</b> at batch ${f(b).best.B} (exact, H200, context 1k).` };
    },
    lin: () => {
      const f = memo((B) => prop1(DG, H200, B, 1024).Pi);
      return { label: 'batch <i>B</i>', values: BATCHES, i: 0, show: String, ylab: sub('Π', 'lin'), y: f,
        read: (B) => `An AR step touches ${fmt(M.distinctExperts(DG, B))} of 128 experts per layer, a canvas ${fmt(M.distinctExperts(DG, B * W0.b))}: ${hsub('Π', 'lin')} = <b>${fmt(f(B))}</b>.` };
    },
    attn: () => {
      const c = prop1(M.MODELS.dgemma_sparse.build(), H200, 1, 1024).c, f = (C) => c.qkvTD(C) / c.qkv(C);
      return { label: 'context <i>C</i>', values: CTX4, i: 7, show: kctx, ylab: sub('Π', 'attn'), y: f,
        read: (C) => `With sparse global layers (<i>m</i> = 2048), a canvas reads the union of ${comma(c.unionRead(C, W0.b))} cached tokens per sparse layer where one query reads ${comma(Math.min(C, c.msel))}; over all attention layers ${hsub('Π', 'attn')} = <b>${fmt(f(C))}</b>.` };
    },
    batch: () => {
      const f = memo((B) => steady(DG, H200, B, 1024));
      return { label: 'batch <i>B</i>', values: BATCHES, i: 0, show: String, ylab: 'Γ', y: (B) => f(B).gain, ref: 1,
        read: (B) => { const r = f(B); return `An <span class="q-ar">AR step uses <b>${pct(util(r.ar))}</b></span> of the peak compute and a <span class="q-td">TD pass <b>${pct(util(r.d1))}</b></span>: TD is ${r.tdCompute ? 'compute' : 'memory'}-bound, Γ = <b>${g2(r.gain)}</b> (exact, H200, context 1k).${fits(DG, H200, B, 1024) ? '' : ' This batch does not fit in 141 GB.'}`; } };
    },
    context: () => {
      // cache share of the bytes of an AR step under Proposition 1: θ_attn of Proposition 2
      const f = memo((C) => { const p = prop1(DG, H200, 4, C), q = p.c.qkv(C); return { p, share: q / (p.c.WAR(4) / 4 + q), exact: steady(DG, H200, 4, C).gain }; });
      return { label: 'context <i>C</i>', values: CONTEXTS, i: 0, show: kctx, ylab: 'cache share', y: (C) => f(C).share, log: false, fmtY: pct,
        read: (C) => { const { p, share, exact } = f(C); return `Proposition 1 at batch 4 (<i>I</i>* = 206): the cache is <b>${pct(share)}</b> of the bytes of an AR step, ${hsub('I', 'AR')} = <b>${fmt(p.IAR)}</b>, ${hsub('I', 'TD')} = <b>${fmt(p.ITD)}</b>, Γ = <b>${g2(p.gain)}</b>. Exact accounting: Γ = <b>${g2(exact)}</b>.`; } };
    },
  };
  // A small curve of spec.y over the slider's values, with a dot at the current one.
  function spark(spec, ys, i) {
    const W = 340, H = 70, m = { l: 6, r: 124, t: 10, b: 10 }, log = spec.log !== false;
    const lo = Math.min(...ys, spec.ref ?? Infinity), hi = Math.max(...ys, spec.ref ?? -Infinity), pad = hi / lo < 1.001 || hi - lo < 1e-9;
    const y = scale(pad ? lo * 0.9 || -0.1 : lo, pad ? hi * 1.1 || 0.1 : hi, H - m.b, m.t, log && lo > 0);
    const x = (k) => m.l + (k / (ys.length - 1)) * (W - m.l - m.r);
    let s = `<svg class="spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="Curve of the quantity against the slider's variable">`;
    if (spec.ref !== undefined) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(spec.ref)}" y2="${y(spec.ref)}" style="stroke:var(--bdr-muted)" stroke-dasharray="3 3"/>`;
    s += `<polyline fill="none" style="stroke:var(--ink)" stroke-width="1.8" stroke-linejoin="round" points="${ys.map((v, k) => `${x(k)},${y(v)}`).join(' ')}"/>`;
    s += `<line x1="${x(i)}" x2="${x(i)}" y1="${m.t - 4}" y2="${H - m.b + 4}" style="stroke:var(--accent)" stroke-opacity="0.5"/>`;
    s += `<circle cx="${x(i)}" cy="${y(ys[i])}" r="4.5" style="fill:var(--accent);stroke:var(--bdr-surface)" stroke-width="1.5"/>`;
    s += `<text x="${W - m.r + 8}" y="${Math.max(m.t + 8, Math.min(H - m.b, y(ys[i]) + 4))}" class="strong">${spec.ylab} = ${(spec.fmtY || fmt)(ys[i])}</text>`;
    return `${s}</svg>`;
  }
  function initMicro(box) {
    const spec = MICRO[box.dataset.micro](), out = el('<div class="readout"></div>'), pic = el('<div class="micro-pic"></div>');
    const sl = slider(spec.label, spec.values, spec.i, spec.show);
    const ys = spec.values.map(spec.y);
    const sync = () => { pic.innerHTML = spark(spec, ys, sl.index); out.innerHTML = spec.read(sl.value); };
    sl.input.addEventListener('input', sync);
    const row = el('<div class="fig-controls micro"></div>'); row.append(sl.lab, pic);
    box.append(row, out); sync();
  }
  function initCards() {
    document.querySelectorAll('.bdr .cards').forEach((grid) => {
      const cards = [...grid.querySelectorAll('.bdr-card')].filter((c) => c.querySelector('.card-more'));
      if (!cards.length) return;
      const setOpen = (card, on) => {
        card.classList.toggle('open', on);
        card.querySelector('.card-more').hidden = !on;
        card.querySelector('.card-toggle').setAttribute('aria-expanded', String(on));
        card.querySelector('.card-toggle').textContent = on ? 'less −' : 'more +';
        if (on) card.querySelectorAll('[data-micro]:empty').forEach(initMicro);
      };
      for (const card of cards) {
        card.classList.add('has-more');
        const t = el('<button type="button" class="card-toggle" aria-expanded="false">more +</button>');
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

  // ---------------------------------------------------------------- mount
  const FIGS = { quiz: figQuiz, accounting: figAccounting, reuse: figReuse, roofline: figRoofline, bestbatch: figBestBatch, context: figContext, accel: figAccel };
  document.querySelectorAll('[data-fig]').forEach((root) => {
    try { FIGS[root.dataset.fig](root); } catch (e) { root.textContent = 'This figure failed to load.'; console.error(e); }
  });
  const redraw = () => rerenders.forEach((f) => f());
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
