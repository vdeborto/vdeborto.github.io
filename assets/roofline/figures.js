// Interactive figures for the Block Diffusion Rooflines page. Every number comes from the note's own
// cost model (window.BDModel) and helpers (window.BDR, exposed by note.js), so the figures agree
// with the tables, the maps and the calculator.
(function () {
  'use strict';
  const M = window.BDModel, X = window.BDR;
  if (!M || !X) return;
  const { HWBYID, REF, boe, omega, fmt, css } = X;

  // ---------------------------------------------------------------- helpers
  const hwOf = (id) => { const h = HWBYID[id]; return { peak: h.tflops * 1e12, peakFp8: 2 * h.tflops * 1e12, bw: h.tbps * 1e12, mem: h.gb * 1e9, fp8: false }; };
  const ridge = (hw) => hw.peak / hw.bw;
  const DG = M.gemma4_26b();
  const W0 = REF.w, O0 = REF.o;
  const steady = (m, hw, B, C, o = O0, w = {}) => M.steady(m, { ...W0, ...w, B }, o, hw, C);
  const fits = (m, hw, B, C, o = O0) => M.footprint(m, { ...W0, B, P: C, G: W0.b }, o, 'td').total <= hw.mem;
  const intensities = (m, hw, B, C, o = O0) => {
    const w = { ...W0, B };
    const a = M.pass(m, M.passes.arDecode(w, C), o), t = M.pass(m, M.passes.tdDenoise(w, C, 1), o);
    return { IAR: a.F / a.Q, ITD: t.F / t.Q };
  };
  const g2 = (x) => (x >= 10 ? x.toFixed(1) : x.toFixed(2));
  const kctx = (C) => (C >= 1 << 20 ? `${fmt(C / (1 << 20))}M` : `${fmt(C / 1024)}k`);
  const pct = (x) => (x >= 0.1 ? `${Math.round(x * 100)}%` : `${fmt(x * 100, 2)}%`);
  const sub = (base, s) => `${base}<tspan baseline-shift="sub" font-size="8.5">${s}</tspan>`;
  const hsub = (base, s) => `<i>${base}</i><sub>${s}</sub>`;
  const colors = () => ({
    ar: css('--bdr-ar'), td: css('--bdr-td'), ink: css('--ink'), muted: css('--bdr-muted'), rule: css('--rule'),
    bg: css('--bdr-bg'), surface: css('--bdr-surface'), accent: css('--accent'), soft: css('--bdr-accent-soft'),
    ok: css('--bdr-int-mem'), comp: css('--bdr-int-comp'),
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
    return { lab, input, get value() { return values[+input.value]; }, set(v) { input.value = nearestIndex(values, v); sync(); } };
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
  // SVGs are drawn at the width they are shown at, so that 11px text stays 11px on a phone.
  const widthOf = (box, lo = 330, hi = 1000) => Math.max(lo, Math.min(hi, Math.round(box.clientWidth || 640)));
  function watchWidth(box, draw) {
    let last = box.clientWidth;
    new ResizeObserver(() => { const w = box.clientWidth; if (Math.abs(w - last) > 4) { last = w; draw(); } }).observe(box);
  }
  const rerenders = [];
  const onTheme = (f) => rerenders.push(f);

  // ---------------------------------------------------------------- roofline renderer (Figure 2, quiz)
  // One accelerator's roofline with an AR dot and a TD dot at their intensities; the bracket on the
  // right is the ratio of their attainable FLOP rates.
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

  // ---------------------------------------------------------------- Figure 1: byte reuse
  function figReuse(root) {
    const st = { b: 64, S: 14, t: 0, timer: 0 };
    const ctl = el('<div class="fig-controls"></div>');
    const bSeg = el('<label class="wide"><span>canvas length <i>b</i></span><span class="segbox"></span></label>');
    bSeg.querySelector('.segbox').appendChild(seg([16, 32, 64, 128, 256].map((v) => [v, String(v)]), st.b, (v) => { st.b = v; reset(); }));
    const sS = slider('denoising passes <i>S</i>', Array.from({ length: 32 }, (_, i) => i + 1), st.S - 1, (v) => `${v}  (ω = ${v + 1})`);
    sS.input.addEventListener('input', () => { st.S = sS.value; reset(); });
    const btns = el('<div class="seg"><button type="button" class="fig-btn primary" data-a="play">Play</button><button type="button" class="fig-btn" data-a="step">Step</button><button type="button" class="fig-btn" data-a="reset">Reset</button></div>');
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
      const c = colors(), W = widthOf(pic), x0 = W < 480 ? 118 : 150, x1 = W - 16, w = x1 - x0, n = om(), t = st.t, b = st.b;
      const cell = w / Math.max(b, n), committed = t >= n;
      let s = `<svg viewBox="0 0 ${W} 214" role="img" aria-label="After ${t} of ${n} passes: AR has ${t} tokens, TD ${committed ? b : 0}">`;
      // shared time axis: one slot per pass, each a read of the weights
      s += `<text x="0" y="27" class="ink">forward passes</text><text x="0" y="41">each reads W</text>`;
      const sw = w / n;
      for (let j = 0; j < n; j++) {
        const done = j < t, last = j === n - 1;
        s += `<rect x="${x0 + j * sw + 1}" y="18" width="${Math.max(2, sw - 2)}" height="22" rx="2" fill="${done ? c.ink : 'none'}" fill-opacity="${done ? 0.82 : 0}" stroke="${c.muted}" stroke-opacity="0.6"/>`;
        if (sw > 22) s += `<text x="${x0 + j * sw + sw / 2}" y="33" text-anchor="middle" style="fill:${done ? c.bg : c.muted};font-size:10px">${last ? (sw >= 46 ? 'commit' : '+1') : j + 1}</text>`;
      }
      // AR: one token per pass
      s += `<text x="0" y="96" class="ar">AR</text><text x="0" y="110">1 token per pass</text>`;
      for (let j = 0; j < n; j++) s += `<rect x="${x0 + j * cell}" y="82" width="${Math.max(1, cell - (cell > 4 ? 1.5 : 0.4))}" height="26" fill="${j < t ? c.ar : 'none'}" stroke="${c.ar}" stroke-opacity="${j < t ? 1 : 0.3}" stroke-width="${cell > 4 ? 1 : 0.4}"/>`;
      // TD: a canvas of b positions, all updated by every pass, committed by the last one
      s += `<text x="0" y="158" class="td">TD</text><text x="0" y="172">b positions</text>`;
      const shade = committed ? 1 : Math.min(t, st.S) / st.S * 0.75;
      for (let j = 0; j < b; j++) s += `<rect x="${x0 + j * cell}" y="144" width="${Math.max(1, cell - (cell > 4 ? 1.5 : 0.4))}" height="26" fill="${c.td}" fill-opacity="${committed ? 1 : shade * 0.85}" stroke="${c.td}" stroke-opacity="${committed ? 1 : 0.35}" stroke-width="${cell > 4 ? 1 : 0.4}"/>`;
      if (!committed && t > 0) s += `<text x="${x0}" y="190">denoising: every position refined by pass ${Math.min(t, st.S)} of ${st.S}</text>`;
      if (committed) s += `<text x="${x0}" y="190" class="td">committed: ${b} tokens</text>`;
      s += '</svg>';
      pic.innerHTML = s;
      const tpf = b / n;
      out.innerHTML = t < n
        ? `After <b>${t}</b> of <b>${n}</b> passes: both decoders have read the weights ${t} time${t === 1 ? '' : 's'}; <span class="ar">AR</span> has committed ${t} token${t === 1 ? '' : 's'}, <span class="td">TD</span> none yet.`
        : `<span class="big">In ${n} passes, <span class="ar">AR</span> commits ${n} tokens and <span class="td">TD</span> commits ${b}: TPF = ${b}/${n} = <b>${fmt(tpf)}</b>.</span><br>Per token, TD reads ${fmt(1 / tpf)} × the weight bytes of AR and performs ${n} × its FLOPs. ${tpf >= 1 ? `While both are memory-bound, TD is ${fmt(tpf)} × faster.` : 'With fewer positions than passes, TD reads more per token than AR and loses even when memory-bound.'}`;
    }
    reset();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 2: live roofline
  const HW_PICK = ['h200', 'b200', 'a100', 'h20', 'spark', 'mi355x', 'v6e'];
  const BATCHES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
  const CONTEXTS = Array.from({ length: 11 }, (_, i) => 1024 * 2 ** i);
  function hwSelect(ids, current) {
    const s = el(`<select>${ids.map((id) => `<option value="${id}"${id === current ? ' selected' : ''}>${HWBYID[id].label}</option>`).join('')}</select>`);
    return s;
  }
  function figRoofline(root) {
    const ctl = el('<div class="fig-controls"></div>');
    const hwLab = el('<label class="sel">accelerator</label>'), hs = hwSelect(HW_PICK, 'h200'); hwLab.appendChild(hs);
    const sB = slider('batch <i>B</i>', BATCHES, 0, (v) => String(v));
    const sC = slider('context <i>C</i>', CONTEXTS, 0, kctx);
    const presets = seg([['r1', 'R1 · batch 1'], ['r2', 'R2 · batch 8'], ['r6', 'R6 · long context']], 'r1', (p) => {
      const [B, C] = { r1: [1, 1024], r2: [8, 1024], r6: [4, 1 << 20] }[p]; sB.set(B); sC.set(C); draw();
    });
    ctl.append(hwLab, sB.lab, sC.lab, presets);
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    const clearPresets = () => presets.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
    sB.input.addEventListener('input', () => { clearPresets(); draw(); });
    sC.input.addEventListener('input', () => { clearPresets(); draw(); });
    hs.addEventListener('change', draw);
    function draw() {
      const hw = hwOf(hs.value), B = sB.value, C = sC.value, x = boe(hw, O0, W0);
      const IAR = x.IAR(B, C), ITD = x.ITD(B, C), Is = ridge(hw), om = omega(W0);
      pic.innerHTML = rooflineSvg({ hw, IAR, ITD, W: widthOf(pic), H: widthOf(pic) < 480 ? 260 : 300 });
      const speed = Math.min(ITD, Is) / Math.min(IAR, Is), G = speed / om, exact = steady(DG, hw, B, C).gain;
      const bound = (I) => (I >= Is ? 'compute-bound' : 'memory-bound');
      out.innerHTML = `${hsub('I', 'AR')} = ${fmt(IAR)} (<span class="ar">AR ${bound(IAR)}</span>), ${hsub('I', 'TD')} = ${fmt(ITD)} (<span class="td">TD ${bound(ITD)}</span>), <i>I</i>* = ${fmt(Is)}.<br>`
        + `<span class="big">TD's FLOPs run ×${fmt(speed)} faster; a canvas costs ω = ${om} passes; <b>Γ = ${fmt(speed)}/${om} = ${g2(G)}</b>.</span> Exact accounting with vocabulary: Γ = ${g2(exact)}.${fits(DG, hw, B, C) ? '' : ` At this batch and context the model and its KV cache do not fit in ${HWBYID[hs.value].gb} GB.`}`;
    }
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 3: find the best batch
  const BSCAN = [...new Set([...Array.from({ length: 40 }, (_, i) => i + 1), ...Array.from({ length: 60 }, (_, i) => Math.round(40 * 1.056 ** (i + 1))), 64, 128, 256, 512, 1024])].filter((B) => B <= 1024).sort((a, b) => a - b);
  const curveCache = new Map();
  function gainCurve(hwId) {
    if (curveCache.has(hwId)) return curveCache.get(hwId);
    const hw = hwOf(hwId), c = M.compact({ ...DG, V: 0, fSC: 0 }, W0, { ...O0, bl: 0 }, hw);
    const pts = BSCAN.map((B) => { const r = steady(DG, hw, B, 1024); return { B, g: r.gain, td: r.tdCompute, ar: r.arCompute, Pi: c.Pi(B), fit: fits(DG, hw, B, 1024) }; });
    const best = pts.reduce((a, p) => (p.g > a.g ? p : a));
    const bstar = (pts.find((p) => p.td) || {}).B;
    const k = pts.findIndex((p) => p.g < 1), parity = k > 0 ? pts[k - 1].B + (1 - pts[k - 1].g) / (pts[k].g - pts[k - 1].g) * (pts[k].B - pts[k - 1].B) : null;
    const res = { pts, best, bstar, parity };
    curveCache.set(hwId, res);
    return res;
  }
  function figBestBatch(root) {
    const st = { i: nearestIndex(BSCAN, 64), revealed: false, hw: 'h200' };
    const ctl = el('<div class="fig-controls"></div>');
    const hwLab = el('<label class="sel">accelerator</label>'), hs = hwSelect(HW_PICK, st.hw); hwLab.appendChild(hs);
    const rev = el('<div class="seg"><button type="button" class="fig-btn primary">Reveal the curve</button></div>');
    ctl.append(hwLab, rev);
    const pic = el('<div></div>'), out = el('<div class="readout"></div>');
    root.append(ctl, pic, out);
    hs.addEventListener('change', () => { st.hw = hs.value; draw(); });
    rev.querySelector('button').addEventListener('click', (ev) => { st.revealed = !st.revealed; ev.target.textContent = st.revealed ? 'Hide the curve' : 'Reveal the curve'; ev.target.classList.toggle('primary', !st.revealed); draw(); });
    function draw() {
      const c = colors(), W = widthOf(pic), H = 300, m = { l: 48, r: 18, t: 18, b: 40 };
      const cur = gainCurve(st.hw), pts = cur.pts, B = BSCAN[st.i], P = pts[st.i];
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
          s += `<text x="${xs(cur.bstar) - 6}" y="${m.t + 14}" text-anchor="end" class="ink">TD memory-bound</text><text x="${xs(cur.bstar) + 6}" y="${m.t + 14}" class="ink">TD compute-bound</text>`;
          s += `<text x="${xs(cur.bstar) - 6}" y="${H - m.b - 8}" text-anchor="end" class="td halo">${sub('B*', 'TD')} = ${cur.bstar}</text>`;
        }
        const fitPts = pts.filter((p) => p.fit), outPts = pts.filter((p, j) => !p.fit || (pts[j + 1] && !pts[j + 1].fit));
        if (fitPts.length > 1) s += `<polyline fill="none" stroke="${c.ink}" stroke-width="2.2" stroke-linejoin="round" points="${fitPts.map((p) => `${xs(p.B)},${ys(p.g)}`).join(' ')}"/>`;
        if (outPts.length > 1) s += `<polyline fill="none" stroke="${c.ink}" stroke-width="2.2" stroke-dasharray="5 4" points="${outPts.map((p) => `${xs(p.B)},${ys(p.g)}`).join(' ')}"/>`;
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
      const hw = hwOf(st.hw), x = boe(hw, O0, W0);
      if (!st.revealed) {
        out.innerHTML = `Your guess: <b>B = ${B}</b>. Reveal the curve to compare.`;
      } else {
        const off = B === cur.best.B ? `${g2(P.g)}×, the best` : `${g2(P.g)}× (${pct(P.g / cur.best.g)} of the best)`;
        const lastFit = [...pts].reverse().find((p) => p.fit);
        out.innerHTML = `<span class="big">Best: <b>${g2(cur.best.g)}× at B = ${cur.best.B}</b>; a TD pass is compute-bound from ${hsub('B*', 'TD')} = ${cur.bstar}${cur.parity ? `, and AR catches up near ${hsub('B', '×')} ≈ ${Math.round(cur.parity)}` : ', and AR does not catch up by batch 1024'}.</span><br>`
          + `${lastFit ? (lastFit.B < 1024 ? `Dashed: batches above ${lastFit.B} do not fit in ${HWBYID[st.hw].gb} GB. ` : '') : `The model does not fit in ${HWBYID[st.hw].gb} GB at any batch. `}`
          + `At the marker, B = ${B}: Γ = ${off}; <span class="ar">AR ${P.ar ? 'compute' : 'memory'}-bound</span> (${hsub('I', 'AR')} = ${fmt(x.IAR(B, 1024))}), <span class="td">TD ${P.td ? 'compute' : 'memory'}-bound</span>; MoE penalty ${hsub('Π', 'lin')} = ${fmt(P.Pi)}.`;
      }
    }
    draggable(pic, () => st.xs, BSCAN, () => st.i, (i) => { st.i = i; draw(); });
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 4: context
  const CSCAN = Array.from({ length: 41 }, (_, i) => Math.round(1024 * 2 ** (i / 4)));
  const CBATCH = [1, 8, 64, 256];
  const LAYOUTS = {
    gemma: { label: 'DiffusionGemma: 5 global, 25 windowed', m: () => DG, o: O0 },
    global: { label: 'no windows: every layer global', m: () => DG, o: { ...O0, honourWindow: false } },
    sparse: { label: 'sparse global layers, m = 2048', m: () => M.MODELS.dgemma_sparse.build(), o: O0 },
  };
  function figContext(root) {
    const st = { layout: 'gemma', i: CSCAN.indexOf(1 << 20) };
    const ctl = el('<div class="fig-controls"></div>');
    const lay = el('<label class="wide"><span>attention layout</span><span class="segbox"></span></label>');
    lay.querySelector('.segbox').appendChild(seg(Object.entries(LAYOUTS).map(([k, v]) => [k, v.label]), st.layout, (k) => { st.layout = k; draw(); }));
    ctl.append(lay);
    const pic = el('<div></div>'), out = el('<div class="readout"></div>');
    root.append(ctl, pic, out);
    const cache = new Map(), hw = hwOf('h200');
    function curves() {
      if (cache.has(st.layout)) return cache.get(st.layout);
      const L = LAYOUTS[st.layout], m = L.m();
      const res = CBATCH.map((B) => CSCAN.map((C) => { const r = steady(m, hw, B, C, L.o); return { C, g: r.gain, share: r.arKVshare, ar: r.arCompute, td: r.tdCompute, fit: fits(m, hw, B, C, L.o) }; }));
      cache.set(st.layout, res);
      return res;
    }
    draggable(pic, () => st.xs, CSCAN, () => st.i, (i) => { st.i = i; draw(); });
    function draw() {
      const c = colors(), W = widthOf(pic), H = 300, m = { l: 48, r: 18, t: 16, b: 40 }, cv = curves(), C = CSCAN[st.i];
      const all = cv.flat().map((p) => p.g);
      const ylo = Math.min(0.5, ...all) * 0.85, yhi = Math.max(...all) * 1.3, xs = scale(1024, 1 << 20, m.l, W - m.r), ys = scale(ylo, yhi, H - m.b, m.t);
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Gain against context for batches ${CBATCH.join(', ')}">`;
      s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => `${v}×`, c);
      for (const C2 of [1024, 4096, 16384, 65536, 262144, 1 << 20]) s += `<text x="${xs(C2)}" y="${H - m.b + 15}" text-anchor="middle">${kctx(C2)}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">context C, tokens in the KV cache</text>`;
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(1)}" y2="${ys(1)}" stroke="${c.muted}" stroke-width="1.5"/>`;
      const op = [0.4, 0.6, 0.8, 1];
      cv.forEach((pts, k) => {
        let seg2 = [], dash = null;
        const flush = () => { if (seg2.length > 1) s += `<polyline fill="none" stroke="${c.accent}" stroke-opacity="${op[k]}" stroke-width="2.2" ${dash ? 'stroke-dasharray="5 4"' : ''} points="${seg2.join(' ')}"/>`; };
        pts.forEach((p, j) => {
          const xy = `${xs(p.C)},${ys(p.g)}`, d = !p.fit;
          if (dash === null) dash = d;
          if (d !== dash) { seg2.push(xy); flush(); seg2 = [xy]; dash = d; } else seg2.push(xy);
        });
        flush();
        s += `<text class="halo" x="${xs(1024) + 6}" y="${ys(pts[0].g) + (k === 2 ? 15 : -7)}" style="fill:${c.accent};opacity:${0.55 + 0.45 * op[k]}">B = ${CBATCH[k]}</text>`;
      });
      s += `<g class="drag" tabindex="0" role="slider" aria-label="Context" aria-valuemin="1024" aria-valuemax="${1 << 20}" aria-valuenow="${C}" aria-valuetext="context ${kctx(C)} tokens"><rect x="${m.l}" y="${m.t}" width="${W - m.r - m.l}" height="${H - m.b - m.t}" fill="transparent"/>`;
      s += `<line x1="${xs(C)}" x2="${xs(C)}" y1="${m.t}" y2="${H - m.b}" stroke="${c.ink}" stroke-opacity="0.55" stroke-width="1.5"/>`;
      s += `<path d="M${xs(C) - 7},${H - m.b} l7,-11 l7,11 z" fill="${c.ink}"/>`;
      cv.forEach((pts, k) => { s += `<circle cx="${xs(C)}" cy="${ys(pts[st.i].g)}" r="4" fill="${c.accent}" fill-opacity="${op[k]}" stroke="${c.bg}" stroke-width="1.5"/>`; });
      s += '</g></svg>';
      pic.innerHTML = s;
      st.xs = xs;
      const at = cv.map((pts) => pts[st.i]), p64 = at[2];
      const spread = Math.max(...at.map((p) => p.g)) / Math.min(...at.map((p) => p.g));
      out.innerHTML = `At <b>C = ${kctx(C)}</b>: Γ = ${at.map((p, k) => `${g2(p.g)}× at B = ${CBATCH[k]}${p.fit ? '' : '*'}`).join(', ')}; the batches differ by a factor ${fmt(spread)}.<br>`
        + `At B = 64 the KV cache is <b>${pct(p64.share)}</b> of an AR step's bytes; <span class="ar">AR ${p64.ar ? 'compute' : 'memory'}-bound</span>, <span class="td">TD ${p64.td ? 'compute' : 'memory'}-bound</span>.${at.some((p) => !p.fit) ? ' * does not fit in 141 GB.' : ''}`;
    }
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- Figure 6: accelerators
  const ACCEL = ['h20', 'rtx5090', 'a100', 'h200', 'v5e', 'b200', 'mi355x', 'h100pcie', 'spark', 'v6e'];
  const SHORT = { h20: 'H20', rtx5090: 'RTX 5090', a100: 'A100', h200: 'H200', v5e: 'TPU v5e', b200: 'B200', mi355x: 'MI355X', h100pcie: 'H100 PCIe', spark: 'DGX Spark', v6e: 'TPU v6e' };
  function figAccel(root) {
    const st = { mode: 'short', sel: 'spark' };
    const ctl = el('<div class="fig-controls"></div>');
    ctl.append(seg([['short', 'short context, best batch'], ['long', 'long context, C = 1M']], st.mode, (v) => { st.mode = v; draw(); }));
    const pic = el('<div></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    let data = null;
    function compute() {
      const Ikv = M.compact(DG, W0, O0, hwOf('h200')).Ikv, om = omega(W0);
      data = ACCEL.map((id) => {
        const hw = hwOf(id), Is = ridge(hw);
        let best = { g: 0, B: 1 }, tdc = null;
        for (let B = 1; B <= 64; B++) { const r = steady(DG, hw, B, 1024); if (r.gain > best.g) best = { g: r.gain, B }; if (tdc === null && r.tdCompute) tdc = B; }
        const lg = steady(DG, hw, 1, 1 << 20).gain;
        const iar = intensities(DG, hw, best.B, 1024).IAR;
        return { id, Is, best, tdc, iar, fitShort: fits(DG, hw, best.B, 1024), long: lg, fitLong: fits(DG, hw, 1, 1 << 20), limit: Is / (om * Ikv) };
      });
      data.Ikv = Ikv; data.om = om;
    }
    function draw() {
      if (!data) compute();
      const c = colors(), W = widthOf(pic), H = 320, m = { l: 48, r: 22, t: 18, b: 42 }, short = st.mode === 'short';
      const xs = scale(25, 800, m.l, W - m.r), ylo = 0.25, yhi = 14, ys = scale(ylo, yhi, H - m.b, m.t);
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Gain against ridge point, by accelerator">`;
      s += logGrid(ys, m.l, W - m.r, ylo, yhi, (v) => `${v}×`, c);
      for (const v of [25, 50, 100, 200, 400, 800]) s += `<text x="${xs(v)}" y="${H - m.b + 15}" text-anchor="middle">${v}</text>`;
      s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">ridge point I* = π/β, FLOP per byte</text>`;
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(1)}" y2="${ys(1)}" stroke="${c.muted}" stroke-width="1.5"/>`;
      if (!short) {
        const f = (Is) => Is / (data.om * data.Ikv);
        const xa = Math.max(25, ylo * data.om * data.Ikv);
        s += `<line x1="${xs(xa)}" x2="${xs(800)}" y1="${ys(f(xa))}" y2="${ys(f(800))}" stroke="${c.td}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
        s += `<text class="td halo" x="${xs(640)}" y="${ys(f(640)) + 18}" text-anchor="end">C → ∞: Γ = I*/(ω ${sub('I', 'attn')})</text>`;
      }
      const sorted = [...data].sort((a, b) => a.Is - b.Is);
      sorted.forEach((d, k) => {
        const g = short ? d.best.g : d.long, fit = short ? d.fitShort : d.fitLong, on = d.id === st.sel;
        const x = xs(d.Is), y = ys(g), up = k % 2 === 0;
        s += `<g class="dot" data-id="${d.id}" tabindex="0" role="button" aria-pressed="${on}" aria-label="${HWBYID[d.id].label}" style="cursor:pointer">`;
        s += `<circle cx="${x}" cy="${y}" r="14" fill="transparent"/>`;
        s += `<circle cx="${x}" cy="${y}" r="${on ? 7.5 : 6}" fill="${fit ? c.td : c.bg}" stroke="${on ? c.ink : c.td}" stroke-width="${on ? 2.5 : 2}"/>`;
        if (W >= 560 || on || ['h20', 'h200', 'spark'].includes(d.id)) s += `<text class="${on ? 'strong' : ''} halo" x="${x}" y="${up ? y - 12 : y + 20}" text-anchor="middle">${SHORT[d.id]}</text>`;
        s += '</g>';
      });
      pic.innerHTML = `${s}</svg>`;
      pic.querySelectorAll('.dot').forEach((g) => {
        const pick = () => { st.sel = g.dataset.id; draw(); pic.querySelector(`.dot[data-id="${st.sel}"]`).focus({ preventScroll: true }); };
        g.addEventListener('click', pick);
        g.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { pick(); ev.preventDefault(); } });
      });
      const d = data.find((x) => x.id === st.sel), h = HWBYID[d.id];
      out.innerHTML = short
        ? `<b>${h.label}</b>: <i>I</i>* = ${fmt(d.Is)} (${fmt(h.tflops)} TFLOP/s over ${fmt(h.tbps)} TB/s). Best <b>Γ = ${g2(d.best.g)}× at B = ${d.best.B}</b>, at the transition: a TD pass is compute-bound from B = ${d.tdc}. At the best batch an AR step reaches ${hsub('I', 'AR')} = ${fmt(d.iar)}, i.e. ${pct(d.iar / d.Is)} of the peak.${d.fitShort ? '' : ` The model does not fit in ${h.gb} GB.`}`
        : `<b>${h.label}</b>: at 1M context and batch 1, <b>Γ = ${g2(d.long)}×</b>; as the context grows it tends to <i>I</i>*/(ω ${hsub('I', 'attn')}) = ${fmt(d.Is)}/(${data.om} × ${fmt(data.Ikv)}) = ${g2(d.limit)}.${d.limit < 1 ? ' The ridge is so low that AR wins at long context.' : ''}${d.fitLong ? '' : ` The model and a 1M-token cache do not fit in ${h.gb} GB.`}`;
    }
    draw();
    onTheme(draw);
    watchWidth(pic, draw);
  }

  // ---------------------------------------------------------------- quiz
  const LL = M.MODELS.llama31_8b.build();
  const QUIZ = [
    { hw: 'h200', m: DG, model: 'DiffusionGemma 26B-A4B', B: 1, C: 1024,
      note: 'This is the batch-1 headline: both decoders are limited by the bytes they read, and TD reads fewer per token.' },
    { hw: 'h200', m: DG, model: 'DiffusionGemma 26B-A4B', B: 160, C: 1024,
      note: 'AR need not be compute-bound to win: it is enough that I_AR > I*/ω, here 206/15 = 13.7. TD then does 15 times the work, and its FLOPs run only I*/I_AR times faster, less than 15.' },
    { hw: 'h20', m: DG, model: 'DiffusionGemma 26B-A4B', B: 16, C: 1024,
      note: 'The H20 has a low ridge point, I* = 37: a TD pass is compute-bound from batch 1, so TD has little idle compute to spend, and AR catches up by batch 5.' },
    { hw: 'h200', m: DG, model: 'DiffusionGemma 26B-A4B', B: 4, C: 1 << 20,
      note: 'At 1M context the KV cache is 85% of an AR step’s bytes. TD reads it once for 256 queries and reaches the ridge; AR reads it for one query and does not.' },
    { hw: 'h200', m: LL, model: 'Llama 3.1 8B and its TD twin', B: 1, C: 1024,
      note: 'A dense model pays no MoE penalty: a canvas reads the same weights as one token, so a TD pass is already past the ridge at batch 1 while AR uses 0.5% of the peak. Γ is then the compute cap I*/(ω I_AR).' },
    { hw: 'h200', m: LL, model: 'Llama 3.1 8B and its TD twin', B: 16, C: 32768,
      note: 'At batch 16 and 1k context AR wins (Γ = 0.94). At 32k, Llama’s cache, 131 kB per token, is 82% of an AR step’s bytes, and the batch no longer helps AR.' },
  ];
  function figQuiz(root) {
    const st = { k: 0, answers: [] };
    const data = QUIZ.map((q) => {
      const hw = hwOf(q.hw), r = steady(q.m, hw, q.B, q.C), I = intensities(q.m, hw, q.B, q.C);
      const Is = ridge(hw);
      return { ...q, hw: hw, hwLabel: HWBYID[q.hw].label, gain: r.gain, arC: r.arCompute, tdC: r.tdCompute, ...I, speed: Math.min(I.ITD, Is) / Math.min(I.IAR, Is) };
    });
    const box = el('<div></div>');
    root.appendChild(box);
    function draw() {
      const q = data[st.k], ans = st.answers[st.k], done = ans !== undefined, td = q.gain > 1;
      const score = st.answers.filter((a, i) => a === (data[i].gain > 1)).length;
      box.innerHTML = `
        <div class="quiz-head"><span>Setting ${st.k + 1} of ${data.length}</span><span>${st.answers.length ? `${score} of ${st.answers.filter((a) => a !== undefined).length} right so far` : ''}</span></div>
        <div class="quiz-case">
          <div><div class="k">model</div><div class="v">${q.model}</div></div>
          <div><div class="k">accelerator</div><div class="v">${q.hwLabel}</div></div>
          <div><div class="k">batch</div><div class="v">${q.B}</div></div>
          <div><div class="k">context</div><div class="v">${kctx(q.C)} tokens</div></div>
        </div>
        <p class="quiz-q">Canvas <i>b</i> = 256, <i>S</i> = 14 denoising passes. Which decoder has the higher throughput?</p>
        <div class="quiz-choices">
          <button type="button" data-a="td">TD is faster<small>block diffusion</small></button>
          <button type="button" data-a="ar">AR is faster<small>autoregressive decoding</small></button>
        </div>
        <div class="quiz-reveal" ${done ? '' : 'hidden'}>
          <div>
            <div class="quiz-verdict ${ans === td ? 'ok' : 'no'}">${ans === td ? 'Right' : 'Not quite'}: ${td ? 'TD' : 'AR'} is faster, Γ = ${g2(q.gain)}.</div>
            <p class="quiz-why"><span class="ar">AR</span> is ${q.arC ? 'compute' : 'memory'}-bound (I<sub>AR</sub> = ${fmt(q.IAR)}), <span class="td">TD</span> is ${q.tdC ? 'compute' : 'memory'}-bound (I<sub>TD</sub> = ${fmt(q.ITD)}), and the ridge is at I* = ${fmt(ridge(q.hw))}. TD's FLOPs run ×${fmt(q.speed)} faster and it does ω = ${omega(W0)} times as many, so Γ ≈ ${g2(q.speed / omega(W0))}${q.m.selfCond ? '; the exact Γ is lower because DiffusionGemma’s denoising passes also run its self-conditioning layers' : ''}. ${q.note.replaceAll('I_AR', 'I<sub>AR</sub>')}</p>
          </div>
          <div class="quiz-mini">${done ? rooflineSvg({ hw: q.hw, IAR: q.IAR, ITD: q.ITD, W: Math.min(340, widthOf(box)), H: 220, mini: true }) : ''}</div>
        </div>
        <div class="quiz-nav">
          <button type="button" class="fig-btn" data-n="-1" ${st.k ? '' : 'disabled'}>Previous</button>
          <button type="button" class="fig-btn ${done ? 'primary' : ''}" data-n="1" ${st.k < data.length - 1 ? '' : 'disabled'}>Next</button>
          <span class="quiz-dots">${data.map((d, i) => `<span class="${st.answers[i] === undefined ? '' : st.answers[i] === (d.gain > 1) ? 'ok' : 'no'} ${i === st.k ? 'cur' : ''}"></span>`).join('')}</span>
          ${st.answers.filter((a) => a !== undefined).length === data.length ? '<button type="button" class="fig-btn" data-r="1">Start again</button>' : ''}
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
  // Each card with a .card-more panel opens in place, across the full row; its slider, if any, is
  // built on first opening.
  const CTX4 = [...CONTEXTS, 2 << 20, 4 << 20];
  const BAREC = (m, w = W0) => M.compact({ ...m, V: 0, fSC: 0 }, w, { ...O0, bl: 0 }, hwOf('h200'));
  const canvasCache = new Map();
  const MICRO = {
    omega: () => ({ label: 'denoising passes <i>S</i>', values: Array.from({ length: 32 }, (_, i) => i + 1), i: 13, show: String,
      read: (S) => { const x = boe(hwOf('h200'), O0, { ...W0, S }); return `ω = ${S + 1}; Proposition 1 on an H200: Γ = <b>${g2(x.gain(1, 1024))}</b> at batch 1, <b>${g2(x.gain(8, 1024))}</b> at batch 8.`; } }),
    canvas: () => ({ label: 'canvas length <i>b</i>', values: [16, 32, 64, 128, 256, 512, 1024], i: 4, show: String,
      read: (b) => {
        if (!canvasCache.has(b)) {
          const hw = hwOf('h200'); let tdc = null, best = { g: 0, B: 1 };
          for (let B = 1; B <= 256; B++) { const r = steady(DG, hw, B, 1024, O0, { b }); if (tdc === null && r.tdCompute) tdc = B; if (r.gain > best.g) best = { g: r.gain, B }; }
          canvasCache.set(b, { tdc, best });
        }
        const { tdc, best } = canvasCache.get(b);
        return `TPF = ${b}/15 = <b>${fmt(b / 15)}</b>; a TD pass is compute-bound from batch <b>${tdc ?? '> 256'}</b>; best Γ = <b>${g2(best.g)}</b> at batch ${best.B} (exact, H200, context 1k).`;
      } }),
    lin: () => ({ label: 'batch <i>B</i>', values: BATCHES, i: 0, show: String,
      read: (B) => `AR touches ${fmt(M.distinctExperts(DG, B))} of 128 experts per layer, a canvas ${fmt(M.distinctExperts(DG, B * W0.b))}: ${hsub('Π', 'lin')} = <b>${fmt(BAREC(DG).Pi(B))}</b>.` }),
    attn: () => ({ label: 'context <i>C</i>', values: CTX4, i: 7, show: kctx,
      read: (C) => { const c = BAREC(M.MODELS.dgemma_sparse.build()); return `With sparse global layers (<i>m</i> = 2048), a canvas reads the union of ${X.comma(c.unionRead(C, W0.b))} cached tokens per sparse layer where one query reads ${X.comma(Math.min(C, c.msel))}; over all attention layers ${hsub('Π', 'attn')} = <b>${fmt(c.qkvTD(C) / c.qkv(C))}</b>.`; } }),
    batch: () => ({ label: 'batch <i>B</i>', values: BATCHES, i: 0, show: String,
      read: (B) => { const hw = hwOf('h200'), I = intensities(DG, hw, B, 1024), r = steady(DG, hw, B, 1024); return `${hsub('I', 'AR')} = <b>${fmt(I.IAR)}</b>, ${hsub('I', 'TD')} = <b>${fmt(I.ITD)}</b> (<i>I</i>* = 206): TD ${r.tdCompute ? 'compute' : 'memory'}-bound, Γ = <b>${g2(r.gain)}</b> (exact, context 1k).`; } }),
    context: () => ({ label: 'context <i>C</i>', values: CONTEXTS, i: 0, show: kctx,
      read: (C) => { const hw = hwOf('h200'), I = intensities(DG, hw, 64, C), r = steady(DG, hw, 64, C); return `At batch 64 the cache is <b>${pct(r.arKVshare)}</b> of an AR step's bytes; ${hsub('I', 'AR')} = <b>${fmt(I.IAR)}</b>, ${hsub('I', 'TD')} = <b>${fmt(I.ITD)}</b>; Γ = <b>${g2(r.gain)}</b>.`; } }),
  };
  function initMicro(box) {
    const spec = MICRO[box.dataset.micro](), out = el('<div class="readout"></div>');
    const sl = slider(spec.label, spec.values, spec.i, spec.show);
    const sync = () => { out.innerHTML = spec.read(sl.value); };
    sl.input.addEventListener('input', sync);
    const row = el('<div class="fig-controls micro"></div>'); row.appendChild(sl.lab);
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
  const FIGS = { quiz: figQuiz, reuse: figReuse, roofline: figRoofline, bestbatch: figBestBatch, context: figContext, accel: figAccel };
  document.querySelectorAll('[data-fig]').forEach((root) => {
    try { FIGS[root.dataset.fig](root); } catch (e) { root.textContent = 'This figure failed to load.'; console.error(e); }
  });
  const redraw = () => rerenders.forEach((f) => f());
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
