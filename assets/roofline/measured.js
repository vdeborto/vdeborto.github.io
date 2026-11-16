// Interactive charts of the measurements of Parts 8-11 (DiffusionGemma against Gemma 4 26B-A4B on
// one H200, vLLM, FP8). Every value comes from the JSON that bin/build-roofline.py writes into the
// page (#bdr-practice), itself exported from the campaign data by roofscope's pages/book/export.py.
// Nothing is computed here beyond ratios of those values.
//
// Mounts on [data-mfig]: ladder (data-series="a,b,c" or "c,d"), eff, comp (data-src="eta2" or
// "eta3"), recover, vs2, chip (Part 10: the H200 and the two kernels).
(function () {
  'use strict';
  const node = document.getElementById('bdr-practice');
  if (!node) return;
  let D;
  try { D = JSON.parse(node.textContent); } catch (e) { return; }
  const Bs = D.Bs, Cs = D.Cs;

  // ---------------------------------------------------------------- helpers
  const css = (v) => getComputedStyle(document.querySelector('.bdr') || document.documentElement).getPropertyValue(v).trim();
  const colors = () => ({
    ar: css('--bdr-ar'), td: css('--bdr-td'), meas: css('--bdr-meas'), ink: css('--ink'), muted: css('--bdr-muted'), rule: css('--rule'),
    bg: css('--bdr-bg'), surface: css('--bdr-surface'), hi: css('--bdr-hi'), ink2: css('--bdr-ink-2'),
  });
  const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const kc = (C) => `${C / 1024}k`;
  const f2 = (x) => (x == null ? '–' : x.toFixed(2));
  const pc = (x) => `${Math.round(100 * x)}%`;
  const key = (B, C) => `${B},${C}`;
  const widthOf = (box) => Math.max(320, Math.min(760, Math.round(box.clientWidth || 640)));
  const redraws = [];
  function watch(box, draw) {
    let last = box.clientWidth;
    new ResizeObserver(() => { const w = box.clientWidth; if (Math.abs(w - last) > 4) { last = w; draw(); } }).observe(box);
    redraws.push(draw);
  }
  // A captioned button group: role=group named by its caption (a <label> would forward clicks to its first button).
  let groupId = 0;
  function seg(label, options, current, onPick) {
    const id = `bdr-mgrp-${++groupId}`;
    const lab = el(`<div class="ctl wide" role="group" aria-labelledby="${id}"><span id="${id}">${label}</span><span class="segbox"><span class="seg"></span></span></div>`);
    const box = lab.querySelector('.seg');
    for (const [value, text] of options) {
      const b = el(`<button type="button" aria-pressed="${value === current}">${text}</button>`);
      b.addEventListener('click', () => { box.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); onPick(value); });
      box.appendChild(b);
    }
    return lab;
  }
  // Spread label y positions so that no two are closer than `gap` (times the number of lines of the
  // upper label, `lines[i]`), keeping their order.
  function spread(ys, gap, lo, hi, lines = []) {
    const idx = ys.map((y, i) => [y, i]).sort((a, b) => a[0] - b[0]);
    const out = idx.map(([y]) => y), h = (k) => gap * (lines[idx[k][1]] || 1);
    for (let k = 1; k < out.length; k++) out[k] = Math.max(out[k], out[k - 1] + h(k - 1));
    const last = out.length - 1;
    if (out.length && out[last] + gap * ((lines[idx[last][1]] || 1) - 1) > hi) { out[last] = hi - gap * ((lines[idx[last][1]] || 1) - 1); for (let k = last - 1; k >= 0; k--) out[k] = Math.min(out[k], out[k + 1] - h(k)); }
    if (out.length && out[0] < lo) { const s = lo - out[0]; for (let k = 0; k < out.length; k++) out[k] += s; }
    const res = []; idx.forEach(([, i], k) => { res[i] = out[k]; });
    return res;
  }

  // Line chart against the measured batch sizes, which are powers of two and drawn equally spaced.
  // series: [{ label, color, width, dash, dots, ys: [v|null per batch], tag }]
  // opt: { W, H, y0, y1, ticks, yFmt, yTitle, ref: { y, label }, bands: [{ ys0, ys1, color }], hover(i) -> html, out }
  function lineChart(pic, series, opt) {
    const c = colors(), W = opt.W, H = opt.H || 300;
    // narrow screens: the series names go in a legend under the chart instead of the right margin
    const narrow = W < 520;
    const labW = narrow ? 0 : Math.min(204, Math.max(92, ...series.map((s) => 7.4 * Math.max(...s.label.split('\n').map((t) => t.length)) + 14)));
    const m = { l: 46, r: narrow ? 12 : labW + 8, t: 30, b: 42 };
    const n = Bs.length, X = (i) => m.l + (i / (n - 1)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - opt.y0) / (opt.y1 - opt.y0)) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opt.aria || ''}">`;
    for (const t of opt.ticks) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="${c.rule}"/><text x="${m.l - 7}" y="${Y(t) + 4}" text-anchor="end">${opt.yFmt(t)}</text>`;
    Bs.forEach((B, i) => { s += `<text x="${X(i)}" y="${H - m.b + 16}" text-anchor="middle">${B}</text>`; });
    s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" class="ink">batch B (log scale)</text>`;
    s += `<text x="4" y="14" class="ink">${opt.yTitle}</text>`;
    for (const bd of opt.bands || []) {
      // fill between two series, split where they cross, coloured by which one is above
      for (let i = 0; i < n - 1; i++) {
        const a0 = bd.ys0[i], a1 = bd.ys0[i + 1], b0 = bd.ys1[i], b1 = bd.ys1[i + 1];
        if ([a0, a1, b0, b1].some((v) => v == null)) continue;
        const d0 = a0 - b0, d1 = a1 - b1;
        const poly = (pts, col) => { s += `<polygon points="${pts.map(([x, y]) => `${x},${y}`).join(' ')}" fill="${col}" fill-opacity="0.16"/>`; };
        if (d0 * d1 >= 0) poly([[X(i), Y(a0)], [X(i + 1), Y(a1)], [X(i + 1), Y(b1)], [X(i), Y(b0)]], (d0 + d1) >= 0 ? bd.up : bd.down);
        else {
          const t = d0 / (d0 - d1), xm = X(i) + t * (X(i + 1) - X(i)), ym = Y(a0 + t * (a1 - a0));
          poly([[X(i), Y(a0)], [xm, ym], [X(i), Y(b0)]], d0 > 0 ? bd.up : bd.down);
          poly([[xm, ym], [X(i + 1), Y(a1)], [X(i + 1), Y(b1)]], d1 > 0 ? bd.up : bd.down);
        }
      }
    }
    for (const ar of opt.areas || []) {
      // a shaded band between lo and hi, broken where either is missing
      let seg = [];
      const flushA = () => { if (seg.length > 1) s += `<polygon points="${[...seg.map(([i, l]) => `${X(i)},${Y(l)}`), ...seg.slice().reverse().map(([i, , h]) => `${X(i)},${Y(h)}`)].join(' ')}" fill="${ar.color}" fill-opacity="0.15"/>`; seg = []; };
      ar.lo.forEach((l, i) => { const h = ar.hi[i]; if (l == null || h == null) flushA(); else seg.push([i, l, h]); });
      flushA();
    }
    if (opt.ref) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(opt.ref.y)}" y2="${Y(opt.ref.y)}" stroke="${c.ink2}" stroke-width="1.3" stroke-dasharray="${opt.ref.dash || ''}"/>`;
    const ends = [];
    for (const sr of series) {
      const pts = sr.ys.map((v, i) => (v == null ? null : [X(i), Y(v)]));
      let run = [];
      const flush = () => { if (run.length > 1) s += `<polyline fill="none" stroke="${sr.color}" stroke-width="${sr.width || 2}" stroke-linejoin="round" stroke-linecap="round" ${sr.dash ? `stroke-dasharray="${sr.dash}"` : ''} stroke-opacity="${sr.op ?? 1}" points="${run.join(' ')}"/>`; run = []; };
      pts.forEach((p) => { if (p) run.push(`${p[0]},${p[1]}`); else flush(); });
      flush();
      if (sr.dots) pts.forEach((p) => { if (p) s += `<circle cx="${p[0]}" cy="${p[1]}" r="${sr.width > 2 ? 3.6 : 3}" fill="${sr.color}" stroke="${c.bg}" stroke-width="1.5"/>`; });
      sr.pts = pts;
      const last = [...pts].reverse().find((p) => p);
      if (last && sr.label) ends.push({ x: last[0], y: last[1], sr });
    }
    // the reference line's label goes last, over the curves, at the right end
    if (opt.ref) s += opt.ref.at === 'right' ? `<text x="${W - m.r - 4}" y="${Y(opt.ref.y) - 6}" text-anchor="end" class="halo">${opt.ref.label}</text>` : `<text x="${m.l + 6}" y="${Y(opt.ref.y) - 6}" class="halo">${opt.ref.label}</text>`;
    // the selected batch: an enlarged point on every highlighted series
    s += '<g class="sel" pointer-events="none">';
    series.forEach((sr) => { if (sr.group) return; sr.pts.forEach((p, i) => { if (p) s += `<circle data-i="${i}" cx="${p[0]}" cy="${p[1]}" r="5.5" fill="${sr.color}" stroke="${c.bg}" stroke-width="2" visibility="hidden"/>`; }); });
    s += '</g>';
    const lab = ends.filter((e) => !e.sr.group), nl = lab.map((e) => e.sr.label.split('\n').length);
    const ly = spread(lab.map((e) => e.y + 4), 15, m.t + 6, H - m.b, nl);
    if (!narrow) lab.forEach((e, k) => {
      s += `<line x1="${e.x + 4}" x2="${W - m.r + 4}" y1="${e.y}" y2="${ly[k] - 4}" stroke="${e.sr.color}" stroke-opacity="0.35" stroke-dasharray="1.5 3"/>`;
      const tl = e.sr.label.split('\n').map((t, j) => `<tspan x="${W - m.r + 8}" dy="${j ? 15 : 0}">${t}</tspan>`).join('');
      s += `<text x="${W - m.r + 8}" y="${ly[k]}" style="fill:${e.sr.labelColor || e.sr.color};font-weight:${e.sr.bold ? 600 : 400}">${tl}</text>`;
    });
    // hover columns
    s += '<g class="hov">';
    Bs.forEach((B, i) => {
      const x0 = i === 0 ? m.l - 10 : (X(i - 1) + X(i)) / 2, x1 = i === n - 1 ? W - m.r + 4 : (X(i) + X(i + 1)) / 2;
      s += `<rect data-i="${i}" x="${x0}" y="${m.t}" width="${x1 - x0}" height="${H - m.b - m.t}" fill="transparent"/>`;
    });
    s += `</g><line class="cross" pointer-events="none" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="${c.ink2}" stroke-width="1" visibility="hidden"/>`;
    const leg = []; const seen = new Set();
    for (const e of [...ends].reverse()) {
      if (e.sr.group) { if (seen.has(e.sr.group)) continue; seen.add(e.sr.group); leg.push({ sr: { ...e.sr, label: e.sr.group } }); } else leg.push(e);
    }
    const legItems = narrow ? leg : leg.filter((e) => series.some((sr) => sr.group === e.sr.label));
    const legend = legItems.length ? `<div class="mlegend">${legItems.map((e) => `<span><svg width="22" height="10" aria-hidden="true"><line x1="1" x2="21" y1="5" y2="5" stroke="${e.sr.color}" stroke-width="${Math.min(3, e.sr.width || 2)}" ${e.sr.dash ? `stroke-dasharray="${e.sr.dash}"` : ''}/></svg>${e.sr.label.replace('\n', ' ')}</span>`).join('')}</div>` : '';
    pic.innerHTML = `${s}</svg>${legend}`;
    const svg = pic.querySelector('svg'), cross = svg.querySelector('.cross');
    const show = (i) => {
      pic.dataset.sel = i;
      cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('visibility', 'visible');
      svg.querySelectorAll('.sel circle').forEach((c0) => c0.setAttribute('visibility', +c0.dataset.i === i ? 'visible' : 'hidden'));
      if (opt.out && opt.hover) opt.out.innerHTML = opt.hover(i);
    };
    svg.querySelectorAll('.hov rect').forEach((r) => {
      r.addEventListener('pointerenter', () => show(+r.dataset.i));
      r.addEventListener('pointerdown', () => show(+r.dataset.i));
    });
    // keyboard: focus the chart, then the arrow keys step through the batches
    let ki = pic.dataset.sel != null ? +pic.dataset.sel : 0;
    svg.setAttribute('tabindex', '0');
    svg.addEventListener('keydown', (ev) => {
      const d = { ArrowRight: 1, ArrowLeft: -1, Home: -n, End: n }[ev.key];
      if (!d) return;
      ev.preventDefault();
      ki = Math.max(0, Math.min(n - 1, +(pic.dataset.sel || 0) + d));
      show(ki);
    });
    // the selection persists: it starts at the first batch and stays where the reader leaves it
    const first = series.find((sr) => !sr.group) ? Math.max(0, series.find((sr) => !sr.group).pts.findIndex((p) => p)) : 0;
    // a kept selection that falls on a batch not measured here moves to the nearest measured one
    const main = series.filter((sr) => !sr.group);
    const has = (i) => main.some((sr) => sr.pts[i]);
    let want = pic.dataset.sel != null && pic.dataset.sel !== '' ? Math.min(n - 1, +pic.dataset.sel) : first;
    if (!has(want)) { for (let d = 1; d < n; d++) { if (want - d >= 0 && has(want - d)) { want -= d; break; } if (want + d < n && has(want + d)) { want += d; break; } } }
    show(want);
  }
  // a readout: a heading and a short list of label, value pairs
  const ro = (head, items) => `<div class="ro-h">${head}</div><div class="ro-g">${items.map(([l, v, cls]) => `<span class="ro-l${cls ? ` ${cls}` : ''}">${l}</span><span class="ro-v${cls ? ` ${cls}` : ''}">${v}</span>`).join('')}</div>`;
  const frame = (root) => {
    const ctl = el('<div class="fig-controls mctl"></div>'), pic = el('<div class="mchart"></div>'), out = el('<div class="readout" aria-live="polite"></div>');
    root.append(ctl, pic, out);
    return { ctl, pic, out };
  };
  const contextSeg = (st, draw) => seg('context C', Cs.map((C) => [C, kc(C)]), st.C, (v) => { st.C = v; draw(); });

  // ---------------------------------------------------------------- the gain, from the book's bound to the measurement
  const LADDER = {
    a: { k: 'roof_ind_ds', label: '(a) datasheet rates,\nindependent routing', long: '(a) datasheet rates, independent routing', dash: '2 4', w: 1.8, col: 'muted' },
    b: { k: 'roof_ind', label: '(b) empirical ceilings', long: '(b) empirical ceilings, independent routing', dash: '7 4', w: 1.8, col: 'muted' },
    c: { k: 'roof', label: '(c) empirical ceilings,\nmeasured routing', long: '(c) empirical ceilings and measured routing', dash: '', w: 2.2, col: 'ink' },
    d: { k: 'measured', label: '(d) measured', long: '(d) measured in vLLM', dash: '', w: 2.8, col: 'meas', dots: true },
  };
  function mLadder(root) {
    const keys = (root.dataset.series || 'a,b,c,d').split(',');
    const st = { C: Cs[0] };
    const { ctl, pic, out } = frame(root);
    ctl.append(contextSeg(st, () => draw()));
    const all = Object.values(D.gains).flatMap((r) => keys.map((k) => r[LADDER[k].k])).filter((v) => v != null);
    const y1 = Math.ceil(Math.max(...all) + 0.4);
    function draw() {
      const c = colors();
      const series = keys.map((k) => {
        const L = LADDER[k];
        return { label: L.label, color: c[L.col], labelColor: L.col === 'muted' ? c.ink2 : undefined, width: L.w, dash: L.dash, dots: L.dots || k === 'c', bold: k === 'd' || k === 'c',
          ys: Bs.map((B) => (D.gains[key(B, st.C)] || {})[L.k] ?? null) };
      });
      lineChart(pic, series, {
        W: widthOf(pic), H: 310, y0: 0, y1, ticks: Array.from({ length: y1 + 1 }, (_, i) => i), yFmt: (v) => `${v}×`, yTitle: 'gain Γ, TD over AR',
        ref: { y: 1, label: 'parity: below it AR is faster' }, out, aria: `Gain against batch at context ${kc(st.C)}`,
        hover: (i) => {
          const r = D.gains[key(Bs[i], st.C)], head = `B = ${Bs[i]}, C = ${kc(st.C)}`;
          if (!r) return ro(head, [['not measured', 'the batch did not fit in memory']]);
          return ro(head, keys.map((k) => [LADDER[k].long, `${f2(r[LADDER[k].k])}×`, k === 'd' ? 'meas' : '']));
        },
      });
    }
    draw(); watch(pic, draw);
  }

  // ---------------------------------------------------------------- Proposition 9: the two efficiencies
  function mEff(root) {
    const st = { C: Cs[0] };
    const { ctl, pic, out } = frame(root);
    ctl.append(contextSeg(st, () => draw()));
    function draw() {
      const c = colors(), g = (B) => D.gains[key(B, st.C)] || {};
      const td = Bs.map((B) => g(B).eta_td ?? null), ar = Bs.map((B) => g(B).eta_ar ?? null);
      lineChart(pic, [
        { label: 'TD', color: c.td, width: 2.6, dots: true, bold: true, ys: td },
        { label: 'AR', color: c.ar, width: 2.6, dots: true, bold: true, ys: ar },
      ], {
        W: widthOf(pic), H: 300, y0: 0, y1: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], yFmt: (v) => v.toFixed(1), yTitle: 'efficiency η = roofline / measured time',
        bands: [{ ys0: td, ys1: ar, up: c.td, down: c.ar }], out, aria: `Step efficiencies of TD and AR against batch at context ${kc(st.C)}`,
        hover: (i) => {
          const r = D.gains[key(Bs[i], st.C)], head = `B = ${Bs[i]}, C = ${kc(st.C)}`;
          if (!r) return ro(head, [['not measured', '']]);
          const q = r.eta_td / r.eta_ar;
          return ro(head, [['TD step efficiency η<sub>TD</sub>', f2(r.eta_td), 'td'], ['AR step efficiency η<sub>AR</sub>', f2(r.eta_ar), 'ar'],
            ['measured gain / bound = η<sub>TD</sub>/η<sub>AR</sub>', `${f2(q)}: the bound is ${q >= 1 ? 'conservative' : 'optimistic'} here (${f2(r.measured)} against ${f2(r.roof)})`]]);
        },
      });
    }
    draw(); watch(pic, draw);
  }

  // ---------------------------------------------------------------- efficiency of each component
  const COMPS = [['experts', 'experts'], ['attnG', 'global attention'], ['dense', 'dense GEMMs and router'], ['attnW', 'window attention'], ['head', 'head, sampler, self-cond.']];
  // target efficiency of each component, Definition 7
  const TARGET = (dec, comp) => (comp === 'experts' ? (dec === 'td' ? 0.81 : 0.58) : comp === 'attnG' && dec === 'td' ? 0.73 : 0.8);
  function mComp(root) {
    const src0 = root.dataset.src || 'eta3';
    const st = { C: Cs[0], focus: 'experts', mode: src0 === 'eta2' ? 'td2' : 'td3' };
    const MODES = { td3: ['eta3', 'td', 'TD, new kernels'], td2: ['eta2', 'td', 'TD, vLLM kernels'], ar: ['eta3', 'ar', 'AR'] };
    const { ctl, pic, out } = frame(root);
    if (src0 !== 'eta2') ctl.append(seg('decoder', Object.entries(MODES).map(([k, v]) => [k, v[2]]), st.mode, (v) => { st.mode = v; draw(); }));
    ctl.append(seg('highlight', COMPS.map(([k, n]) => [k, n.replace(', sampler, self-cond.', ' and sampler')]), st.focus, (v) => { st.focus = v; draw(); }));
    ctl.append(contextSeg(st, () => draw()));
    function draw() {
      const c = colors(), [src, m, name] = MODES[st.mode];
      const val = (B, comp) => { const r = (D[src][m] || {})[key(B, st.C)]; return r ? r[comp] ?? null : null; };
      const series = COMPS.filter(([k]) => k !== st.focus).map(([k, n]) => ({ label: n, color: c.rule, labelColor: c.muted, width: 1.6, ys: Bs.map((B) => val(B, k)) }));
      series.push({ label: 'whole step', color: c.ink2, width: 1.8, dash: '6 4', ys: Bs.map((B) => val(B, 'step')) });
      const fn = COMPS.find(([k]) => k === st.focus)[1], tgt = TARGET(m, st.focus);
      series.push({ label: fn, color: m === 'td' ? c.td : c.ar, width: 3, dots: true, bold: true, ys: Bs.map((B) => val(B, st.focus)) });
      lineChart(pic, series, {
        W: widthOf(pic), H: 310, y0: 0, y1: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], yFmt: (v) => v.toFixed(1), yTitle: 'efficiency η = roofline / measured time',
        ref: { y: tgt, label: `target η* = ${tgt.toFixed(2)} (Definition 7)`, dash: '3 3', at: 'right' }, out, aria: `${name}: efficiency of each component against batch`,
        hover: (i) => {
          const r = (D[src][m] || {})[key(Bs[i], st.C)], head = `${name}, B = ${Bs[i]}, C = ${kc(st.C)}`;
          if (!r) return ro(head, [['not measured', '']]);
          const parts = COMPS.map(([k, n]) => [n, r[k], k === st.focus]).filter(([, v]) => v != null).sort((a, b) => a[1] - b[1]);
          return ro(head, [['whole step', f2(r.step)], ...parts.map(([n, v, on]) => [n.replace('head, sampler, self-cond.', 'vocabulary head, sampler and self-conditioning'), f2(v), on ? 'hi' : '']), [`target of ${fn} (Definition 7)`, tgt.toFixed(2)]]);
        },
      });
    }
    draw(); watch(pic, draw);
  }

  // ---------------------------------------------------------------- where the remaining time goes
  const RCOMPS = [['experts', 'experts'], ['outside', 'outside the model'], ['head', 'head, sampler, self-cond.'], ['dense', 'dense GEMMs and router'], ['attnG', 'global attention'], ['attnW', 'window attention'], ['idle', 'idle']];
  function mRecover(root) {
    const st = { C: Cs[0], B: 8 };
    const { ctl, pic, out } = frame(root);
    const bseg = () => seg('batch B', Bs.filter((B) => D.share.td[key(B, st.C)]).map((B) => [B, String(B)]), st.B, (v) => { st.B = v; draw(); });
    let bl = bseg();
    ctl.append(contextSeg(st, () => { if (!D.share.td[key(st.B, st.C)]) st.B = Bs.filter((B) => D.share.td[key(B, st.C)]).pop(); const nb = bseg(); ctl.replaceChild(nb, bl); bl = nb; draw(); }), bl);
    function draw() {
      const c = colors(), W = widthOf(pic), sh = D.share.td[key(st.B, st.C)], rc = D.recoverable.td[key(st.B, st.C)];
      const rows = RCOMPS.map(([k, n]) => ({ k, n, s: sh[k] || 0, r: rc[k] || 0 })).sort((a, b) => b.r - a.r || b.s - a.s);
      const narrow = W < 520, lw = narrow ? 128 : 200, m = { l: lw, r: 44, t: 22, b: 30 }, rh = 30, H = m.t + rows.length * rh + m.b;
      if (narrow) rows.forEach((x) => { x.n = x.n.replace('head, sampler, self-cond.', 'head, sampler…').replace('dense GEMMs and router', 'dense, router').replace('outside the model', 'outside model').replace('window attention', 'window attn.').replace('global attention', 'global attn.'); });
      const xmax = Math.max(0.2, Math.ceil(Math.max(...rows.map((x) => x.s)) * 10) / 10), X = (v) => m.l + (v / xmax) * (W - m.l - m.r);
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of TD's step by component, recoverable part highlighted">`;
      for (let t = 0; t <= xmax + 1e-9; t += narrow && xmax > 0.25 ? 0.2 : 0.1) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${m.t - 4}" y2="${H - m.b}" stroke="${c.rule}"/><text x="${X(t)}" y="${H - m.b + 15}" text-anchor="middle">${pc(t)}</text>`;
      s += `<text x="4" y="${m.t - 9}" class="ink">share of TD's step, B = ${st.B}, C = ${kc(st.C)}</text>`;
      rows.forEach((x, i) => {
        const y = m.t + i * rh + 5, h = rh - 10, keep = Math.max(0, x.s - x.r);
        s += `<text x="${m.l - 8}" y="${y + h / 2 + 4}" text-anchor="end" class="ink">${x.n}</text>`;
        s += `<rect x="${X(0)}" y="${y}" width="${X(keep) - X(0)}" height="${h}" fill="${c.muted}" fill-opacity="0.35"><title>${x.n}: at target ${pc(keep)}</title></rect>`;
        s += `<rect x="${X(keep)}" y="${y}" width="${X(x.s) - X(keep)}" height="${h}" fill="${c.td}"><title>${x.n}: recoverable ${pc(x.r)}</title></rect>`;
        s += `<text x="${X(x.s) + 6}" y="${y + h / 2 + 4}" class="${x.r >= 0.05 ? 'strong' : ''}">${pc(x.r)}</text>`;
      });
      pic.innerHTML = `${s}</svg>`;
      const top = rows[0], tot = rows.reduce((a, x) => a + x.r, 0);
      out.innerHTML = ro(`B = ${st.B}, C = ${kc(st.C)}`, [['largest recoverable share', `${top.n}, ${pc(top.r)} of the step`, 'td'], ['all components together', `${pc(tot)} of the step`]]);
    }
    draw(); watch(pic, draw);
  }

  // ---------------------------------------------------------------- what the two kernels change
  function mVs2(root) {
    const { pic, out } = frame(root);
    function draw() {
      const c = colors(), op = [0.35, 0.55, 0.78, 1];
      const series = Cs.map((C, j) => ({ label: `context ${kc(C)}`, color: c.td, width: 1.5 + 0.5 * j, dots: true, bold: j === Cs.length - 1,
        ys: Bs.map((B) => (D.vs2.td[key(B, C)] || {}).ratio ?? null), op: op[j] }));
      lineChart(pic, series, {
        W: widthOf(pic), H: 290, y0: 1, y1: 1.3, ticks: [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3], yFmt: (v) => `${v.toFixed(2)}×`, yTitle: 'throughput, new kernels / vLLM\'s',
        out, aria: 'TD speed-up from the two kernels against batch, one line per context',
        hover: (i) => ro(`B = ${Bs[i]}: new kernels over vLLM's`, Cs.map((C) => { const r = D.vs2.td[key(Bs[i], C)]; return [`context ${kc(C)}`, r ? `${f2(r.ratio)}×` : 'not measured']; })),
      });
    }
    draw(); watch(pic, draw);
  }

  // ---------------------------------------------------------------- Proposition 1 against the measurement
  function mProp1(root) {
    const st = { C: Cs[0] };
    const { ctl, pic, out } = frame(root);
    ctl.append(contextSeg(st, () => draw()));
    function draw() {
      const c = colors(), p = (B) => D.prop1[key(B, st.C)] || {}, g = (B) => D.gains[key(B, st.C)] || {};
      lineChart(pic, [
        { label: 'Prop. 1,\nindependent routing', color: c.muted, labelColor: c.ink2, width: 1.8, dash: '6 4', ys: Bs.map((B) => p(B).gain_ind ?? null) },
        { label: 'Prop. 1,\nmeasured routing', color: c.ink2, width: 2, dash: '2 3', bold: true, ys: Bs.map((B) => p(B).gain_meas ?? null) },
        { label: 'measured', color: c.meas, width: 2.8, dots: true, bold: true, ys: Bs.map((B) => (D.gains[key(B, st.C)] ? g(B).measured : null)) },
      ], {
        W: widthOf(pic), H: 300, y0: 0, y1: 8, ticks: [0, 1, 2, 3, 4, 5, 6, 7, 8], yFmt: (v) => `${v}×`, yTitle: 'gain Γ, TD over AR',
        ref: { y: 1, label: 'parity: below it AR is faster' }, out, aria: `Proposition 1 and the measured gain against batch at context ${kc(st.C)}`,
        hover: (i) => {
          const q = D.prop1[key(Bs[i], st.C)], r = D.gains[key(Bs[i], st.C)], head = `B = ${Bs[i]}, C = ${kc(st.C)}`;
          if (!q || !r) return ro(head, [['not measured', '']]);
          return ro(head, [['Proposition 1, independent routing (A5)', `${f2(q.gain_ind)}× (Π<sub>lin</sub> = ${f2(q.Pi_ind)})`], ['Proposition 1, measured routing', `${f2(q.gain_meas)}× (Π<sub>lin</sub> = ${f2(q.Pi_meas)})`], ['measured', `${f2(r.measured)}×`, 'meas']]);
        },
      });
    }
    draw(); watch(pic, draw);
  }

  // ---------------------------------------------------------------- the experts a pass touches
  function mRouting(root) {
    const st = { C: Cs[0], m: 'td' };
    const { ctl, pic, out } = frame(root);
    ctl.append(seg('decoder', [['td', 'TD: 256 positions per sequence'], ['ar', 'AR: one token per sequence']], st.m, (v) => { st.m = v; draw(); }), contextSeg(st, () => draw()));
    function draw() {
      const c = colors(), R = D.routing[st.m], v = (B, k) => (R[key(B, st.C)] || {})[k] ?? null, col = st.m === 'td' ? c.td : c.ar;
      lineChart(pic, [
        { label: 'independent, uniform', color: c.muted, labelColor: c.ink2, width: 1.8, dash: '2 4', ys: Bs.map((B) => v(B, 'uni')) },
        { label: 'independent,\nmeasured popularity', color: c.muted, labelColor: c.ink2, width: 1.8, dash: '7 4', ys: Bs.map((B) => v(B, 'pop')) },
        { label: `${st.m.toUpperCase()}, measured`, color: col, width: 2.8, dots: true, bold: true, ys: Bs.map((B) => v(B, 'meas')) },
      ], {
        W: widthOf(pic), H: 300, y0: 0, y1: 128, ticks: [0, 32, 64, 96, 128], yFmt: (x) => String(x), yTitle: 'distinct experts per layer and pass, U (of 128)',
        areas: [{ lo: Bs.map((B) => v(B, 'meas_lo')), hi: Bs.map((B) => v(B, 'meas_hi')), color: col }], out,
        aria: `Distinct experts per layer against batch, ${st.m.toUpperCase()}, context ${kc(st.C)}`,
        hover: (i) => {
          const r = R[key(Bs[i], st.C)], head = `${st.m.toUpperCase()}, B = ${Bs[i]}, C = ${kc(st.C)}`;
          if (!r) return ro(head, [['not measured', '']]);
          return ro(`${head}: ${Math.round(r.N)} tokens per pass`, [['measured U', `${r.meas.toFixed(1)} (repeats ${r.meas_lo.toFixed(0)}–${r.meas_hi.toFixed(0)})`, st.m],
            ['as many as', `${r.neff != null ? r.neff.toFixed(1) : '–'} independent uniform tokens`], ['independent, measured popularity', r.pop.toFixed(1)], ['independent, uniform', r.uni.toFixed(1)]]);
        },
      });
    }
    draw(); watch(pic, draw);
  }

  // ---------------------------------------------------------------- the chip, and what the two kernels of Part 10 change
  // Written for readers who do not write kernels. Every number is stated in Part 10 (or, for the chip,
  // elsewhere in the text): the bytes each expert kernel writes per MoE layer at TD batch 32 (Figure 35),
  // shared memory per thread block (Figure 38), the busy share of FlashAttention 4's tensor cores, and
  // the per-tile timings of the pair kernel's exchange. Totals only add the sizes of Figure 35.
  // rows: [kernel, how many kernels, MB written to HBM (null: no size given; 'L2': summed on the chip), what it does]
  const CHIP = {
    experts: {
      old: { name: "vLLM's kernels", short: 'vLLM', rows: [
        ['quantise x', 2, null, 'Convert the layer input <i>x</i> to 8-bit numbers (FP8), the format the matrix multiplications read. vLLM does this in two kernels; Figure 35 gives no size for their output.'],
        ['sort rows by expert', 1, null, 'Each token goes to 8 of the 128 experts. This kernel sorts the (token, expert) pairs by expert, so that the rows of one expert sit together.'],
        ['GEMM 1 (gate, up)', 1, 184, 'First matrix multiplication (GEMM): each row times its expert\'s gate and up matrices. The result leaves the chip: 184 MB written to HBM.'],
        ['act_and_mul', 1, 92, 'A separate kernel reads that result back from HBM, applies the activation gelu(gate) · up, and writes it to HBM again: 92 MB.'],
        ['quantise h', 2, 46, 'Two more kernels read the activation back, convert it to FP8 and write it again: 46 MB.'],
        ['GEMM 2, a row per pair', 1, 369, 'Second matrix multiplication, with the down matrix. It writes one output row per (token, expert) pair, so 8 rows per token: 369 MB, the largest tensor of the layer.'],
        ['moe_sum', 1, 46, 'A last kernel reads the 369 MB back from HBM only to add up each token\'s 8 rows, weighted by the router, into the output: 46 MB.'],
      ] },
      new: { name: 'the rewrite (moe_ws)', short: 'moe_ws', rows: [
        ['quantise x, per token', 1, 23, 'Convert <i>x</i> to FP8 with one scale per token, in a single kernel: 23 MB.'],
        ['sort rows by expert', 1, null, 'As before, sort the (token, expert) pairs by expert.'],
        ['GEMM 1 + activation', 1, 100, 'First matrix multiplication with the activation built in. The weights are stored once so that a tile holds gate and up columns side by side, so gelu(gate) · up is applied in the SM\'s registers before anything leaves it. Only the activation is written: 100 MB.'],
        ['quantise h, per row', 1, 50, 'Convert the activation to FP8 with one scale per row. The scale needs the largest value of the whole row, which no single tile of the GEMM sees, so this stays a separate pass: 50 MB.'],
        ['zero y', 1, null, 'Set the output <i>y</i> to zero, since the next kernel adds into it.'],
        ['GEMM 2 + add into y', 1, 'L2', 'Second matrix multiplication. Each output row is multiplied by its routing weight and added straight into its token\'s output, with atomic additions (additions that stay correct when several SMs add to the same place). A token\'s 8 rows meet in the L2 cache, so the 369 MB never reach HBM.'],
      ] },
    },
  };
  const mbOf = (rows, upto = rows.length - 1) => rows.slice(0, upto + 1).reduce((n, r) => n + (typeof r[2] === 'number' ? r[2] : 0), 0);
  const CHIP_ATTN = {
    old: { name: 'FlashAttention 4', sms: 1, segs: [['Q', 64, 'ink2'], ['K', 64, 'k'], ['V', 64, 'v'], ['P', 8, 'muted']],
      rowLabel: () => 'one thread block, all 512 columns', fits: 'Fits one stage: loads and work take turns',
      key: ['Q, K, V: tiles of 64 queries, keys and', 'values, each at all 512 columns', 'P: the 64 × 64 attention scores'] },
    new: { name: 'the pair kernel (attn512)', sms: 2, segs: [['Q', 64, 'ink2'], ['K1', 32, 'k'], ['K2', 32, 'k'], ['V1', 32, 'v'], ['V2', 32, 'v'], ['Z', 32, 'muted']],
      rowLabel: (k) => `thread block ${k + 1}, columns ${256 * k}–${256 * k + 255}`, fits: 'Fits two stages: loads overlap the work',
      key: ['Q: 128 query rows at 256 columns', 'K1, V1 and K2, V2: two buffers (stages)', 'of keys and values', 'Z: two buffers for the exchanged scores'] },
  };
  const P = (head, ...paras) => `<div class="ro-h">${head}</div>${paras.map((t) => `<p class="ro-p">${t}</p>`).join('')}`;
  const CHIP_TIPS = {
    sm: ['132 SMs (streaming multiprocessors)', 'The chip\'s processors. Each has its own tensor cores (the units that multiply matrices), 256 KB of registers and about 227 KB of shared memory, a small scratchpad next to the tensor cores. A kernel is one program, spread over all the SMs.'],
    l2: ['L2 cache', 'A memory on the chip, shared by all 132 SMs: much smaller than HBM, and faster to reach.'],
    hbm: ['HBM, the main memory', '141 GB at 4.8 TB/s. It sits next to the chip, not on it, so it is the slowest level to reach. What one kernel writes here, the next kernel has to read back.'],
    x: ['The exchange', 'The attention scores need all 512 columns, but each SM holds only 256. After every tile of keys, each SM writes its partial scores into its partner\'s shared memory and waits for the partner\'s.'],
    'tl-load': ['Loading keys and values', 'Copying the next tile of keys and values from HBM into shared memory.'],
    'tl-comp': ['Tensor cores', 'The time the tensor cores spend multiplying: the useful work.'],
    'tl-idle': ['Tensor cores idle', 'The tensor cores wait: for a load to finish (FlashAttention 4) or for the exchange (the pair kernel).'],
    'tl-x': ['The exchange', 'About 0.67 µs to send the partial scores and 0.5 µs to wait for the partner, against 1.5 µs of tensor work per tile of keys (64-key tiles, two stages).'],
  };
  function mChip(root) {
    const st = { tab: 'experts', ver: 'old', hover: null, step: null };
    const { ctl, pic, out } = frame(root);
    let timer = null;
    const steps = el('<div class="ctl wide" role="group" aria-label="Go through the kernels one at a time"><span>kernels</span><span class="segbox chip-steps"></span></div>');
    const box = steps.querySelector('.chip-steps');
    const btn = (t, aria) => { const b = el(`<button type="button" class="fig-btn"${aria ? ` aria-label="${aria}"` : ''}>${t}</button>`); box.appendChild(b); return b; };
    const allB = btn('all at once'), prevB = btn('◀', 'previous kernel'), nextB = btn('▶', 'next kernel'), playB = btn('play');
    const rowsNow = () => CHIP.experts[st.ver].rows;
    const stop = () => { if (timer) { clearInterval(timer); timer = null; } playB.textContent = 'play'; };
    const go = (d) => { const n = rowsNow().length; st.step = st.step == null ? (d > 0 ? 0 : n - 1) : Math.max(0, Math.min(n - 1, st.step + d)); st.hover = null; draw(); };
    allB.addEventListener('click', () => { stop(); st.step = null; draw(); });
    prevB.addEventListener('click', () => { stop(); go(-1); });
    nextB.addEventListener('click', () => { stop(); go(1); });
    playB.addEventListener('click', () => {
      if (timer) { stop(); return; }
      st.step = null; go(1); playB.textContent = 'pause';
      timer = setInterval(() => { if (st.step >= rowsNow().length - 1) stop(); else go(1); }, 2800);
    });
    ctl.append(seg('kernel', [['experts', 'experts (MoE)'], ['attn', 'global attention']], st.tab, (v) => { stop(); st.tab = v; st.hover = null; st.step = null; draw(); }),
      seg('version', [['old', "vLLM's kernels"], ['new', 'the rewrite']], st.ver, (v) => { stop(); st.ver = v; st.hover = null; st.step = null; draw(); }), steps);
    const T = (x, y, t, o = {}) => `<text x="${x}" y="${y}"${o.a ? ` text-anchor="${o.a}"` : ''} style="fill:${o.c || 'currentColor'};font-size:${o.s || 12}px${o.w ? ';font-weight:600' : ''}">${t}</text>`;
    const arrowDown = (x, y1, y2, w, col, op = 0.85) => `<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2 - 5}" stroke="${col}" stroke-width="${w}" stroke-opacity="${op}"/><path d="M${x - 3 - w / 2},${y2 - 7} L${x},${y2} L${x + 3 + w / 2},${y2 - 7} z" fill="${col}" fill-opacity="${op}"/>`;
    function draw() {
      const c = colors(), crit = css('--bdr-critical') || '#b3261e', green = css('--bdr-s3') || c.ink2, amber = css('--bdr-s4') || c.ink2;
      const fillOf = { ink2: c.ink2, k: green, v: amber, muted: c.muted };
      const narrow = (pic.clientWidth || 640) < 560;
      const W = narrow ? 340 : 680, chipW = narrow ? 324 : 330;
      const px = narrow ? 8 : 356, py = narrow ? 384 : 26;  // the right (or lower) panel
      const bw = W - px - 8;
      steps.hidden = st.tab !== 'experts';
      [prevB.disabled, nextB.disabled] = [st.step === 0, st.step != null && st.step >= rowsNow().length - 1];
      allB.classList.toggle('primary', st.step == null);
      let g = '', H = 0;
      // the chip
      const exNew = st.tab === 'experts' && st.ver === 'new', stepRow = st.tab === 'experts' && st.step != null ? rowsNow()[st.step] : null;
      g += `<rect x="4" y="4" width="${chipW}" height="262" rx="10" fill="${c.surface}" stroke="${c.ink}" stroke-width="1.3"/>` + T(16, 22, 'the H200 chip', { c: c.ink, w: 1 });
      const cols = 12, cw = (chipW - 28) / cols, ch = 12.5, gx = 3, gy = 3.2, x0 = 16, y0 = 32;
      const hiSM = st.tab === 'attn' ? (CHIP_ATTN[st.ver].sms === 2 ? [40, 41] : [40]) : [];
      g += `<g data-k="sm" style="cursor:help">`;
      for (let i = 0; i < 132; i++) {
        const cx = x0 + (i % cols) * cw, cy = y0 + Math.floor(i / cols) * (ch + gy), hit = hiSM.includes(i);
        const op = hit ? 0.9 : st.hover === 'sm' || stepRow ? 0.5 : 0.3;
        g += `<rect x="${cx}" y="${cy}" width="${cw - gx}" height="${ch}" rx="2" fill="${hit ? c.hi : c.muted}" fill-opacity="${op}"/>`;
      }
      if (hiSM.length === 2) g += `<rect x="${x0 + 4 * cw - 2}" y="${y0 + 3 * (ch + gy) - 2}" width="${2 * cw + 1}" height="${ch + 4}" rx="3" fill="none" stroke="${c.ink}" stroke-dasharray="3 2"/>`;
      const gridBottom = y0 + 11 * (ch + gy);
      g += T(x0, gridBottom + 11, '132 SMs: the processors', { s: 11.5, c: c.ink }) + T(x0, gridBottom + 25, 'each: tensor cores, registers, shared memory', { s: 10.5 }) + `</g>`;
      const ly = gridBottom + 34;
      const l2text = exNew && (stepRow == null || stepRow[2] === 'L2') ? 'L2: each token\'s 8 rows are added here' : 'L2 cache: on the chip, shared by all SMs';
      g += `<g data-k="l2" style="cursor:help"><rect x="${x0}" y="${ly}" width="${chipW - 28}" height="24" rx="4" fill="${c.muted}" fill-opacity="${st.hover === 'l2' || (stepRow && stepRow[2] === 'L2') ? 0.4 : 0.2}" stroke="${c.muted}"/>` + T(x0 + (chipW - 28) / 2, ly + 16, l2text, { a: 'middle', c: c.ink2, s: 11.5 }) + `</g>`;
      // HBM, under the chip
      const hy = 312;
      g += `<g data-k="hbm" style="cursor:help"><rect x="4" y="${hy}" width="${chipW}" height="40" rx="6" fill="${c.surface}" stroke="${c.ink2}" stroke-opacity="${st.hover === 'hbm' ? 1 : 0.7}"/>`
        + T(4 + chipW / 2, hy + 16, 'HBM: the main memory, off the chip', { a: 'middle', c: c.ink, s: 11.5 }) + T(4 + chipW / 2, hy + 31, '141 GB, read at 4.8 TB/s: the slowest level', { a: 'middle', c: c.ink2, s: 10.5 }) + `</g>`;
      H = hy + 46;
      if (st.tab === 'experts') {
        const v = CHIP.experts[st.ver], rows = v.rows;
        // traffic to HBM: in the overview one arrow per kernel that writes there; step by step, the current one
        const a1 = ly + 24, a2 = hy - 1;
        if (stepRow == null) {
          const writes = rows.map((r, i) => [r, i]).filter(([r]) => typeof r[2] === 'number');
          writes.forEach(([r, i], k) => {
            const ax = x0 + 24 + (k + 0.5) * ((chipW - 76) / Math.max(1, writes.length)), on = st.hover === `r${i}`;
            g += arrowDown(ax, a1, a2, Math.max(1.4, Math.sqrt(r[2]) / 2.2), on ? crit : c.ink2, on || st.hover == null ? 0.85 : 0.35);
          });
        } else if (typeof stepRow[2] === 'number') {
          const ax = 4 + chipW / 2;
          g += arrowDown(ax, a1, a2, Math.max(1.6, Math.sqrt(stepRow[2]) / 2.2), crit) + T(ax + 14, (a1 + a2) / 2 + 4, `${stepRow[2]} MB written`, { c: crit, s: 11.5, w: 1 });
        } else if (stepRow[2] === 'L2') {
          g += T(4 + chipW / 2, (a1 + a2) / 2 + 4, 'nothing written to HBM', { a: 'middle', c: c.ink2, s: 11.5 });
        }
        // the kernels, in order, with what each writes to HBM
        const rh = 30, maxMB = 369, barX = px + bw - 108;
        g += T(px, py - 8, `${v.short}: ${rows.reduce((n, r) => n + r[1], 0)} kernels`, { c: c.ink, w: 1 }) + T(barX, py - 8, 'written to HBM', { s: 11 });
        rows.forEach((r, i) => {
          const y = py + i * rh, on = st.hover === `r${i}` || st.step === i, dim = st.step != null && st.step !== i;
          g += `<g data-k="r${i}" style="cursor:pointer" opacity="${dim ? 0.45 : 1}"><rect x="${px}" y="${y}" width="${bw}" height="${rh - 4}" rx="4" fill="${on ? c.hi : 'transparent'}" fill-opacity="0.14" stroke="${st.step === i ? c.ink : c.muted}" stroke-opacity="${st.step === i ? 1 : 0.6}"/>`;
          g += T(px + 8, y + 17, `${i + 1}. ${r[0]}${r[1] > 1 ? ` (${r[1]} kernels)` : ''}`, { c: c.ink, s: 11.5 });
          if (typeof r[2] === 'number') g += `<rect x="${barX}" y="${y + 8}" width="${Math.max(2, 54 * r[2] / maxMB)}" height="10" rx="2" fill="${st.step === i ? crit : c.ink2}" fill-opacity="0.8"/>` + T(barX + 58, y + 17, `${r[2]} MB`, { c: c.ink2, s: 11 });
          else if (r[2] === 'L2') g += T(barX, y + 17, 'stays in L2', { c: c.ink2, s: 11 });
          else g += T(barX, y + 17, '–', { c: c.muted, s: 11 });
          g += `</g>`;
        });
        // running totals: bytes written to HBM by one MoE layer, both versions
        let ty = py + rows.length * rh + 14;
        const tw = bw - 150;
        const full = { old: mbOf(CHIP.experts.old.rows), new: mbOf(CHIP.experts.new.rows) }, top = Math.max(full.old, full.new);
        g += T(px, ty + 4, 'written to HBM by one MoE layer (TD batch 32)', { s: 11, c: c.ink });
        ty += 12;
        for (const k of ['old', 'new']) {
          const cur = k === st.ver, sofar = cur && st.step != null ? mbOf(CHIP.experts[k].rows, st.step) : full[k];
          g += T(px, ty + 13, k === 'old' ? "vLLM's" : 'rewrite', { s: 11, c: cur ? c.ink : c.muted });
          g += `<rect x="${px + 52}" y="${ty + 4}" width="${tw * full[k] / top}" height="11" rx="2" fill="${c.muted}" fill-opacity="0.18"/>`;
          g += `<rect x="${px + 52}" y="${ty + 4}" width="${Math.max(0.5, tw * sofar / top)}" height="11" rx="2" fill="${cur ? crit : c.muted}" fill-opacity="${cur ? 0.8 : 0.5}"/>`;
          g += T(px + 58 + tw * full[k] / top, ty + 13, cur && st.step != null ? `${sofar} / ${full[k]} MB` : `${full[k]} MB`, { s: 11, c: cur ? c.ink : c.muted });
          ty += 20;
        }
        H = Math.max(H, ty + 8);
      } else {
        const v = CHIP_ATTN[st.ver];
        // what one SM's shared memory holds (two SMs for the pair, joined by the exchange)
        const sc = (bw - 56) / 227, lim = px + 227 * sc, gapY = 78, y0b = py + 34;
        g += T(px, py - 8, 'shared memory of one SM', { c: c.ink, w: 1 });
        g += T(lim, py + 12, 'limit: 227 KB per SM', { a: 'end', c: crit, s: 10.5 });
        for (let k = 0; k < v.sms; k++) {
          const y = y0b + k * gapY;
          let x = px;
          g += T(px, y - 5, v.rowLabel(k), { s: 11 });
          for (const [nm, kb, col] of v.segs) {
            const w = kb * sc;
            g += `<rect x="${x}" y="${y}" width="${w - 1.5}" height="24" rx="3" fill="${fillOf[col]}" fill-opacity="0.8"/>`;
            g += T(x + w / 2, y + 16, nm, { a: 'middle', c: c.bg, s: 10.5 });
            x += w;
          }
          g += `<line x1="${lim}" y1="${y - 2}" x2="${lim}" y2="${y + 26}" stroke="${crit}" stroke-dasharray="4 3"/>`;
          g += T(lim + 5, y + 16, `${v.segs.reduce((n, sg) => n + sg[1], 0)} KB`, { c: c.ink, s: 11 });
        }
        if (v.sms === 2) {  // the exchange, in the gap between the two SMs' bars
          const ya = y0b + 28, yb = y0b + gapY - 18, on = st.hover === 'x', sw = on ? 2.6 : 1.6;
          g += `<g data-k="x" style="cursor:help"><rect x="${px}" y="${ya - 2}" width="${bw - 56}" height="${yb - ya + 4}" fill="transparent"/>`
            + `<line x1="${px + 14}" y1="${ya}" x2="${px + 14}" y2="${yb}" stroke="${c.ink}" stroke-width="${sw}"/><path d="M${px + 10},${yb - 6} L${px + 14},${yb} L${px + 18},${yb - 6}" fill="none" stroke="${c.ink}" stroke-width="${sw}"/>`
            + `<line x1="${px + 30}" y1="${yb}" x2="${px + 30}" y2="${ya}" stroke="${c.ink}" stroke-width="${sw}"/><path d="M${px + 26},${ya + 6} L${px + 30},${ya} L${px + 34},${ya + 6}" fill="none" stroke="${c.ink}" stroke-width="${sw}"/>`
            + T(px + 42, (ya + yb) / 2 + 4, 'exchange of partial scores', { s: 11, c: c.ink2 }) + `</g>`;
        }
        let yk = y0b + (v.sms - 1) * gapY + 46;
        g += T(px, yk, v.fits, { c: c.ink, s: 11.5, w: 1 });
        v.key.forEach((part, k) => { g += T(px, yk + 17 + 14 * k, part, { s: 10.5 }); });
        yk += 17 + 14 * v.key.length;
        // the timeline: what the tensor cores do, tile after tile
        const tx0 = narrow ? 8 : 4, tw = W - tx0 - 8, labW = narrow ? 0 : 150;
        let ty = Math.max(H, yk) + 22;
        if (narrow) {
          g += T(tx0, ty, 'Over time, for three tiles of keys', { c: c.ink, w: 1, s: 12 });
          ty += 16; g += T(tx0, ty, st.ver === 'old' ? 'loading and computing take turns' : '64-key tiles, two stages', { s: 11, c: c.ink2 });
        } else g += T(tx0, ty, st.ver === 'old' ? 'Over time, for three tiles of keys: loading and computing take turns' : 'Over time, for three tiles of keys (64-key tiles, two stages)', { c: c.ink, w: 1, s: 12 });
        ty += 12;
        const lanes = st.ver === 'old' ? ['load keys and values', 'tensor cores'] : ['load keys and values', 'tensor cores', 'exchange with the partner'];
        const laneH = 18, lx = tx0 + labW, lw = tw - labW, laneY = [];
        lanes.forEach((nm, k) => {
          if (narrow) { ty += 13; g += T(tx0, ty, nm, { s: 10.5 }); ty += 4; } else g += T(tx0, ty + 13, nm, { s: 11 });
          laneY.push(ty); ty += laneH + (narrow ? 4 : 8);
        });
        const seg2 = (k, a, b, kind) => {
          const y = laneY[k], x = lx + a * lw, w = Math.max(1, (b - a) * lw - 1.5), on = st.hover === `tl-${kind}`;
          if (kind === 'idle') return `<rect data-k="tl-idle" x="${x}" y="${y}" width="${w}" height="${laneH}" rx="2" fill="url(#chip-hatch)" stroke="${c.muted}" stroke-opacity="${on ? 0.9 : 0.5}" style="cursor:help"/>`;
          const fill = kind === 'comp' ? c.ink2 : kind === 'load' ? green : c.hi;
          return `<rect data-k="tl-${kind}" x="${x}" y="${y}" width="${w}" height="${laneH}" rx="2" fill="${fill}" fill-opacity="${on ? 1 : 0.75}" style="cursor:help"/>`;
        };
        g += `<defs><pattern id="chip-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="${c.muted}" stroke-width="1.2"/></pattern></defs>`;
        if (st.ver === 'old') {
          for (let k = 0; k < 3; k++) {
            const a = k / 3, m = a + 0.64 / 3, b = (k + 1) / 3;
            g += seg2(0, a, m, 'load') + seg2(1, a, m, 'idle') + seg2(1, m, b, 'comp');
          }
          g += T(lx, ty + 8, narrow ? 'tensor cores busy 36% of the time' : 'the tensor cores are busy 36% of the time (Nsight Compute)', { s: 11, c: c.ink2 });
        } else {
          const per = 1.5 + 0.67 + 0.5, u = 1 / (3 * per);
          for (let k = 0; k < 3; k++) {
            const t0 = k * per * u, tc = t0 + 1.5 * u, tx = tc + 0.67 * u, te = tx + 0.5 * u;
            g += seg2(0, t0 + 0.08 * u, tc - 0.12 * u, 'load') + seg2(1, t0, tc, 'comp') + seg2(1, tc, te, 'idle') + seg2(2, tc, tx, 'x') + seg2(2, tx, te, 'idle');
            if (k === 0 && !narrow) g += T(lx + (t0 + tc) / 2 * lw, ty + 8, '1.5 µs', { a: 'middle', s: 10.5 }) + T(lx + (tc + tx) / 2 * lw, ty + 8, '0.67', { a: 'middle', s: 10.5 }) + T(lx + (tx + te) / 2 * lw, ty + 8, '0.5', { a: 'middle', s: 10.5 });
          }
          g += T(lx, ty + (narrow ? 8 : 24), narrow ? 'per tile: 1.5 µs work, 0.67 send, 0.5 wait' : 'the next tile loads during the work; each tile then waits for the exchange', { s: 11, c: c.ink2 });
          if (!narrow) ty += 16;
        }
        H = ty + 18;
      }
      const aria = st.tab === 'experts' ? `The H200: 132 SMs, an L2 cache shared by all of them and HBM; ${CHIP.experts[st.ver].name} write ${mbOf(CHIP.experts[st.ver].rows)} MB to HBM per MoE layer` : `The H200 and the shared memory of one SM under ${CHIP_ATTN[st.ver].name}, with a timeline of loading and computing`;
      pic.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="${aria}">${g}</svg>`;
      pic.querySelectorAll('[data-k]').forEach((n) => {
        const k = n.getAttribute('data-k');
        n.addEventListener('mouseenter', () => { st.hover = k; draw(); });
        n.addEventListener('click', () => {
          if (k[0] === 'r') { stop(); st.step = +k.slice(1); st.hover = null; } else st.hover = k;
          draw();
        });
      });
      pic.querySelector('svg').addEventListener('mouseleave', () => { if (st.hover) { st.hover = null; draw(); } });
      out.innerHTML = readout();
    }
    function readout() {
      const h = st.hover;
      if (h && CHIP_TIPS[h]) return P(CHIP_TIPS[h][0], CHIP_TIPS[h][1]);
      if (st.tab === 'experts') {
        const rows = CHIP.experts[st.ver].rows;
        if (h && h[0] === 'r') { const r = rows[+h.slice(1)]; return P(`${+h.slice(1) + 1}. ${r[0]}`, r[3]); }
        if (st.step != null) { const r = rows[st.step]; return P(`Kernel ${st.step + 1} of ${rows.length}: ${r[0]}`, r[3]); }
        return st.ver === 'old'
          ? P("vLLM's expert path: 9 kernels", 'Between two kernels, an intermediate tensor goes through HBM: one kernel writes it, and the next reads it back. vLLM\'s path splits one MoE layer into 9 kernels, so its intermediate tensors make round trips through the slowest memory: 737 MB written per layer at TD batch 32 (the sum of the sizes in Figure 35). Half of it is the 369 MB of rows that the second GEMM writes and moe_sum reads back only to add them up.', 'Press ▶ to go through the kernels one at a time.')
          : P('The rewrite (moe_ws): 6 kernels', 'The two matrix multiplications are as fast as vLLM\'s; the rewrite removes the traffic between them. The activation is applied in registers inside the first GEMM, and the second GEMM adds its rows straight into the output, where a token\'s 8 rows meet in the L2 cache. One MoE layer writes 173 MB instead of 737. The expert path is 19–28% faster, and the TD step 4–8%.', 'Press ▶ to go through the kernels one at a time.');
      }
      return st.ver === 'old'
        ? P('FlashAttention 4 at head dimension 512', 'An SM works on a tile of 64 queries against a tile of 64 keys and values. At head dimension 512, these tiles fill 200 KB of the 227 KB of shared memory, so there is no room for a second buffer: the SM cannot load the next keys while it uses the current ones. Loading and computing take turns, and the tensor cores are busy 36% of the time. It runs at 296–357 TFLOP/s, against 700–729 at head dimension 256.')
        : P('The pair kernel (attn512): one head split across two SMs', 'Each of two SMs takes 256 of the 512 columns, so each tile takes half the room. The freed space holds either a second buffer, so that the next keys load during the work (two stages of 64 keys, as drawn), or larger tiles of 80 keys in one stage, the fastest configuration with the 80-token KV pages that TD uses.', 'The price is the exchange: the scores need all 512 columns, so after each tile of keys the two SMs swap their partial scores. Global attention then runs at 0.71–0.74 of the attention ceiling, 1.36–1.37 times faster than FlashAttention 4 in dedicated runs.');
    }
    draw(); watch(pic, draw);
  }

  const MF = { prop1: mProp1, routing: mRouting, ladder: mLadder, eff: mEff, comp: mComp, recover: mRecover, vs2: mVs2, chip: mChip };
  document.querySelectorAll('[data-mfig]').forEach((root) => {
    try { MF[root.dataset.mfig](root); } catch (e) { root.textContent = 'This figure failed to load.'; console.error(e); }
  });
  const redraw = () => redraws.forEach((f) => f());
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
