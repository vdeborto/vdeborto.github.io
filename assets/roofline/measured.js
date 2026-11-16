// Interactive charts of the measurements of Parts 8-11 (DiffusionGemma against Gemma 4 26B-A4B on
// one H200, vLLM, FP8). Every value comes from the JSON that bin/build-roofline.py writes into the
// page (#bdr-practice), itself exported from the campaign data by roofscope's pages/book/export.py.
// Nothing is computed here beyond ratios of those values.
//
// Mounts on [data-mfig]: ladder (data-series="a,b,c" or "c,d"), eff, comp (data-src="eta2" or
// "eta3"), recover, vs2.
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

  const MF = { prop1: mProp1, routing: mRouting, ladder: mLadder, eff: mEff, comp: mComp, recover: mRecover, vs2: mVs2 };
  document.querySelectorAll('[data-mfig]').forEach((root) => {
    try { MF[root.dataset.mfig](root); } catch (e) { root.textContent = 'This figure failed to load.'; console.error(e); }
  });
  const redraw = () => redraws.forEach((f) => f());
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
