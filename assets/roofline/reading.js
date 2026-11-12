// Reading aids for the Block Diffusion Rooflines page:
//  - the intuition / full-math switch: in intuition mode (the default) every definition and
//    proposition folds to its title and one-sentence summary, and every captioned table to its
//    caption, each with a button to open it;
//  - the contents list: a rail in the left margin on wide screens, a sheet opened by a "Contents"
//    button elsewhere, with the reading progress;
//  - the symbol key: selecting a formula lists what its symbols mean.
(function () {
  'use strict';
  const page = document.querySelector('.bdr');
  if (!page) return;
  const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const KEY = 'bdr-depth';

  // ---------------------------------------------------------------- reading depth
  function foldButton(host, label) {
    const btn = el(`<button type="button" class="box-more" aria-expanded="false">show ${label} +</button>`);
    btn.addEventListener('click', () => {
      const on = !host.classList.contains('open');
      host.classList.toggle('open', on);
      btn.setAttribute('aria-expanded', String(on));
      btn.textContent = on ? `hide ${label} −` : `show ${label} +`;
    });
    return btn;
  }
  for (const box of page.querySelectorAll('.box.prop, .box.def')) {
    const gist = box.querySelector('.gist');
    if (gist) gist.after(foldButton(box, 'the statement'));
  }
  for (const wrap of page.querySelectorAll('section .table-wrap:not(.nofold)')) {
    if (!wrap.querySelector('caption') || wrap.closest('.box, .calc, .bdr-fig, .card-more')) continue;
    wrap.classList.add('foldable');
    wrap.appendChild(foldButton(wrap, 'the table'));
  }
  const switches = [];
  function setDepth(depth, save) {
    page.classList.toggle('intuition', depth === 'intuition');
    switches.forEach((s) => s.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.depth === depth))));
    if (save) { try { localStorage.setItem(KEY, depth); } catch (e) { /* storage unavailable */ } }
  }
  function depthSwitch(cls) {
    const s = el(`<div class="depth ${cls}" role="group" aria-label="Reading depth"><span>reading</span><button type="button" data-depth="intuition">intuition</button><button type="button" data-depth="full">full math</button></div>`);
    s.addEventListener('click', (ev) => {
      const d = ev.target.dataset && ev.target.dataset.depth; if (!d) return;
      // keep the reader's place: the element at the top of the window stays there
      const probe = [...page.querySelectorAll('h2, h3, .bdr-fig, p')].find((e) => e.getBoundingClientRect().top > 80), y0 = probe && probe.getBoundingClientRect().top;
      setDepth(d, true);
      if (probe) scrollBy({ top: probe.getBoundingClientRect().top - y0, behavior: 'instant' });
    });
    switches.push(s);
    return s;
  }
  const hero = page.querySelector('.hero .jump');
  if (hero) hero.after(depthSwitch('depth-hero'));

  // ---------------------------------------------------------------- contents: rail or sheet
  const sections = [...page.querySelectorAll('section[id]')].map((sec) => {
    const h = sec.querySelector('h2');
    const label = h ? [...h.childNodes].filter((n) => !(n.classList && n.classList.contains('anchor'))).map((n) => n.textContent).join('').replace(/\s+/g, ' ').trim() : sec.id;
    const subs = [...sec.querySelectorAll('h3[id], .bdr-fig[id]')].map((x) => ({
      id: x.id,
      label: x.classList.contains('bdr-fig') ? x.querySelector('.fig-k').textContent.replace(/·.*$/, '').trim() : x.textContent.replace('#', '').trim(),
      fig: x.classList.contains('bdr-fig'),
    }));
    return { id: sec.id, label, subs };
  });
  const rail = el('<nav class="bdr-rail" aria-label="Contents"><div class="rail-progress"><i></i></div><a class="rail-top" href="#top">Block Diffusion Rooflines</a><ol></ol></nav>');
  const ol = rail.querySelector('ol');
  if (page.querySelector('#fig-quiz')) ol.appendChild(el('<li><a href="#fig-quiz" data-id="fig-quiz">Quiz</a></li>'));
  if (page.querySelector('#fig-accounting')) ol.appendChild(el('<li><a href="#fig-accounting" data-id="fig-accounting">How the numbers are counted</a></li>'));
  for (const s of sections) {
    const li = el(`<li><a href="#${s.id}" data-id="${s.id}">${s.label}</a></li>`);
    if (s.subs.length) {
      const sub = el('<ol class="sub"></ol>');
      for (const x of s.subs) sub.appendChild(el(`<li><a href="#${x.id}" data-id="${x.id}" class="${x.fig ? 'fig' : ''}">${x.label}</a></li>`));
      li.appendChild(sub);
    }
    ol.appendChild(li);
  }
  rail.appendChild(depthSwitch('depth-rail'));
  const fab = el('<button type="button" class="rail-fab" aria-expanded="false" aria-controls="bdr-contents">Contents <span></span></button>');
  const backdrop = el('<div class="rail-backdrop" hidden></div>');
  rail.id = 'bdr-contents';
  document.body.append(backdrop, rail, fab);
  const setSheet = (on) => { rail.classList.toggle('open', on); backdrop.hidden = !on; fab.setAttribute('aria-expanded', String(on)); };
  fab.addEventListener('click', () => setSheet(!rail.classList.contains('open')));
  backdrop.addEventListener('click', () => setSheet(false));
  rail.addEventListener('click', (ev) => { if (ev.target.closest('a')) setSheet(false); });
  addEventListener('keydown', (ev) => { if (ev.key === 'Escape') { setSheet(false); closeKey(); } });

  const bar = rail.querySelector('.rail-progress i'), fabPct = fab.querySelector('span');
  const links = [...rail.querySelectorAll('a[data-id]')];
  const targets = links.map((a) => [a, document.getElementById(a.dataset.id)]).filter(([, t]) => t);
  function update() {
    const h = document.documentElement.scrollHeight - innerHeight, done = h > 0 ? Math.min(100, (100 * scrollY) / h) : 0;
    bar.style.width = `${done}%`;
    fabPct.textContent = `${Math.round(done)}%`;
    // current = last target whose top is above a line 30% down the window
    const line = innerHeight * 0.3;
    let cur = null, curSub = null;
    for (const [a, t] of targets) {
      if (t.getBoundingClientRect().top > line) continue;
      if (a.closest('ol.sub')) curSub = a; else { cur = a; curSub = null; }
    }
    for (const [a] of targets) a.classList.toggle('active', a === cur || a === curSub);
    rail.querySelectorAll(':scope > ol > li').forEach((li) => li.classList.toggle('current', !!cur && li.contains(cur)));
  }
  addEventListener('scroll', update, { passive: true });
  addEventListener('resize', update);

  // ---------------------------------------------------------------- symbol key
  // [pattern in the TeX source, symbol as HTML, meaning]. Patterns avoid lookbehind (older Safari).
  const S = (b, s) => `<i>${b}</i><sub>${s}</sub>`;
  const SYMBOLS = [
    [/\\Gamma/, 'Γ', 'throughput gain: tokens per second of TD over those of AR'],
    [/\\omega/, 'ω = <i>Sr</i> + 1', 'forward passes that one canvas costs'],
    [/\\mathrm\{TPF\}/, 'TPF = <i>b</i>/ω', 'tokens committed per forward pass'],
    [/I\^\\star/, '<i>I</i>* = π/β', 'ridge point of the accelerator: a pass is compute-bound when its intensity reaches it (206 on an H200)'],
    [/I_\{AR\}/, S('I', 'AR'), 'arithmetic intensity of one AR step: its FLOPs per byte read from memory'],
    [/I_\{TD\}/, S('I', 'TD'), 'arithmetic intensity of one TD pass over a canvas of <i>b</i> positions'],
    [/I\^(\{(AR|TD|X)\}|X)_\{\\mathrm\{lin\}\}/, '<i>I</i><sup>X</sup><sub>lin</sub>', 'linear part of an intensity: linear-layer FLOPs over weight bytes'],
    [/I\^(\{(AR|TD|X)\}|X)_\{\\mathrm\{attn\}\}/, '<i>I</i><sup>X</sup><sub>attn</sub>', 'attention part of an intensity: attention FLOPs over KV-cache bytes'],
    [/I_\{\\mathrm\{attn\}\}/, S('I', 'attn'), 'attention FLOPs per cache byte for one query on a global layer (8 for DiffusionGemma)'],
    [/\\Pi_\{\\mathrm\{lin\}\}/, S('Π', 'lin'), 'linear penalty: weight bytes a TD pass reads over those an AR step reads (7.9 at batch 1, 1 at large batch)'],
    [/\\Pi_\{\\mathrm\{attn\}\}/, S('Π', 'attn'), 'attention penalty: cache bytes a TD pass reads over those an AR step reads (1 for dense and windowed attention)'],
    [/\\Pi_E/, S('Π', 'E'), 'distinct experts a TD pass touches over those an AR step touches'],
    [/\\Pi(?![_A-Za-z])/, 'Π', 'overall penalty: bytes per sequence of a TD pass over those of an AR step'],
    [/B\^\\star_\{TD\}/, '<i>B</i>*<sub>TD</sub>', 'smallest batch at which a TD pass is compute-bound'],
    [/B\^\\star_\{AR\}/, '<i>B</i>*<sub>AR</sub>', 'smallest batch at which an AR step is compute-bound'],
    [/B_\\times/, S('B', '×'), 'parity batch: beyond it AR is faster'],
    [/N_\{\\mathrm\{act\}\}/, S('N', 'act'), 'parameters active per token (3.08B for DiffusionGemma, without the vocabulary)'],
    [/W_\{\\mathrm\{tot\}\}/, S('W', 'tot'), 'all parameters (24.5B, without the vocabulary)'],
    [/W_\{AR\}/, S('W', 'AR'), 'weights an AR step reads at batch <i>B</i>, in parameters'],
    [/W_\{TD\}/, S('W', 'TD'), 'weights a TD pass reads at batch <i>B</i>, in parameters'],
    [/U_\{E,k\}/, '<i>U</i><sub>E,k</sub>(<i>N</i>)', 'expected number of distinct experts that <i>N</i> positions touch in a layer (<i>E</i> = 128 experts, <i>k</i> = 8 per token)'],
    [/U_\{C,m\}/, '<i>U</i><sub>C,m</sub>(<i>n</i>)', 'cached tokens that <i>n</i> queries read together in a sparse layer, each selecting <i>m</i>'],
    [/\\theta_E/, S('θ', 'E'), 'expert share of the weights that an AR step reads'],
    [/\\theta(?!_E)/, 'θ', 'share of the bytes of a pass carried by one of its parts'],
    [/[FQ]\^(\{(AR|TD|X)\}|X)/, '<i>F</i>, <i>Q</i>', 'FLOPs and bytes of one pass, per sequence'],
    [/\\nu_/, 'ν', 'KV-cache bytes per cached token (G: global, W: windowed, S: sparse layers, I: indexer)'],
    [/\\alpha_/, 'α', 'attention FLOPs per query and per cached token (same subscripts as ν)'],
    [/s_w/, S('s', 'w'), 'bytes per weight (2 in BF16)'],
    [/\\pi(?![A-Za-z])/, 'π', 'peak FLOPs per second of the accelerator'],
    [/\\beta(?![A-Za-z])/, 'β', 'memory bandwidth of the accelerator, in bytes per second'],
    [/(^|[^\\A-Za-z_])B(?![A-Za-z^_])/, '<i>B</i>', 'batch: sequences decoded together'],
    [/(^|[^\\A-Za-z_])C(?![A-Za-z_])/, '<i>C</i>', 'context: tokens already in the KV cache of each sequence'],
    [/(^|[^\\A-Za-z_])b(?![A-Za-z])/, '<i>b</i>', 'canvas length, in tokens'],
    [/(^|[^\\A-Za-z_])S(?![A-Za-z_])/, '<i>S</i>', 'denoising passes per canvas'],
    [/(^|[^\\A-Za-z_])r(?![A-Za-z])/, '<i>r</i>', 'cost of a denoising pass, as a fraction of a forward pass'],
  ];
  const key = el('<div class="sym-key" role="dialog" aria-label="Symbols in this formula" hidden></div>');
  document.body.appendChild(key);
  function closeKey() { key.hidden = true; }
  function texOf(container) {
    try {
      for (const item of window.MathJax.startup.document.math) if (item.typesetRoot === container) return item.math;
    } catch (e) { /* MathJax not ready */ }
    return null;
  }
  page.addEventListener('click', (ev) => {
    const mj = ev.target.closest('mjx-container');
    if (!mj) return;
    const card = mj.closest('.bdr-card');
    if (mj.closest('summary, button, a') || (card && !mj.closest('.card-more'))) return;
    const src = texOf(mj);
    if (!src) return;
    const tex = src.replace(/\\text\{[^}]*\}/g, ''); // \text{B} is "billion", not the batch
    const hits = SYMBOLS.filter(([re]) => re.test(tex)).slice(0, 14);
    if (!hits.length) return;
    key.innerHTML = `<dl>${hits.map(([, sym, what]) => `<dt>${sym}</dt><dd>${what}</dd>`).join('')}</dl>`;
    key.hidden = false;
    const r = mj.getBoundingClientRect(), w = key.offsetWidth, vw = document.documentElement.clientWidth;
    key.style.left = `${Math.max(8, Math.min(vw - w - 8, r.left + r.width / 2 - w / 2)) + scrollX}px`;
    const below = r.bottom + 8 + key.offsetHeight < innerHeight || r.top < key.offsetHeight + 8;
    key.style.top = `${(below ? r.bottom + 8 : r.top - key.offsetHeight - 8) + scrollY}px`;
    ev.stopPropagation();
  });
  document.addEventListener('click', (ev) => { if (!key.hidden && !ev.target.closest('.sym-key, mjx-container')) closeKey(); });

  // ---------------------------------------------------------------- start
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* storage unavailable */ }
  setDepth(saved === 'full' ? 'full' : 'intuition', false);
  update();
})();
