// Reading aids for the Block Diffusion Rooflines page:
//  - the intuition / full-math switch: in intuition mode (the default) every definition and
//    proposition folds to its title and one-sentence summary, and every captioned table to its
//    caption, each with a button to open it;
//  - the contents list: a rail in the left margin on wide screens, a sheet opened by a "Contents"
//    button elsewhere, with the reading progress;
//  - the symbol key: selecting a formula lists what its symbols mean.
//  - the citation block's copy button, and the comment thread under each chapter (giscus).
(function () {
  'use strict';
  const page = document.querySelector('.bdr');
  if (!page) return;
  try {
    const meta = JSON.parse(document.getElementById('bdr-book').textContent);
    if (meta.ids) {
      const id = decodeURIComponent(location.hash.slice(1)), calc = meta.chapters.find((c) => c.slug === 'calculator');
      if (location.search && calc) { location.replace(calc.url + location.search + (location.hash || '#calc')); return; }
      if (id && !document.getElementById(id) && meta.ids[id] !== undefined) {
        const to = meta.chapters.find((c) => c.slug === meta.ids[id]);
        if (to) { location.replace(to.url + location.hash); return; }
      }
    }
  } catch (e) { /* no book data */ }
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
    if (!wrap.querySelector('caption') || wrap.closest('.box, .calc, .bdr-fig, .card-more, details.numbers')) continue;
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
  // The note is a short book (one page per chapter, described by #bdr-book); the rail lists the chapters.
  let book = null;
  try { book = JSON.parse(document.getElementById('bdr-book').textContent); } catch (e) { /* no book data */ }
  const chapters = book ? book.chapters : [];
  const here = book ? chapters.find((c) => c.slug === book.current) : null;
  const labelOf = (c) => (c.num && c.slug ? `${c.num} · ${c.title}` : c.title);
  const rail = el(`<nav class="bdr-rail" aria-label="Contents"><div class="rail-progress"><i></i></div><a class="rail-top" href="${book ? book.home : '#top'}">${book ? book.title : 'Contents'}</a><ol></ol></nav>`);
  const ol = rail.querySelector('ol');
  for (const c of chapters) {
    const cur = c === here;
    const li = el(`<li${cur ? ' class="current"' : ''}><a href="${cur ? '#top' : c.url}"${cur ? ' class="active" aria-current="page"' : ''}>${labelOf(c)}</a></li>`);
    ol.appendChild(li);
  }
  // The current chapter lists its sections, and the one being read is highlighted.
  const heads = [...page.querySelectorAll('section h3[id]')].filter((h) => h.textContent.trim());
  let subLinks = [];
  if (here && heads.length > 1) {
    const sub = el('<ol class="rail-sub"></ol>');
    for (const h of heads) {
      const t = h.cloneNode(true); t.querySelectorAll('mjx-assistive-mml, .anchor').forEach((x) => x.remove());
      const a = el(`<li><a href="#${h.id}"></a></li>`);
      a.firstChild.textContent = t.textContent.replace(/\s+/g, ' ').trim();
      sub.appendChild(a); subLinks.push([h, a.firstChild]);
    }
    ol.querySelector('li.current').appendChild(sub);
  }
  function spy() {
    let cur = null;
    for (const [h, a] of subLinks) if (h.getBoundingClientRect().top < 140) cur = a;
    for (const [, a] of subLinks) a.classList.toggle('on', a === cur);
  }
  // A chapter page without the rail's switch on narrow screens still gets one under its title.
  const partLine = page.querySelector('.book-part');
  if (partLine) partLine.after(depthSwitch('depth-hero'));
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
  function update() {
    const h = document.documentElement.scrollHeight - innerHeight, done = h > 0 ? Math.min(100, (100 * scrollY) / h) : 0;
    bar.style.width = `${done}%`;
    fabPct.textContent = `${Math.round(done)}%`;
    spy();
  }
  addEventListener('scroll', update, { passive: true });
  addEventListener('resize', update);

  // ---------------------------------------------------------------- steppers: one panel at a time
  for (const st of page.querySelectorAll('.pfig-steps')) {
    const btns = [...st.querySelectorAll('.pfig-step-buttons button')], panels = [...st.querySelectorAll(':scope > .pfig-step')];
    const show = (k) => {
      panels.forEach((p) => { p.hidden = p.dataset.k !== k; });
      btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.k === k)));
    };
    st.classList.add('js');
    btns.forEach((b) => b.addEventListener('click', () => show(b.dataset.k)));
    st.addEventListener('keydown', (ev) => {
      if (ev.key !== 'ArrowRight' && ev.key !== 'ArrowLeft') return;
      const i = btns.findIndex((b) => b.getAttribute('aria-pressed') === 'true'), n = btns.length;
      const j = (i + (ev.key === 'ArrowRight' ? 1 : n - 1)) % n;
      show(btns[j].dataset.k); btns[j].focus(); ev.preventDefault();
    });
    show('0');
  }

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
    [/\\pi(?![A-Za-z])/, 'π', 'peak FLOPs per second of the accelerator', 'T'],
    [/\\beta(?![A-Za-z])/, 'β', 'memory bandwidth of the accelerator, in bytes per second', 'T'],
    [/(^|[^\\A-Za-z_])B(?![A-Za-z^_])/, '<i>B</i>', 'batch: sequences decoded together'],
    [/(^|[^\\A-Za-z_])C(?![A-Za-z_])/, '<i>C</i>', 'context: tokens already in the KV cache of each sequence'],
    [/(^|[^\\A-Za-z_])b(?![A-Za-z])/, '<i>b</i>', 'canvas length, in tokens'],
    [/(^|[^\\A-Za-z_])S(?![A-Za-z_])/, '<i>S</i>', 'denoising passes per canvas', 'T'],
    [/(^|[^\\A-Za-z_])r(?![A-Za-z])/, '<i>r</i>', 'cost of a denoising pass, as a fraction of a forward pass', 'T'],
    // Parts 8-11 (measurements)
    [/\\Gamma_\{\\mathrm\{meas\}\}/, 'Γ<sub>meas</sub>', 'measured gain: TD tokens per second over AR tokens per second', 'P'],
    [/\\Gamma_\{\\mathrm\{roof\}\}/, 'Γ<sub>roof</sub>', 'roofline gain at the measured ceilings and routing', 'P'],
    [/\\tau/, 'τ', 'empirical roofline time: max(<i>F</i>/π̂, <i>Q</i>/β̂), of a component <i>c</i> or of a whole step <i>X</i>', 'P'],
    [/\\hat\{?\\pi/, 'π̂', 'empirical compute ceiling: the highest FLOP rate measured on this H200 (FP8 1,468, BF16 781, attention 638 TFLOP/s)', 'P'],
    [/\\hat\{?\\beta/, 'β̂', 'empirical bandwidth: 4.32 TB/s measured on this H200', 'P'],
    [/\\eta/, 'η', 'efficiency: roofline time over measured time (1 means at the roofline)', 'P'],
    [/(^|[^\\A-Za-z_])t_\{?(X|AR|TD|c)/, '<i>t</i>', 'measured time of a step (<i>t</i><sub>AR</sub>, <i>t</i><sub>TD</sub>) or of a component (<i>t</i><sub>c</sub>)', 'P'],
    [/(^|[^\\A-Za-z_])n_\{?(X|AR|TD)/, '<i>n</i>', 'tokens a step commits: <i>B</i> for AR, <i>Bb</i>/ω for TD', 'P'],
    [/U\^\{\\mathrm\{meas\}\}/, '<i>U</i><sup>meas</sup>', 'distinct experts per layer that a pass touches, measured', 'P'],
    [/U\^\{\\mathrm\{freq\}\}/, '<i>U</i><sup>freq</sup>', 'distinct experts predicted from the measured expert frequencies, tokens independent', 'P'],
    [/N\^\{\\mathrm\{eff\}\}/, '<i>N</i><sup>eff</sup>', 'effective independent tokens: the <i>N</i> at which <i>U</i><sub>E,k</sub>(<i>N</i>) equals the measured count', 'P'],
    [/\\rho_/, 'ρ<sub>e</sub>', 'frequency with which a token selects expert <i>e</i>', 'P'],
    [/\\bar\{?m/, '<i>m̄</i>', 'rows per touched expert, <i>Nk</i>/<i>U</i>', 'P'],
    [/f_E/, S('f', 'E'), 'width of one expert (704)', 'P'],
    [/L_q|L_k/, '<i>L</i><sub>q</sub>, <i>L</i><sub>k</sub>', 'queries and keys per sequence in one global-attention call', 'P'],
    [/T_q|T_k/, '<i>T</i><sub>q</sub>, <i>T</i><sub>k</sub>', 'query rows and keys of one attention tile', 'P'],
    [/n_q|n_\{kv/, '<i>n</i><sub>q</sub>, <i>n</i><sub>kv</sub>', 'query heads and KV heads of a layer (16 and 2 on the global layers)'],
    [/\\mathbf\{Z\}/, '<b>Z</b>', 'attention scores <b>QK</b><sup>⊤</sup> of one tile', 'P'],
    [/(^|[^\\A-Za-z_])h(?![A-Za-z_])/, '<i>h</i>', 'head dimension of a global layer (512)', 'P'],
  ];
  const PRACTICE = new Set(['effective', 'stack', 'kernelwork', 'conclusion']);
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
    const sec = mj.closest('section'), scope = sec && PRACTICE.has(sec.id) ? 'P' : 'T';
    const hits = SYMBOLS.filter(([re, , , sc]) => (!sc || sc === scope) && re.test(tex)).slice(0, 14);
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

  // ---------------------------------------------------------------- citation: copy the BibTeX entry
  for (const btn of page.querySelectorAll('.cite-copy')) {
    btn.addEventListener('click', async () => {
      const pre = btn.closest('.cite-entry').querySelector('pre');
      const say = (t) => { btn.textContent = t; setTimeout(() => { btn.textContent = 'Copy'; }, 1600); };
      try { await navigator.clipboard.writeText(pre.textContent); say('Copied'); } catch (e) {
        const r = document.createRange(); r.selectNodeContents(pre);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r); say('Selected: copy with ⌘C or Ctrl-C');
      }
    });
  }

  // ---------------------------------------------------------------- comments (giscus, GitHub Discussions)
  // The thread is loaded when the reader reaches the end of the chapter, with the site's current
  // theme; the site's own theme switch then keeps iframe.giscus-frame in step (assets/js/theme.js).
  const thread = document.getElementById('bdr-comments');
  if (thread && thread.dataset.repo) {
    const load = () => {
      if (thread.dataset.loaded) return;
      thread.dataset.loaded = '1';
      const dark = typeof determineComputedTheme === 'function' ? determineComputedTheme() === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
      const s = document.createElement('script');
      const attrs = {
        src: 'https://giscus.app/client.js', 'data-repo': thread.dataset.repo, 'data-repo-id': thread.dataset.repoId,
        'data-category': thread.dataset.category, 'data-category-id': thread.dataset.categoryId,
        'data-mapping': 'specific', 'data-term': thread.dataset.term, 'data-strict': '1', 'data-reactions-enabled': '1',
        'data-emit-metadata': '0', 'data-input-position': 'top', 'data-theme': dark ? 'dark' : 'light', 'data-lang': 'en',
        crossorigin: 'anonymous',
      };
      Object.entries(attrs).forEach(([k, v]) => s.setAttribute(k, v));
      s.async = true;
      thread.querySelector('.comments-frame').appendChild(s);
    };
    // giscus reports a missing installation or an unreachable repository to its host page: say so
    // in words instead of leaving its error box. ("Discussion not found" is the normal state of a
    // chapter nobody has commented on yet, and is left to giscus.)
    addEventListener('message', (ev) => {
      if (ev.origin !== 'https://giscus.app' || !ev.data || !ev.data.giscus || !ev.data.giscus.error) return;
      if (!/not installed|Bad credentials|rate limit/i.test(String(ev.data.giscus.error))) return;
      const frame = thread.querySelector('.comments-frame');
      frame.innerHTML = `<p class="small">The comment thread is not available at the moment. You can write in the <a href="https://github.com/${thread.dataset.repo}/discussions">discussions of the site's repository</a>.</p>`;
    });
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); load(); } }, { rootMargin: '600px' });
      io.observe(thread);
    } else load();
  }

  // ---------------------------------------------------------------- a linked figure or table is shown
  // A link to #table-N or #figure-N opens the disclosures and folded tables around its target.
  function revealHash() {
    const id = decodeURIComponent(location.hash.slice(1));
    const t = id && document.getElementById(id);
    if (!t) return;
    let opened = false;
    for (let e = t; e && e !== document.body; e = e.parentElement) {
      if (e.tagName === 'DETAILS' && !e.open) { e.open = true; opened = true; }
      if (e.classList.contains('foldable') && !e.classList.contains('open')) {
        const btn = e.querySelector(':scope > .box-more');
        if (btn) btn.click(); else e.classList.add('open');
        opened = true;
      }
    }
    if (opened) requestAnimationFrame(() => { const box = t.closest('details.numbers, .table-wrap') || t; scrollTo(0, box.getBoundingClientRect().top + scrollY - 96); });
  }
  addEventListener('hashchange', revealHash);
  setTimeout(revealHash, 0);

  // ---------------------------------------------------------------- previews of cross-references
  // Hovering a link to a figure, table, statement or part shows what it is: the figure's claim, the
  // table's caption, the statement's one-line summary or the chapter's opening paragraph. Other
  // chapters are fetched once and cached. Touch screens follow the link instead.
  const docs = new Map();
  const getDoc = (url) => {
    if (!docs.has(url)) docs.set(url, fetch(url).then((r) => r.text()).then((t) => new DOMParser().parseFromString(t, 'text/html')).catch(() => null));
    return docs.get(url);
  };
  const strip = (node, sel) => { node.querySelectorAll(sel).forEach((x) => x.remove()); return node; };
  function previewOf(doc, hash) {
    if (!doc) return null;
    if (!hash) {
      const part = doc.querySelector('.book-part'), h = doc.querySelector('.post-title, h1'), p = doc.querySelector('.bdr .chapter-intro') || doc.querySelector('.bdr section > p');
      const num = part ? (part.textContent.match(/^(Part \S+)/) || [])[1] : '';
      return h ? `<div class="xk">${num || 'Chapter'}</div><p class="xt">${h.textContent.replace(/\s+/g, ' ').trim()}</p>${p ? `<p>${p.innerHTML}</p>` : ''}` : null;
    }
    const t = doc.getElementById(hash);
    if (!t) return null;
    if (hash.startsWith('figure-')) {
      const host = t.parentElement, title = host && host.querySelector('.fig-title');
      return `<div class="xk">${t.textContent.trim()}</div>${title ? `<p class="xt">${title.innerHTML}</p>` : ''}`;
    }
    if (hash.startsWith('table-')) {
      const c = t.innerHTML.split(/(?<=\.)\s/).slice(0, 2).join(' ');
      return `<p>${c}</p>`;
    }
    const tag = t.querySelector('.tag'), title = t.querySelector('.bdr-title, .title, summary'), gist = t.querySelector('.gist');
    if (!tag) return null;
    const head = title ? strip(title.cloneNode(true), '.tag').textContent.trim() : '';
    const g = gist ? strip(gist.cloneNode(true), 'span:first-child').innerHTML : '';
    return `<div class="xk">${tag.textContent.trim()}</div><p class="xt">${head}</p>${g ? `<p>${g}</p>` : ''}`;
  }
  const card = el('<div class="xref-card" role="tooltip" hidden></div>');
  document.body.appendChild(card);
  let hoverTimer = null, overCard = false, current = null;
  const hideCard = () => { card.hidden = true; current = null; };
  page.addEventListener('pointerover', (ev) => {
    const a = ev.target.closest && ev.target.closest('a.xref');
    if (!a || ev.pointerType === 'touch' || a === current) return;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(async () => {
      const u = new URL(a.href, location.href), hash = decodeURIComponent(u.hash.slice(1));
      const doc = await getDoc(u.pathname);
      const html = previewOf(doc, hash);
      if (!html || !a.matches(':hover')) return;
      current = a; card.innerHTML = html; card.hidden = false;
      const r = a.getBoundingClientRect(), w = card.offsetWidth, vw = document.documentElement.clientWidth;
      card.style.left = `${Math.max(8, Math.min(vw - w - 8, r.left)) + scrollX}px`;
      const below = r.bottom + 10 + card.offsetHeight < innerHeight;
      card.style.top = `${(below ? r.bottom + 8 : r.top - card.offsetHeight - 8) + scrollY}px`;
      if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([card]).catch(() => {});
    }, 260);
  });
  page.addEventListener('pointerout', (ev) => {
    const a = ev.target.closest && ev.target.closest('a.xref');
    if (!a) return;
    clearTimeout(hoverTimer);
    setTimeout(() => { if (!overCard && current === a && !a.matches(':hover')) hideCard(); }, 120);
  });
  card.addEventListener('pointerenter', () => { overCard = true; });
  card.addEventListener('pointerleave', () => { overCard = false; hideCard(); });
  addEventListener('scroll', () => { if (!card.hidden && !overCard) hideCard(); }, { passive: true });

  // ---------------------------------------------------------------- calculator verdict on phones
  // On a narrow screen the form comes before the result, so its first line stays in view while the
  // form is scrolled: a sticky copy of the verdict's headline.
  const verdictEl = document.getElementById('c-verdict');
  if (verdictEl) {
    const bar = el('<div class="calc-sticky" aria-hidden="true"></div>');
    verdictEl.closest('.calc').prepend(bar);
    // shown only while the real verdict is off screen
    new IntersectionObserver(([en]) => bar.classList.toggle('off', en.isIntersecting), { rootMargin: '-64px 0px 0px 0px' }).observe(verdictEl);
    const sync = () => { const h = verdictEl.firstElementChild; bar.textContent = h ? h.textContent.replace(/\s+/g, ' ').trim() : ''; };
    new MutationObserver(sync).observe(verdictEl, { childList: true, subtree: true, characterData: true });
    sync();
  }

  // ---------------------------------------------------------------- wide diagrams on narrow screens
  // A diagram wider than the column scrolls sideways; say so, and fade the edge that hides more.
  function markScrollers() {
    page.querySelectorAll('.pfig-scroll, .mapfig .chart, .table-wrap, pre.envelope').forEach((box) => {
      const over = box.scrollWidth > box.clientWidth + 4;
      box.classList.toggle('can-scroll', over);
      let hint = box.previousElementSibling;
      if (!hint || !hint.classList.contains('scroll-hint')) { hint = el('<div class="scroll-hint">scroll sideways to see the whole figure →</div>'); box.before(hint); }
      hint.hidden = !over;
      const edge = () => {
        const end = box.scrollLeft + box.clientWidth >= box.scrollWidth - 4;
        box.classList.toggle('at-end', end); box.classList.toggle('at-start', box.scrollLeft <= 2);
        hint.hidden = !over || end;
      };
      if (!box.dataset.watched) { box.dataset.watched = '1'; box.addEventListener('scroll', edge, { passive: true }); }
      edge();
    });
  }
  markScrollers();
  addEventListener('resize', () => requestAnimationFrame(markScrollers));

  // ---------------------------------------------------------------- equations that do not fit
  // A display equation written in two forms (.eq-wide, then .eq-narrow) shows the wide one only
  // where it fits its column; the CSS switch on the viewport width cannot see the column.
  function fitEquations() {
    page.querySelectorAll('.eq-wide').forEach((w) => {
      const n = w.nextElementSibling;
      if (!n || !n.classList.contains('eq-narrow')) return;
      w.classList.remove('eq-off'); n.classList.remove('eq-on');
      const m = w.querySelector('mjx-container');
      if (!m || !w.offsetParent) return;
      if (m.scrollWidth > w.clientWidth + 2) { w.classList.add('eq-off'); n.classList.add('eq-on'); }
    });
  }
  const whenTypeset = () => (window.MathJax && MathJax.startup && MathJax.startup.promise) || null;
  (function waitTypeset(k) {
    const p = whenTypeset();
    if (p) p.then(() => { fitEquations(); addEventListener('resize', () => requestAnimationFrame(fitEquations)); page.addEventListener('toggle', fitEquations, true); page.addEventListener('click', () => setTimeout(fitEquations, 0)); });
    else if (k < 50) setTimeout(() => waitTypeset(k + 1), 100);
  })(0);

  // ---------------------------------------------------------------- start
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* storage unavailable */ }
  setDepth(saved === 'full' ? 'full' : 'intuition', false);
  update();
})();
