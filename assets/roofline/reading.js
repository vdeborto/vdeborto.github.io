// Reading aids for the Block Diffusion Rooflines page:
//  - the intuition / full-math switch: in intuition mode (the default) every definition and
//    proposition folds to its title and one-sentence summary, and every captioned table to its
//    caption, each with a button to open it;
//  - the contents list: a rail in the left margin on wide screens, a sheet opened by a "Contents"
//    button elsewhere, with the reading progress;
//  - the symbol key: selecting a formula lists what its symbols mean.
//  - the right margin on wide screens: sidenotes, and the symbols of the formula being read.
//  - the citation block's copy button, and the comment thread under each chapter (giscus).
(function () {
  'use strict';
  const page = document.querySelector('.bdr');
  if (!page) return;
  try {
    const meta = JSON.parse(document.getElementById('bdr-book').textContent);
    if (meta.ids) {
      const id = decodeURIComponent(location.hash.slice(1)), calc = meta.chapters.find((c) => c.slug === 'calculator');
      // Old single-page links carried the calculator's state in the query (?B=64); forward those, and
      // only those, so that tracking or sign-in parameters (utm_*, giscus) leave the reader where they are.
      const CALC_KEY = /^(hw|peak8?|bw|mem|prec|kern|model|route|P|G|b|S|r|B|lt|la|bl|rho|actl2|kvonce|win|s-\w+|m-\w+)$/;
      if (calc && [...new URLSearchParams(location.search).keys()].some((k) => CALC_KEY.test(k))) { location.replace(calc.url + location.search + (location.hash || '#calc')); return; }
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
    if (box.closest('#setup')) { box.classList.add('open'); continue; }  // Setup's definitions are always shown in full
    const gist = box.querySelector('.gist');
    if (gist) gist.after(foldButton(box, box.classList.contains('def') ? 'the definition' : 'the statement'));
  }
  for (const wrap of page.querySelectorAll('section .table-wrap:not(.nofold)')) {
    if (!wrap.querySelector('caption') || wrap.closest('.box, .calc, .bdr-fig, .card-more, details.numbers, details.answer')) continue;
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
  // The site's layout titles a page "Chapter | site"; name the book in the browser tab instead.
  if (here && here.slug) document.title = `${here.title} · ${book.title}`;
  const rail = el(`<nav class="bdr-rail" aria-label="Contents"><div class="rail-progress"><i></i></div><a class="rail-top" href="${book ? book.home : '#top'}">${book ? book.title : 'Contents'}</a><ol></ol></nav>`);
  const ol = rail.querySelector('ol');
  for (const c of chapters) {
    const cur = c === here;
    const li = el(`<li${cur ? ' class="current"' : ''}><a href="${cur ? '#top' : c.url}"${cur ? ' class="active" aria-current="page"' : ''}>${labelOf(c)}</a></li>`);
    ol.appendChild(li);
  }
  // The current chapter lists its sections, and the one being read is highlighted.
  const heads = [...page.querySelectorAll('section h3[id]')].filter((h) => h.textContent.trim() && !h.closest('.book-comments, .book-cite'));
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
  const folds = !(here && here.slug === 'symbols'); // the Symbols page has nothing to fold
  if (partLine && folds) partLine.after(depthSwitch('depth-hero'));
  if (folds) rail.appendChild(depthSwitch('depth-rail'));
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
  // [pattern on a formula's TeX, symbol in TeX, meaning, scope]: assets/roofline/symbols.js, which the
  // build makes from _roofline/symbols.txt, the file behind the Symbols page as well.
  // A scope keeps a letter that means two things to its chapters: T is Parts 1-7, P is Parts 8-11.
  const SYMBOLS = (window.BDR_SYMBOLS || []).map(([re, sym, what, scope]) => [new RegExp(re), sym, what, scope]);
  const sheet = chapters.find((c) => c.slug === 'symbols');
  const PRACTICE = new Set(['effective', 'stack', 'kernelwork', 'conclusion']);
  const key = el('<div class="sym-key" role="dialog" aria-label="Symbols in this formula" hidden></div>');
  document.body.appendChild(key);
  let keyTurn = 0; // a key that is still being typeset is dropped when another click comes first
  function closeKey() { key.hidden = true; keyTurn++; }
  // The key and the margin show a symbol as the formulas do: MathJax typesets their TeX, one
  // request at a time. show() replaces the content of a node and resolves once it is typeset.
  let mjQueue = Promise.resolve();
  function show(node, html) {
    const MJ = window.MathJax;
    try { MJ.typesetClear([node]); } catch (e) { /* nothing typeset there yet */ }
    node.innerHTML = html;
    if (!MJ || !MJ.typesetPromise) return Promise.resolve();
    mjQueue = mjQueue.then(() => MJ.typesetPromise([node])).catch(() => {});
    return mjQueue;
  }
  function texOf(container) {
    try {
      for (const item of window.MathJax.startup.document.math) if (item.typesetRoot === container) return item.math;
    } catch (e) { /* MathJax not ready */ }
    return null;
  }
  // the symbols of a formula that the key knows, in the order of SYMBOLS
  function symbolsOf(mj, max = 14) {
    const src = texOf(mj);
    if (!src) return [];
    const tex = src.replace(/\\text\{[^}]*\}/g, ''); // \text{B} is "billion", not the batch
    const sec = mj.closest('section'), scope = sec && PRACTICE.has(sec.id) ? 'P' : 'T';
    const plain = tex.replace(/\\hat\{?\\(pi|beta)\}?/g, 'HAT');
    return SYMBOLS.filter(([re, , , sc]) => (!sc || sc === scope) && re.test(re.source.includes('hat') ? tex : plain)).slice(0, max);
  }
  const keyList = (hits) => `<dl>${hits.map(([, sym, what]) => `<dt>\\(${sym}\\)</dt><dd>${what}</dd>`).join('')}</dl>`;
  page.addEventListener('click', (ev) => {
    const mj = ev.target.closest('mjx-container');
    if (!mj) return;
    const card = mj.closest('.bdr-card');
    if (mj.closest('summary, button, a, .sym-sheet') || (card && !mj.closest('.card-more'))) return;
    const hits = symbolsOf(mj);
    if (!hits.length) return;
    ev.stopPropagation();
    const turn = ++keyTurn;
    key.style.visibility = 'hidden'; // laid out, so that it can be measured, and shown once typeset
    key.hidden = false;
    show(key, keyList(hits) + (sheet ? `<a class="sym-all" href="${sheet.url}">All symbols</a>` : '')).then(() => {
      if (turn !== keyTurn) return;
      const r = mj.getBoundingClientRect(), w = key.offsetWidth, vw = document.documentElement.clientWidth;
      key.style.left = `${Math.max(8, Math.min(vw - w - 8, r.left + r.width / 2 - w / 2)) + scrollX}px`;
      const below = r.bottom + 8 + key.offsetHeight < innerHeight || r.top < key.offsetHeight + 8 + 64; // 64: the site's fixed header
      key.style.top = `${(below ? r.bottom + 8 : r.top - key.offsetHeight - 8) + scrollY}px`;
      key.style.visibility = '';
    });
  });
  document.addEventListener('click', (ev) => { if (!key.hidden && !ev.target.closest('.sym-key, mjx-container')) closeKey(); });

  // ---------------------------------------------------------------- the right margin (wide screens)
  // Where the window leaves room to the right of the text column, two things go there:
  //  - sidenotes: each footnote sits beside the sentence that cites it (the Notes list is then hidden);
  //  - the symbols of the display formula nearest the reading line, beside that formula.
  // Narrower windows keep the Notes list and the symbol key opened by selecting a formula.
  const MARGIN_MIN = 230, GAP = 30, NOTE_W = 300;
  const layer = el('<div class="bdr-margin"></div>');
  document.body.appendChild(layer);
  const symNote = el('<aside class="margin-sym" aria-label="Symbols in the formula beside" hidden><div class="mk">In this formula</div><div class="mb"></div></aside>');
  layer.appendChild(symNote);
  let sides = [], marginX = 0, marginW = 0, marginOn = false, symFor = null;
  const docTop = (r) => r.top + scrollY;
  // boxes that reach into the margin (a wide figure or table): notes are pushed below them
  function obstacles(right) {
    return [...page.querySelectorAll('section > *, .wide, .calc, figure, .bdr-fig, .table-wrap')]
      .map((e) => e.getBoundingClientRect()).filter((r) => r.width && r.right > right + 4)
      .map((r) => [docTop(r) - 6, docTop(r) + r.height + 6]);
  }
  function placeAll(items, obs) {
    // items: [{node, want}] in reading order; each goes as near its anchor as the ones above allow
    let floor = -Infinity;
    for (const it of items) {
      let y = Math.max(it.want, floor);
      for (let moved = true; moved;) {
        moved = false;
        for (const [a, b] of obs) if (y < b && y + it.node.offsetHeight > a) { y = b; moved = true; }
      }
      it.node.style.top = `${y}px`;
      floor = y + it.node.offsetHeight + 12;
    }
  }
  function layoutMargin() {
    const col = page.getBoundingClientRect(), vw = document.documentElement.clientWidth;
    const room = vw - col.right - GAP - 16;
    marginOn = room >= MARGIN_MIN && !!col.width;
    page.classList.toggle('has-margin', marginOn);
    // the contents' entry for the Notes list goes with the list
    const nh = page.querySelector('section.footnotes h3[id]');
    if (nh) document.querySelectorAll(`a[href$="#${nh.id}"]`).forEach((a) => { const li = a.closest('li') || a; li.hidden = marginOn; });
    layer.hidden = !marginOn;
    if (!marginOn) return;
    marginX = col.right + scrollX + GAP;
    marginW = Math.min(NOTE_W, room);
    layer.style.left = `${marginX}px`;
    layer.style.width = `${marginW}px`;
    // sidenotes: one per visible footnote reference
    const refs = [...page.querySelectorAll('sup.fn-ref')].filter((r) => r.offsetParent && !r.closest('.footnotes'));
    sides.forEach((n) => n.remove());
    sides = refs.map((ref) => {
      const a = ref.querySelector('a'), lab = a ? decodeURIComponent(a.getAttribute('href').slice(1)) : '';
      const li = lab && document.getElementById(lab);
      const n = el(`<aside class="sidenote"><span class="sn-num">${a ? a.textContent : ''}</span> </aside>`);
      if (li) { const c = li.cloneNode(true); c.querySelectorAll('.fn-back').forEach((x) => x.remove()); n.insertAdjacentHTML('beforeend', c.innerHTML); }
      layer.appendChild(n);
      return n;
    });
    const obs = obstacles(col.right);
    placeAll(sides.map((n, i) => ({ node: n, want: docTop(refs[i].getBoundingClientRect()) - 4 })), obs);
    placeSym();
  }
  // the display formula nearest the reading line (40% down the window), in the text column
  function currentFormula() {
    const line = innerHeight * 0.4, colR = page.getBoundingClientRect().right;
    let best = null, bestD = Infinity;
    page.querySelectorAll('mjx-container[display="true"]').forEach((mj) => {
      if (!mj.offsetParent) return;
      const r = mj.getBoundingClientRect();
      if (r.bottom < 60 || r.top > innerHeight - 40 || r.right > colR + 4 || !r.height) return;
      if (mj.closest('.bdr-card:not(.open), summary, .sym-key, .xref-card')) return;
      const d = r.bottom < line ? line - r.bottom : r.top > line ? r.top - line : 0;
      if (d < bestD) { bestD = d; best = mj; }
    });
    return best;
  }
  function placeSym() {
    if (!marginOn) { symNote.hidden = true; return; }
    const mj = currentFormula();
    const hits = mj ? symbolsOf(mj, 10) : [];
    if (!hits.length) { symNote.hidden = true; symFor = null; return; }
    if (mj !== symFor) { // typeset the new list, then place the note at its final height
      symFor = mj;
      symNote.style.visibility = 'hidden';
      show(symNote.querySelector('.mb'), keyList(hits)).then(() => { if (symFor === mj) { symNote.style.visibility = ''; placeSym(); } });
    }
    symNote.hidden = false;
    const r = mj.getBoundingClientRect();
    let want = docTop(r) + Math.max(0, (r.height - symNote.offsetHeight) / 2);
    // keep it in the window: not under the site header, not past the bottom edge
    want = Math.max(Math.min(want, scrollY + innerHeight - symNote.offsetHeight - 12), scrollY + 72);
    // never over a sidenote or a wide box
    const obs = obstacles(page.getBoundingClientRect().right).concat(sides.map((n) => [n.offsetTop - 10, n.offsetTop + n.offsetHeight + 10]));
    placeAll([{ node: symNote, want }], obs);
  }
  // on wide screens a footnote number points at its sidenote, not at the hidden Notes list
  page.addEventListener('click', (ev) => {
    const a = ev.target.closest('sup.fn-ref a');
    if (!a || !marginOn) return;
    const i = [...page.querySelectorAll('sup.fn-ref')].filter((r) => r.offsetParent).indexOf(a.parentElement);
    const n = sides[i];
    if (!n) return;
    ev.preventDefault();
    n.classList.add('flash'); setTimeout(() => n.classList.remove('flash'), 1200);
  });
  let marginTick = 0;
  const relayout = () => { if (!marginTick) marginTick = requestAnimationFrame(() => { marginTick = 0; layoutMargin(); }); };
  addEventListener('resize', relayout);
  addEventListener('scroll', () => { if (marginOn) requestAnimationFrame(placeSym); }, { passive: true });
  if (window.ResizeObserver) new ResizeObserver(relayout).observe(page);
  page.addEventListener('toggle', relayout, true);
  page.addEventListener('click', () => setTimeout(relayout, 0));
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(relayout);
  (function waitMath(k) {
    const p = window.MathJax && MathJax.startup && MathJax.startup.promise;
    if (p) p.then(() => { layoutMargin(); if (window.MathJax.typesetPromise) layer.querySelectorAll('.sidenote').length && MathJax.typesetPromise([layer]).catch(() => {}); });
    else if (k < 50) setTimeout(() => waitMath(k + 1), 100);
  })(0);

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
      if ((e.classList.contains('foldable') || e.matches('.box.prop, .box.def')) && !e.classList.contains('open') && page.classList.contains('intuition')) {
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
  const hideCard = () => { card.hidden = true; current = null; symNote.style.visibility = ''; };
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
      const r = a.getBoundingClientRect(), vw = document.documentElement.clientWidth;
      card.classList.toggle('in-margin', marginOn);
      if (marginOn) {
        // wide windows: in the margin, level with the link, over the symbol note but never over the text
        card.style.width = `${marginW}px`;
        card.style.left = `${marginX}px`;
        const y = Math.max(72, Math.min(r.top - 8, innerHeight - card.offsetHeight - 12));
        card.style.top = `${y + scrollY}px`;
        symNote.style.visibility = 'hidden';
      } else {
        card.style.width = '';
        const w = card.offsetWidth;
        card.style.left = `${Math.max(8, Math.min(vw - w - 8, r.left)) + scrollX}px`;
        const below = r.bottom + 10 + card.offsetHeight < innerHeight;
        card.style.top = `${(below ? r.bottom + 8 : r.top - card.offsetHeight - 8) + scrollY}px`;
      }
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
  let remark = 0;
  const sizeWatch = window.ResizeObserver ? new ResizeObserver(() => { if (!remark) remark = requestAnimationFrame(() => { remark = 0; markScrollers(); }); }) : null;
  function markScrollers() {
    page.querySelectorAll('.pfig-scroll, .mapfig .chart, .table-wrap, pre.envelope, .code-body').forEach((box) => {
      const over = box.scrollWidth > box.clientWidth + 4;
      box.classList.toggle('can-scroll', over);
      let hint = box.previousElementSibling;
      if (!hint || !hint.classList.contains('scroll-hint')) { hint = el(`<div class="scroll-hint">scroll sideways to see ${box.classList.contains('code-body') ? 'whole lines' : 'the whole figure'} →</div>`); box.before(hint); }
      hint.hidden = !over;
      const edge = () => {
        const end = box.scrollLeft + box.clientWidth >= box.scrollWidth - 4;
        box.classList.toggle('at-end', end); box.classList.toggle('at-start', box.scrollLeft <= 2);
        hint.hidden = !over || end;
      };
      if (!box.dataset.watched) {
        box.dataset.watched = '1'; box.addEventListener('scroll', edge, { passive: true });
        // the content changes width after load (MathJax, web fonts, numbers filled in by script)
        if (sizeWatch) { sizeWatch.observe(box); [...box.children].forEach((c) => sizeWatch.observe(c)); }
      }
      edge();
    });
  }
  markScrollers();
  addEventListener('resize', () => requestAnimationFrame(markScrollers));
  page.addEventListener('toggle', (e) => { if (e.target.matches && e.target.matches('details.code-card') && e.target.open) requestAnimationFrame(markScrollers); }, true);

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
