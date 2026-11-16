// site-edit: the edit mode added to every page served by bin/site-edit (http://localhost:4100).
// Edit mode (button, or the E key): click a block of text; the panel at the bottom shows the source
// that produced it, and the block is redrawn in place as you type. Save (Cmd-S) writes the file;
// the page reloads once the site is rebuilt, at the same place. Esc closes without saving.
(function () {
  'use strict';
  if (window.__siteEdit) return;
  window.__siteEdit = true;
  const API = '/__edit/';
  const BLOCKS = 'p, li, h1, h2, h3, h4, h5, h6, figcaption, caption, dt, dd, td, th, blockquote, summary, pre, ' +
    '.gist, .fig-title, .fig-lede, .card-k, .card-v, .tile, .chapter-intro, .post-description, .title, .bdr-title, .rn, .rf, .announcement, .news td, ' +
    '.pub-title, .pub-authors, .pub-venue, .pub-award, .pub-summary, .pub-meta';
  const ON_KEY = 'se-on', SCROLL_KEY = 'se-scroll:' + location.pathname;

  const el = (h) => { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content.firstElementChild; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const post = (path, body) => fetch(API + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then((r) => r.json());
  const get = (path) => fetch(API + path).then((r) => r.json());
  let on = false;

  // ------------------------------------------------------------------ toolbar
  const bar = el(`<div class="se-ui se-bar" role="toolbar" aria-label="Site editor">
      <button type="button" class="se-toggle" title="Edit mode (E)">✎ Edit</button>
      <span class="se-more">
        <button type="button" data-act="page" title="Edit the whole source file of this page">Page source</button>
        <button type="button" data-act="news" title="Add a news item to the homepage">+ News</button>
        <button type="button" data-act="undo" title="Undo the last save">Undo</button>
        <button type="button" data-act="publish" class="se-publish" title="Commit and push the changes">Publish <span class="se-n"></span></button>
        <button type="button" data-act="help" title="How this works">?</button>
      </span>
    </div>`);
  const toast = el('<div class="se-ui se-toast" role="status" aria-live="polite" hidden></div>');
  document.body.append(bar, toast);
  let toastTimer = 0;
  function say(msg, kind = '', ms = 4000) {
    toast.textContent = msg; toast.className = 'se-ui se-toast ' + kind; toast.hidden = false;
    clearTimeout(toastTimer); if (ms) toastTimer = setTimeout(() => { toast.hidden = true; }, ms);
  }
  function setOn(v) {
    on = v; document.documentElement.classList.toggle('se-editing', on);
    bar.querySelector('.se-toggle').textContent = on ? '✎ Editing: click a block' : '✎ Edit';
    bar.classList.toggle('on', on);
    try { sessionStorage.setItem(ON_KEY, on ? '1' : ''); } catch (e) { /* private mode */ }
    if (!on) hover(null);
  }
  bar.querySelector('.se-toggle').addEventListener('click', () => setOn(!on));
  async function refreshCount() {
    try {
      const c = await get('changes');
      const n = c.site.length + c.book.length + (c.book_ahead ? 1 : 0);
      bar.querySelector('.se-n').textContent = n ? `(${n})` : '';
      bar.querySelector('.se-publish').classList.toggle('has', n > 0);
    } catch (e) { /* server restarting */ }
  }
  refreshCount();

  // ------------------------------------------------------------------ hover and click in edit mode
  let hovered = null;
  function hover(b) { if (hovered) hovered.classList.remove('se-hover'); hovered = b; if (b) b.classList.add('se-hover'); }
  function blockOf(target) {
    if (!target || !target.closest || target.closest('.se-ui') || target.closest(NAV)) return null;
    const b = target.closest(BLOCKS);
    if (!b || b.closest('.se-ui') || !b.textContent.trim()) return null;
    return b;
  }
  document.addEventListener('mouseover', (ev) => { if (on && !panel.isConnected) hover(blockOf(ev.target)); }, true);
  // Menus, the book's contents and chapter links keep working in edit mode.
  const NAV = 'nav, body > header, .navbar, .bdr-rail, .book-nav, .book-part, .rail-fab, .depth, .jump';
  document.addEventListener('click', (ev) => {
    if (!on || ev.target.closest('.se-ui') || ev.target.closest(NAV)) return;
    const b = blockOf(ev.target);
    ev.preventDefault(); ev.stopPropagation();
    if (b) openBlock(b);
  }, true);
  addEventListener('keydown', (ev) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName) || document.activeElement.isContentEditable;
    if (ev.key === 'e' && !typing && !ev.metaKey && !ev.ctrlKey && !ev.altKey) { setOn(!on); ev.preventDefault(); }
    if (ev.key === 'Escape' && panel.isConnected) { closePanel(true); ev.preventDefault(); }
    if ((ev.metaKey || ev.ctrlKey) && ev.key === 's' && panel.isConnected) { ev.preventDefault(); doSave(); }
  });

  // The text of a block as the reader sees it, with each formula replaced by its TeX (MathJax keeps it).
  function blockText(b) {
    const c = b.cloneNode(true);
    c.querySelectorAll('mjx-assistive-mml, .anchor, .se-ui, script, style, button.box-more').forEach((x) => x.remove());
    c.querySelectorAll('mjx-container').forEach((m) => m.replaceWith(' '));
    return c.textContent.replace(/\s+/g, ' ').trim();
  }

  // ------------------------------------------------------------------ the panel
  const panel = el(`<div class="se-ui se-panel" role="dialog" aria-label="Edit source">
      <div class="se-head">
        <span class="se-file"></span>
        <select class="se-alt" title="Other places where this text was found" hidden></select>
        <span class="se-sp"></span>
        <button type="button" data-p="whole" title="Edit the whole file">Whole file</button>
        <button type="button" data-p="open" title="Open the file at this line in Emacs">Emacs</button>
        <button type="button" data-p="cancel" title="Close without saving (Esc)">Cancel</button>
        <button type="button" data-p="save" class="se-save" title="Save (⌘S)">Save</button>
      </div>
      <div class="se-note"></div>
      <textarea class="se-src" spellcheck="true" autocapitalize="off" autocomplete="off"></textarea>
      <pre class="se-err" hidden></pre>
    </div>`);
  const ta = panel.querySelector('.se-src'), noteEl = panel.querySelector('.se-note'), errEl = panel.querySelector('.se-err');
  const altSel = panel.querySelector('.se-alt');
  let cur = null; // {block, original (clone), match, matches, whole, previewNode}

  async function openBlock(b) {
    if (panel.isConnected) closePanel(true);
    hover(null);
    b.classList.add('se-busy');
    let res;
    try { res = await post('locate', { page: location.pathname, text: blockText(b) }); } catch (e) { res = { ok: false }; }
    b.classList.remove('se-busy');
    if (!res.ok || !res.matches.length) {
      say('This text is not in an editable file: it comes from the theme or is generated (publications come from _bibliography/papers.bib). Try "Page source".', 'warn', 7000);
      return;
    }
    cur = { block: b, original: b.cloneNode(true), matches: res.matches, whole: false, previewNode: null };
    showMatch(0);
    document.body.appendChild(panel);
    document.documentElement.classList.add('se-panel-open');
    const r = b.getBoundingClientRect();
    if (r.top < 60 || r.bottom > innerHeight * 0.52) scrollBy({ top: r.top - 90, behavior: 'smooth' });
    ta.focus();
  }

  function showMatch(i) {
    restoreBlock();
    const m = cur.matches[i];
    cur.match = m; cur.whole = false;
    ta.value = m.source; ta.dataset.old = m.source;
    panel.querySelector('.se-file').textContent = `${m.file} · line ${m.start}${m.end > m.start ? '–' + m.end : ''}`;
    altSel.hidden = cur.matches.length < 2;
    altSel.innerHTML = cur.matches.map((x, k) => `<option value="${k}"${k === i ? ' selected' : ''}>${k ? 'or: ' : 'match: '}${esc(x.file)}:${x.start} (${Math.round(100 * x.score)}%)</option>`).join('');
    const notes = [];
    if (m.score < 0.75) notes.push('The match is approximate: check that this is the text you clicked, or pick another match.');
    if (m.kind === 'code') notes.push('This text sits in a program file: edit only the words inside the quotes.');
    if (m.book) notes.push('Book text: saving rebuilds the book (a few seconds); a save that breaks the build is undone. Math is TeX between \\( \\) or \\[ \\]. This is HTML, not Markdown: a list is <ul><li>…</li><li>…</li></ul> after a blank line, bold is <b>…</b>, a link is <a href="…">…</a>.');
    if (m.kind === 'md') notes.push('Markdown: **bold**, *italic*, [text](link). A blank line starts a new paragraph.');
    if (m.kind === 'bib') notes.push('BibTeX: keep the braces and the comma at the end of each field.');
    noteEl.textContent = notes.join(' ');
    errEl.hidden = true;
    fit();
  }
  altSel.addEventListener('change', () => showMatch(+altSel.value));

  function fit() { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight + 4, innerHeight * 0.4) + 'px'; }

  // ------------------------------------------------------------------ live preview in place
  let md = null;
  function renderSource(src, kind) {
    if (kind === 'md') {
      if (!md && window.markdownit) md = window.markdownit({ html: true, linkify: false, typographer: false });
      const body = src.replace(/^\s*\{:[^}]*\}\s*$/gm, '');
      return md ? md.render(body) : null;
    }
    if (kind === 'html') return src;
    if (kind === 'yml') { const m = src.match(/^\s*-?\s*[\w-]+:\s*(.*)$/); return m ? esc(m[1].replace(/^["']|["']$/g, '')) : null; }
    return null;
  }
  function restoreBlock() {
    if (!cur) return;
    if (cur.previewNode && cur.previewNode.isConnected) cur.previewNode.replaceWith(cur.block);
    cur.previewNode = null;
    if (cur.block.isConnected) cur.block.innerHTML = cur.original.innerHTML;
  }
  let previewTimer = 0;
  function preview() {
    if (!cur || cur.whole) return;
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => {
      const h = renderSource(ta.value, cur.match.kind);
      if (h == null) return;
      const frag = document.createElement('div');
      frag.innerHTML = h;
      // Same element in the source as on the page (a paragraph for a paragraph): keep the page's own
      // element, with its classes, and replace only what is inside it.
      let top = frag;
      while (top.children.length === 1 && !top.firstChild.nodeValue && top.firstElementChild.tagName !== cur.block.tagName &&
             /^(UL|OL|DIV|SECTION)$/.test(top.firstElementChild.tagName)) top = top.firstElementChild;
      const one = top.children.length === 1 ? top.firstElementChild : null;
      let target;
      const inline = (x) => { if (cur.previewNode) { cur.previewNode.replaceWith(cur.block); cur.previewNode = null; } cur.block.innerHTML = x; target = cur.block; };
      if (cur.match.kind === 'yml' || frag.children.length === 0) inline(h);
      // One paragraph, or the same element: keep the page's own element (its classes, a table cell, a
      // list item) and replace only what is inside it.
      else if (one && (one.tagName === cur.block.tagName || one.tagName === 'P')) inline(one.innerHTML);
      else if (/^(TD|TH|LI|DD|DT|CAPTION|SUMMARY|FIGCAPTION|H[1-6])$/.test(cur.block.tagName)) inline(top.innerHTML);
      else {
        const wrap = document.createElement('div'); wrap.className = 'se-preview'; wrap.innerHTML = h;
        (cur.previewNode || cur.block).replaceWith(wrap); cur.previewNode = wrap; target = wrap;
      }
      if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([target]).catch(() => {});
    }, 120);
  }
  ta.addEventListener('input', () => { fit(); preview(); });

  // Paste or drop an image: it is saved under assets/img/uploads/ and a link is inserted.
  async function addFiles(files) {
    for (const f of files) {
      say(`Adding ${f.name || 'image'}…`, '', 0);
      const r = await fetch(API + 'upload?name=' + encodeURIComponent(f.name || 'pasted.png'), { method: 'POST', body: f }).then((x) => x.json());
      if (!r.ok) { say(r.error, 'err', 6000); continue; }
      const kind = cur && cur.match.kind;
      const pdf = r.url.endsWith('.pdf');
      const snippet = kind === 'md' ? (pdf ? `[${f.name}](${r.url})` : `![](${r.url})`) : (pdf ? `<a href="${r.url}">${esc(f.name)}</a>` : `<img src="${r.url}" alt="">`);
      ta.setRangeText(snippet, ta.selectionStart, ta.selectionEnd, 'end');
      ta.dispatchEvent(new Event('input'));
      say('Added ' + r.url.split('/').pop(), 'ok');
    }
  }
  ta.addEventListener('paste', (ev) => { const fs = [...(ev.clipboardData || {}).files || []]; if (fs.length) { ev.preventDefault(); addFiles(fs); } });
  ta.addEventListener('drop', (ev) => { const fs = [...(ev.dataTransfer || {}).files || []]; if (fs.length) { ev.preventDefault(); addFiles(fs); } });

  function closePanel(restore) {
    if (restore) restoreBlock();
    if (cur && cur.block.isConnected) cur.block.classList.remove('se-hover');
    panel.remove(); document.documentElement.classList.remove('se-panel-open');
    cur = null;
  }

  // ------------------------------------------------------------------ save, wait for the rebuild, reload
  async function waitAndReload(t0, label) {
    try { sessionStorage.setItem(SCROLL_KEY, String(scrollY)); } catch (e) { /* ignore */ }
    say(label + ' · rebuilding the page…', '', 0);
    for (let k = 0; k < 240; k++) {
      await new Promise((r) => setTimeout(r, 500));
      try { const m = await get('mtime?page=' + encodeURIComponent(location.pathname)); if (m.mtime > t0) break; } catch (e) { /* jekyll busy */ }
    }
    await new Promise((r) => setTimeout(r, 400));
    location.reload();
  }
  async function doSave() {
    if (!cur) return;
    const btn = panel.querySelector('.se-save');
    btn.disabled = true; btn.textContent = cur.match.book ? 'Saving and building…' : 'Saving…';
    errEl.hidden = true;
    const t0 = Date.now() / 1000;
    const req = cur.whole ? { file: cur.match.file, whole: true, old: ta.dataset.old, new: ta.value }
      : { file: cur.match.file, start: cur.match.start, end: cur.match.end, old: ta.dataset.old, new: ta.value };
    let r;
    try { r = await post('save', req); } catch (e) { r = { ok: false, error: 'The editor is not running (bin/site-edit).' }; }
    btn.disabled = false; btn.textContent = 'Save';
    if (!r.ok) { errEl.textContent = r.error; errEl.hidden = false; return; }
    if (r.unchanged) { closePanel(true); say('No change.'); return; }
    closePanel(false);
    waitAndReload(t0, 'Saved ' + r.file);
  }

  panel.addEventListener('click', async (ev) => {
    const a = ev.target.closest('button') && ev.target.closest('button').dataset.p;
    if (a === 'save') doSave();
    if (a === 'cancel') closePanel(true);
    if (a === 'open') post('open', { file: cur.match.file, line: cur.match.start });
    if (a === 'whole' && cur && !cur.whole) {
      const r = await get('file?file=' + encodeURIComponent(cur.match.file));
      if (!r.ok) return say(r.error, 'err');
      restoreBlock();
      const line = cur.match.start;
      cur.whole = true; ta.value = r.content; ta.dataset.old = r.content;
      panel.querySelector('.se-file').textContent = r.file + ' · whole file';
      noteEl.textContent = 'The whole file. The page is redrawn after saving.';
      ta.style.height = innerHeight * 0.4 + 'px';
      const pos = r.content.split('\n').slice(0, line - 1).join('\n').length;
      ta.setSelectionRange(pos + 1, pos + 1); ta.focus();
      ta.scrollTop = Math.max(0, (line - 4) * parseFloat(getComputedStyle(ta).lineHeight || 18));
    }
  });

  async function openWhole() {
    const r = await get('file?page=' + encodeURIComponent(location.pathname));
    if (!r.ok) return say(r.error, 'warn', 6000);
    if (panel.isConnected) closePanel(true);
    cur = { block: document.createElement('div'), original: document.createElement('div'), matches: [{ file: r.file, start: 1, end: 1, source: r.content, kind: r.kind, book: r.book, score: 1 }], whole: false };
    showMatch(0);
    document.body.appendChild(panel); document.documentElement.classList.add('se-panel-open');
    panel.querySelector('[data-p="whole"]').click();
  }

  // ------------------------------------------------------------------ toolbar actions
  const modal = el('<div class="se-ui se-modal" role="dialog" aria-modal="true"><div class="se-card"></div></div>');
  function showModal(html) { modal.querySelector('.se-card').innerHTML = html; document.body.appendChild(modal); return modal.querySelector('.se-card'); }
  modal.addEventListener('click', (ev) => { if (ev.target === modal || ev.target.closest('[data-m="close"]')) modal.remove(); });

  bar.addEventListener('click', async (ev) => {
    const a = ev.target.closest('button') && ev.target.closest('button').dataset.act;
    if (a === 'page') openWhole();
    if (a === 'undo') {
      const t0 = Date.now() / 1000;
      const r = await post('undo', {});
      if (!r.ok) return say(r.error, 'warn');
      waitAndReload(t0, 'Undid the last save in ' + r.file);
    }
    if (a === 'news') {
      const card = showModal(`<h3>New news item</h3><p>One or two sentences for the homepage, dated today. Markdown works: [text](https://…).</p>
        <textarea class="se-in" rows="4"></textarea><div class="se-row"><button data-m="close">Cancel</button><button class="se-go">Add</button></div>`);
      card.querySelector('textarea').focus();
      card.querySelector('.se-go').addEventListener('click', async () => {
        const t0 = Date.now() / 1000;
        const r = await post('news', { text: card.querySelector('textarea').value });
        if (!r.ok) return say(r.error, 'err', 6000);
        modal.remove(); waitAndReload(t0, 'Added ' + r.file);
      });
    }
    if (a === 'publish') {
      const c = await get('changes');
      const li = (x) => `<li><code>${esc(x.status)}</code> ${esc(x.path)}</li>`;
      const any = c.site.length || c.book.length || c.book_ahead;
      const card = showModal(`<h3>Publish</h3>
        ${c.site.length ? `<p>Website files to commit and push:</p><ul>${c.site.map(li).join('')}</ul>` : '<p>No website file has changed.</p>'}
        ${c.book.length || c.book_ahead ? `<label class="se-check"><input type="checkbox" class="se-book" checked> Also commit and push the book's private text source (diffusion-roofline)${c.book_ahead ? `, ${c.book_ahead} earlier commit(s) not yet pushed` : ''}</label>` : ''}
        ${c.other.length ? `<p class="se-dim">Changed but not published (outside the content folders): ${c.other.map((x) => esc(x.path)).join(', ')}</p>` : ''}
        ${any ? '<label>Description of the change<input class="se-in se-msg" value="Website: edits"></label>' : ''}
        <div class="se-row"><button data-m="close">Close</button>${any ? '<button class="se-go">Publish</button>' : ''}</div><pre class="se-out" hidden></pre>`);
      const go = card.querySelector('.se-go');
      if (go) go.addEventListener('click', async () => {
        go.disabled = true; go.textContent = 'Publishing…';
        const bk = card.querySelector('.se-book');
        const r = await post('publish', { message: card.querySelector('.se-msg').value, book: bk ? bk.checked : false });
        const out = card.querySelector('.se-out'); out.hidden = false; out.textContent = r.ok ? r.message : r.error;
        out.className = 'se-out ' + (r.ok ? 'ok' : 'err');
        go.remove(); refreshCount();
      });
    }
    if (a === 'help') {
      showModal(`<h3>Editing the website</h3><ul>
        <li><b>Edit</b> (or the E key) turns edit mode on. Click any paragraph, heading, list item, caption or cell: the panel below shows the source text that produced it, and the block is redrawn as you type.</li>
        <li><b>Save</b> (⌘S) writes the file; the page reloads at the same place once the site is rebuilt. <b>Esc</b> closes without saving. <b>Undo</b> reverts the last save, as many times as needed.</li>
        <li>Paste or drop an image (or a PDF) into the source: it is saved under <code>assets/img/uploads/</code> and linked.</li>
        <li><b>Page source</b> edits the whole file of the page (to add a section, say); <b>Emacs</b> opens the file at that line.</li>
        <li><b>+ News</b> adds a dated news item to the homepage. Publications are entries of <code>_bibliography/papers.bib</code>: click a title to edit its entry.</li>
        <li>Nothing is online until <b>Publish</b>, which commits and pushes; the site is live about 3 minutes later.</li>
        <li>Book pages: the text comes from the private source <code>~/code/diffusion-roofline/site/block-diffusion-rooflines.html</code> and the files the build reads; each save rebuilds the book, and a save that breaks the build is undone.</li></ul>
        <div class="se-row"><button data-m="quit" class="se-dim-btn">Stop the editor</button><button data-m="close">Close</button></div>`);
      modal.querySelector('[data-m="quit"]').addEventListener('click', async () => { await post('quit', {}); modal.remove(); say('The editor is stopped. Start it again with the Edit Website app.', 'warn', 0); });
    }
  });

  // ------------------------------------------------------------------ start
  try { if (sessionStorage.getItem(ON_KEY)) setOn(true); } catch (e) { /* ignore */ }
  try {
    const y = sessionStorage.getItem(SCROLL_KEY);
    if (y !== null) {
      sessionStorage.removeItem(SCROLL_KEY);
      const go = () => scrollTo(0, +y);
      go(); addEventListener('load', go); setTimeout(go, 800); setTimeout(go, 2000);
      say('Saved and rebuilt.', 'ok', 2500);
    }
  } catch (e) { /* ignore */ }
})();
