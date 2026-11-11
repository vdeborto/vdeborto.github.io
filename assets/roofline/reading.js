// Reading aids for the Block Diffusion Rooflines page:
//  - the intuition / full-math switch: in intuition mode every definition and proposition folds
//    to its title and one-sentence summary, with a button to open it;
//  - the contents rail (wide screens): sections, the current one's subsections and figures, and
//    a reading-progress bar.
(function () {
  'use strict';
  const page = document.querySelector('.bdr');
  if (!page) return;
  const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const KEY = 'bdr-depth';

  // ---------------------------------------------------------------- reading depth
  const boxes = [...page.querySelectorAll('.box.prop, .box.def')].filter((b) => b.querySelector('.gist'));
  for (const box of boxes) {
    const btn = el('<button type="button" class="box-more" aria-expanded="false">show the statement +</button>');
    btn.addEventListener('click', () => {
      const on = !box.classList.contains('open');
      box.classList.toggle('open', on);
      btn.setAttribute('aria-expanded', String(on));
      btn.textContent = on ? 'hide the statement −' : 'show the statement +';
    });
    box.querySelector('.gist').after(btn);
  }
  const switches = [];
  function setDepth(depth, save) {
    page.classList.toggle('intuition', depth === 'intuition');
    switches.forEach((s) => s.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.depth === depth))));
    if (save) { try { localStorage.setItem(KEY, depth); } catch (e) { /* storage unavailable */ } }
  }
  function depthSwitch(cls) {
    const s = el(`<div class="depth ${cls}" role="group" aria-label="Reading depth"><span>reading</span><button type="button" data-depth="intuition">intuition</button><button type="button" data-depth="full">full math</button></div>`);
    s.addEventListener('click', (ev) => { const d = ev.target.dataset && ev.target.dataset.depth; if (d) setDepth(d, true); });
    switches.push(s);
    return s;
  }
  const hero = page.querySelector('.hero .jump');
  if (hero) hero.after(depthSwitch('depth-hero'));

  // ---------------------------------------------------------------- contents rail
  const sections = [...page.querySelectorAll('section[id]')].map((sec) => {
    const h = sec.querySelector('h2');
    const label = h ? [...h.childNodes].filter((n) => !(n.classList && n.classList.contains('anchor'))).map((n) => n.textContent).join('').replace(/\s+/g, ' ').trim() : sec.id;
    const subs = [...sec.querySelectorAll('h3[id], .bdr-fig[id]')].map((x) => ({
      id: x.id,
      label: x.classList.contains('bdr-fig') ? (x.querySelector('.fig-k').textContent.replace(/·.*$/, '').trim() || 'Quiz') : x.textContent.replace('#', '').trim(),
      fig: x.classList.contains('bdr-fig'),
    }));
    return { id: sec.id, label, subs };
  });
  const quiz = page.querySelector('#fig-quiz');
  const rail = el('<nav class="bdr-rail" aria-label="Contents"><div class="rail-progress"><i></i></div><a class="rail-top" href="#top">Block Diffusion Rooflines</a><ol></ol></nav>');
  const ol = rail.querySelector('ol');
  if (quiz) ol.appendChild(el('<li><a href="#fig-quiz" data-id="fig-quiz">Test your intuition</a></li>'));
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
  document.body.appendChild(rail);

  const bar = rail.querySelector('.rail-progress i');
  const links = [...rail.querySelectorAll('a[data-id]')];
  const targets = links.map((a) => [a, document.getElementById(a.dataset.id)]).filter(([, t]) => t);
  function update() {
    const h = document.documentElement.scrollHeight - innerHeight;
    bar.style.width = `${h > 0 ? Math.min(100, (100 * scrollY) / h) : 0}%`;
    // current = last section heading above a line 30% down the window; same for subsections
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
  update();

  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* storage unavailable */ }
  setDepth(saved === 'intuition' ? 'intuition' : 'full', false);
})();
