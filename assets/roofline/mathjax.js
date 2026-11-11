// Runs after the site's mathjax-setup.js and before MathJax itself (both deferred scripts run in
// document order): on this page only \( \) and \[ \] delimit math, since the note's prose and the
// calculator use "$" for other things, and MathJax leaves the figures' SVG alone.
(function () {
  const mj = window.MathJax;
  if (!mj || mj.version) return; // MathJax already loaded: too late to configure
  mj.tex = Object.assign({}, mj.tex, { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']], tags: 'none' });
  mj.options = Object.assign({}, mj.options, { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'svg'] });
})();
