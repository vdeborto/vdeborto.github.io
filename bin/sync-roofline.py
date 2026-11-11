#!/usr/bin/env python3
"""Copy the Block Diffusion Rooflines note from ~/code/diffusion-roofline into this site.

The page is published unlisted: no link from the site, noindex meta, excluded from the sitemap
and from jekyll-minifier (see _config.yml). It is served at
    /notes/block-diffusion-rooflines-6f924020/
The source of truth stays in the private repo vdeborto/diffusion-roofline; rerun this after
changing it there, then commit and push.

Changes made to the copy:
  - MathJax is loaded from cdnjs (the source site vendors it);
  - links back to the private site's index are removed;
  - the sentence naming the private repository is replaced.
Usage: python3 -I bin/sync-roofline.py   (exit 1 if an expected pattern is missing)
"""
import sys
from pathlib import Path

SRC = Path.home() / "code/diffusion-roofline/site/block-diffusion-rooflines.html"
DST = Path(__file__).resolve().parent.parent / "notes/block-diffusion-rooflines-6f924020/index.html"
MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg-full.js"

EDITS = [
    ('<script src="vendor/mathjax-3.2.2/tex-svg-full.js" defer onerror="var s=document.createElement(\'script\');'
     's.src=\'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg-full.js\';document.head.appendChild(s)"></script>',
     f'<script src="{MATHJAX}" defer></script>'),
    ('<a href="index.html">← Notes</a>', ''),
    ('<a class="rail-home" href="index.html">← All notes</a>', ''),
    ('and the site lives in the private GitHub repository <code>vdeborto/diffusion-roofline</code>, whose history is the changelog.',
     'and the page has no other dependency.'),
]


def main():
    page = SRC.read_text()
    missing = [old for old, _ in EDITS if page.count(old) != 1]
    for old in missing:
        print("PROBLEM: expected exactly one occurrence of:", old[:100])
    if missing:
        return 1
    for old, new in EDITS:
        page = page.replace(old, new)
    for leak in ("vendor/", 'href="index.html"', "diffusion-roofline"):
        if leak in page:
            print("PROBLEM: copy still contains", repr(leak))
            return 1
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(page)
    print(f"wrote {DST} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
