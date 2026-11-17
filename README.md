# vdeborto.github.io

Personal academic site, built with [Jekyll](https://jekyllrb.com/) and the
[al-folio](https://github.com/alshedivat/al-folio) theme.

## How it works

- **This branch holds the source.** The site is *built* by GitHub Actions
  (`.github/workflows/deploy.yml`) and the generated HTML is pushed to the
  `gh-pages` branch. GitHub Pages serves `gh-pages`. Never edit `gh-pages` by hand.
- Push to `main` → the action rebuilds and redeploys, usually within two minutes.

## Where things live

| What | File |
|---|---|
| Bio, short CV, homepage | `_pages/about.md` |
| Publications | `_bibliography/papers.bib` |
| News items on the homepage | `_news/*.md` |
| Teaching page (generated list) | `_pages/teaching.md` |
| Teaching PDFs (91 files, 4 courses) | `teaching/` |
| Name, URL, socials, site settings | `_config.yml`, `_data/socials.yml` |
| Profile photo | `assets/img/prof_pic.jpg` |
| Custom theme (palette, type, blur-in) | `assets/css/custom.css` |
| Publication entry template | `_layouts/bib.liquid` |
| CV PDF | `assets/pdf/cv.pdf` |
| Unlisted Block Diffusion Rooflines note (no links, noindex, not in sitemap), with interactive figures, published as a short book: the Introduction as first page, then one page per section (`summary/`, which also holds the headline numbers, `setup/`, … `appendix/`) with Previous / Next links; `introduction/` forwards to the first page | built by `python3 -I bin/build-roofline.py` (the `CHAPTERS` list maps sections to pages; links between pages are rewritten, and old single-page anchors are forwarded) into `notes/block-diffusion-rooflines-6f924020/` and `assets/roofline/note.{css,js}`; text, cost model and calculator come from the private `diffusion-roofline` repo; figures and the accounting table in `_roofline/figures.html` and `assets/roofline/figures.js` (the build places each figure at an anchor of the text); figures and tables of the practice chapters (Parts 0, 6–9) in `bin/roofline_practice.py`, from `_roofline/practice-data.json` (exported by `pages/book/export.py` in the roofscope campaign-2 checkout); card panels in `_roofline/cards.html`, chapter introductions in `_roofline/intros.html`; every symbol in `_roofline/symbols.txt`, from which `bin/roofline_symbols.py` makes the Symbols page (`symbols/`, the last page: one table per group, each symbol linked to where it is defined, which the build checks) and `assets/roofline/symbols.js`, the list of the symbol key; contents (rail or phone sheet), the symbol key, the right margin on wide windows (footnotes as sidenotes beside the sentence that cites them, with the Notes list hidden; the symbols of the display formula nearest the reading line), the citation's copy button and the comment thread in `assets/roofline/reading.js`; the citation block (first page) and the comment thread of each chapter (giscus: one GitHub Discussion per chapter in this repository, category Announcements; needs Discussions enabled and the giscus app installed here; ids in `GISCUS` in the build script) are written by the build; palette in `assets/roofline/theme.css`; AR/TD quantities in the math are coloured by the build. The measurement charts of Parts 6–9 (gain from bound to measurement, routing, Proposition 1 against the measurement, the two efficiencies of Proposition 9, component efficiency, recoverable time, kernel speed-up) are drawn by `assets/roofline/measured.js` from the `#bdr-practice` JSON that the build writes into those chapters; `roofline_practice.py` places them as `data-mfig` blocks, and tables that back a figure collapse under it (`fold=`). The build also cards the note's own static figures (`LEGACY_CLAIMS` gives their titles), checks that figures and tables are numbered in reading order, and links every Figure, Table, Proposition, Definition, Corollary, Remark and Part reference to its target (`link_references`); `reading.js` shows a preview of a reference on hover, opens a folded target, lists the current chapter's sections in the rail and swaps a wide equation for its narrow form when it does not fit the column. Code cards in Parts 6–9 quote the roofscope harness (github.com/vdeborto/roofscope, private): each `<!-- code:NAME -->` in the note becomes the card NAME of `bin/roofline_code.py`, whose line ranges are given by regular expressions and read with `git show` from `~/code/roofscope` at the commit `SHA`, so excerpts, line numbers and GitHub links agree; to move to a newer commit, push it, set `SHA` and rebuild. The roofscope logo is drawn there too (`logo()`, in the book's colour tokens). Figures 1 and 2 have phone layouts (`v-wide` / `v-narrow`); other wide diagrams and tables scroll on phones, with a hint and edge fades from `reading.js`. Rule kept by the figures: intensities are always Proposition 1's, exact gains are labelled as such, every gain is called a roofline bound |

### Adding a paper

Append a BibTeX entry to `_bibliography/papers.bib`. Useful extra fields:

- `abbr={NeurIPS}` — the coloured venue badge
- `selected={true}` — also show it on the homepage
- `arxiv={2502.02483}` — adds an arXiv button
- `code={https://github.com/...}`, `pdf=`, `website=` — extra buttons

## Editing in the browser

The in-browser editor (the *Edit Website* app, `edit-site` in fish) lives in the private repository
`vdeborto/site-edit`, cloned at `~/code/site-edit`. It edits this repository's content files and the
rooflines book's sources. The book's text files accept Markdown on top of their HTML through
`bin/mdlite.py` (emphasis, links, code, lists, `###` headings, blank-line paragraphs), which
`bin/build-roofline.py` runs and the editor's preview imports; it leaves the existing HTML unchanged.
The standalone private copy of the book (`~/code/diffusion-roofline`) shows such Markdown as typed.

## Building locally

```bash
bundle install
bundle exec jekyll serve      # http://localhost:4000
```

Requires Ruby 3.3 and ImageMagick. On a machine with a non-UTF-8 locale you may
need `LANG=C.UTF-8 LC_ALL=C.UTF-8` or the build fails reading accented filenames.
