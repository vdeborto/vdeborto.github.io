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

### Adding a paper

Append a BibTeX entry to `_bibliography/papers.bib`. Useful extra fields:

- `abbr={NeurIPS}` — the coloured venue badge
- `selected={true}` — also show it on the homepage
- `arxiv={2502.02483}` — adds an arXiv button
- `code={https://github.com/...}`, `pdf=`, `website=` — extra buttons

## Building locally

```bash
bundle install
bundle exec jekyll serve      # http://localhost:4000
```

Requires Ruby 3.3 and ImageMagick. On a machine with a non-UTF-8 locale you may
need `LANG=C.UTF-8 LC_ALL=C.UTF-8` or the build fails reading accented filenames.
