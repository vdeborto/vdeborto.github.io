"""The symbols of the Block Diffusion Rooflines book, from one file: _roofline/symbols.txt.

That file is read by bin/build-roofline.py, which makes two things from it:

  the Symbols page          section(): one table per group (symbol, meaning, where it is defined);
  assets/roofline/symbols.js key_js(): the list behind the symbol key of assets/roofline/reading.js
                             (a click on a formula lists the symbols it uses, and so does the margin).

Format of _roofline/symbols.txt. Lines starting with "#" are comments. "intro:" is the page's opening
paragraph. "== Title" starts a group (a heading and a table on the page). An entry is the symbol in
TeX on a line of its own, followed by indented "field: value" lines:

  what:   the meaning, in HTML with \\( \\) math (required)
  in:     where the symbol is introduced: "Part 2", "Definition 2", "Proposition 8" (required for the
          page; the build checks that a formula there uses the symbol, and links it)
  match:  JavaScript regular expression on a formula's TeX; with it, the symbol is in the symbol key
  rank:   order in the symbol key, which lists at most 14 symbols of a formula, lowest rank first
  scope:  T = only in Parts 1-5, P = only in Parts 6-9 (a letter that means two things)
  key:    the symbol as the key shows it, when it differs from the page's (TeX)
  page:   "no" keeps the entry out of the page (key only)

A symbol that is not math is written "plain R1–R8" (page only). The key gets TeX, which reading.js
has MathJax typeset when it shows an entry, so a symbol looks the same there as in the formulas.
"""
import html as _html
import json
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "_roofline" / "symbols.txt"
FIELDS = {"what", "in", "match", "rank", "scope", "key", "page"}


def parse(text):
    """-> (intro, groups, entries, problems); an entry is a dict with its group and source line."""
    intro, groups, entries, problems = "", [], [], []
    cur = None
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("intro:"):
            intro = line[6:].strip()
        elif line.startswith("=="):
            groups.append(line[2:].strip())
        elif line[0] in " \t":
            m = re.match(r"\s+([a-z]+):\s*(.*)$", line)
            if not m or m.group(1) not in FIELDS or cur is None:
                problems.append(f"symbols.txt line {n}: not a field of an entry: {line.strip()[:60]!r}")
            elif m.group(1) in cur:
                problems.append(f"symbols.txt line {n}: {m.group(1)!r} given twice for {cur['sym']}")
            else:
                cur[m.group(1)] = m.group(2).strip()
        else:
            if not groups:
                problems.append(f"symbols.txt line {n}: entry before the first group")
                groups.append("Symbols")
            sym = line.strip()
            cur = {"sym": sym[6:].strip() if sym.startswith("plain ") else sym, "plain": sym.startswith("plain "),
                   "group": groups[-1], "line": n}
            entries.append(cur)
    ranks = {}
    for e in entries:
        where = f"symbols.txt line {e['line']} ({e['sym']})"
        if not e.get("what"):
            problems.append(f"{where}: no meaning (what:)")
        if e.get("page", "yes") not in ("yes", "no"):
            problems.append(f"{where}: page is {e['page']!r}, expected no")
        if e.get("page") != "no" and not e.get("in"):
            problems.append(f"{where}: no place of definition (in:)")
        if e.get("scope", "") not in ("", "T", "P"):
            problems.append(f"{where}: scope is {e['scope']!r}, expected T or P")
        if ("match" in e) != ("rank" in e):
            problems.append(f"{where}: match and rank go together")
        if e["plain"] and "match" in e:
            problems.append(f"{where}: a plain symbol cannot be in the symbol key")
        if e.get("page") == "no" and "match" not in e:
            problems.append(f"{where}: neither on the page nor in the key")
        if "rank" in e:
            if not e["rank"].isdigit():
                problems.append(f"{where}: rank is not a number")
            elif e["rank"] in ranks:
                problems.append(f"{where}: rank {e['rank']} is also that of {ranks[e['rank']]}")
            ranks[e["rank"]] = e["sym"]
            try:
                re.compile(e["match"])
            except re.error as err:
                problems.append(f"{where}: match is not a regular expression ({err})")
    return intro, groups, entries, problems


def load():
    return parse(SRC.read_text())


# ---------------------------------------------------------------- the symbol key
def key_rows(entries, color=None):
    """[pattern, symbol in TeX, meaning, scope] for each entry of the symbol key, in rank order.
    color() wraps AR and TD quantities as the build does in the text, so the key shows them alike."""
    color = color or (lambda t: t)
    rows = []
    for e in sorted((e for e in entries if "match" in e), key=lambda e: int(e["rank"])):
        sym = color(f"\\({e.get('key', e['sym'])}\\)")
        rows.append([e["match"], sym[2:-2], color(e["what"]), e.get("scope", "")])
    return rows


def key_js(entries, color=None):
    rows = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in key_rows(entries, color))
    return ("// Generated by bin/build-roofline.py from _roofline/symbols.txt. Do not edit.\n"
            "// [pattern on a formula's TeX, symbol in TeX, meaning, scope]: the symbol key of reading.js.\n"
            f"window.BDR_SYMBOLS = [\n{rows}\n];\n")


# ---------------------------------------------------------------- the page
def section(intro, groups, entries):
    """The Symbols chapter, as a section of the note (class names as in the note's source)."""
    s = ['<section class="col" id="symbols">', "  <h2>Symbols</h2>", f"  <p>{intro}</p>"]
    for g in groups:
        rows = [e for e in entries if e["group"] == g and e.get("page") != "no"]
        if not rows:
            continue
        s.append(f"  <h3>{g}</h3>")
        s.append('  <div class="table-wrap"><table class="symbols-table sym-sheet">')
        s.append('    <thead><tr><th class="l">Symbol</th><th class="l">Meaning</th><th class="l">Introduced</th></tr></thead>')
        s.append("    <tbody>")
        for e in rows:
            sym = _html.escape(e["sym"]) if e["plain"] else f"\\({e['sym']}\\)"
            s.append(f'      <tr><td>{sym}</td><td class="l">{e["what"]}</td><td class="l sym-in">{e["in"]}</td></tr>')
        s.append("    </tbody>")
        s.append("  </table></div>")
    s.append("</section>")
    return "\n".join(s)


# ---------------------------------------------------------------- the check of "in:"
_MATH = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", re.S)


def uses(e, text):
    """Does a formula of this HTML use the entry's symbol (its pattern, or the symbol itself)?"""
    if e["plain"]:  # "R1–R8": the first of them is named in the text
        return re.split(r"[–-]", e["sym"])[0] in re.sub(r"<[^>]+>", " ", text)
    rx = re.compile(e["match"]) if "match" in e else None
    for a, b in _MATH.findall(text):
        tex = re.sub(r"\\text\{[^}]*\}", "", a or b)
        if rx is None:
            if all(part.strip() in tex for part in re.split(r",\\ |,\s", e["sym"])):
                return True
        elif rx.search(tex if "hat" in e["match"] else re.sub(r"\\hat\{?\\(pi|beta)\}?", "HAT", tex)):
            return True
    return False
