#!/usr/bin/env python3
"""Shift the rooflines book's figure numbers, to make room for a new figure.

    python3 bin/renumber-figures.py FROM        # every Figure N with N >= FROM becomes N + 1

Figure numbers are written out in the text ("Figure 23", "Figures 5, 9 and 13", #figure-23) and the
build checks that they run 1, 2, 3, ... in reading order. This rewrites every file that holds them:
the private text source, the _roofline/*.html files, the figure code and the build (including
LEGACY_CLAIMS, keyed by number). Then add the new figure as Figure FROM and build. Check: the new
output equals the old output passed through R(), plus the inserted figure (done for Figure 2 on
2026-11-16).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [Path.home() / "code/diffusion-roofline/site/block-diffusion-rooflines.html"] + [ROOT / p for p in (
    "_roofline/intros.html", "_roofline/gists.html", "_roofline/cards.html", "_roofline/figures.html",
    "bin/roofline_practice.py", "bin/build-roofline.py", "bin/roofline_code.py",
    "assets/roofline/figures.js", "assets/roofline/measured.js", "assets/roofline/reading.js")]
REF = re.compile(r"\b(Fig(?:ure|\.)?s?)((?:\s|&nbsp;)+)(\d+(?:(?:\s*(?:–|,|and|or)\s*)\d+)*)")


def R(t, start):
    bump = lambda n: str(int(n) + 1) if int(n) >= start else n
    t = REF.sub(lambda m: m.group(1) + m.group(2) + re.sub(r"\d+", lambda k: bump(k.group(0)), m.group(3)), t)
    return re.sub(r"(figure-)(\d+)", lambda m: m.group(1) + bump(m.group(2)), t)


def main():
    start = int(sys.argv[1])
    for p in FILES:
        t = p.read_text()
        u = R(t, start)
        if p.name == "build-roofline.py":  # LEGACY_CLAIMS = {14: ..., 19: ...} and LEGACY_CLAIMS[19]
            u = re.sub(r"^(    )(\d+)(: )", lambda m: m.group(1) + (str(int(m.group(2)) + 1) if int(m.group(2)) >= start else m.group(2)) + m.group(3), u, flags=re.M)
            u = re.sub(r"LEGACY_CLAIMS\[(\d+)\]", lambda m: f"LEGACY_CLAIMS[{int(m.group(1)) + 1 if int(m.group(1)) >= start else m.group(1)}]", u)
        if u != t:
            p.write_text(u)
            print("renumbered", p)


if __name__ == "__main__":
    main()
