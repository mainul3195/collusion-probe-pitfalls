"""List typed decimals in LaTeX sources that match no current result value.

Every number that comes from an experiment should be a \\R{key} macro (numbers.tex). Typed numbers are allowed for
constants (layer counts, hyper-parameters, literature values), but a typed decimal such as 0.84 that matches no value
in numbers.tex is either stale, a rounding of something else, or not from our results at all; it is listed with its
file and line so it can be checked. Typed decimals that DO match a result are listed separately (count only) as
candidates for conversion to \\R{}.

Usage:  python papers/shared/check_numbers.py thesis/chapters/*.tex [--show-matching]
"""
import re
import sys
from pathlib import Path

NUM = re.compile(r"(?<![\w.{])(\d?\.\d{2,3})(?![\d}])")
HERE = Path(__file__).resolve().parent


def values():
    vals = set()
    for line in (HERE / "numbers.tex").read_text().splitlines():
        m = re.match(r"\\@namedef\{rnum@[^}]+\}\{(.*)\}$", line)
        if m:
            vals.update(re.findall(r"\d?\.\d+", m.group(1)))
    return vals


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show-matching" in sys.argv
    vals = values()
    unknown, matching = [], []
    for f in args:
        for i, line in enumerate(Path(f).read_text().splitlines(), 1):
            code = line.split("%")[0] if not line.lstrip().startswith("%") else ""
            code = re.sub(r"\\R\{[^}]*\}", "", code)
            for m in NUM.finditer(code):
                v = m.group(1)
                (matching if v in vals or v.lstrip("0") in vals or ("0" + v) in vals else unknown).append((f, i, v, code.strip()))
    for f, i, v, c in unknown:
        print(f"UNMATCHED {f}:{i}  {v}   | {c[:110]}")
    print(f"\n{len(unknown)} typed decimals match no result; {len(matching)} match a result (convert to \\R{{}})")
    if show:
        for f, i, v, c in matching:
            print(f"match {f}:{i}  {v}")


if __name__ == "__main__":
    main()
