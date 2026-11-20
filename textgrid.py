"""Minimal reader/writer for Praat long-format TextGrids (MFA's default output)."""
import re

_ITEM = re.compile(r'item \[\d+\]:\s*class = "(\w+)"\s*name = "(.*?)"(.*?)(?=item \[\d+\]:|\Z)', re.S)
_IV = re.compile(r'xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "((?:[^"]|"")*)"', re.S)


def read(path):
    """Returns {tier_name: [(start, end, label), ...]} (interval tiers only)."""
    s = open(path, encoding="utf-8").read()
    if "item [" not in s:
        s = open(path, encoding="utf-16").read()
    tiers = {}
    for cls, name, body in _ITEM.findall(s):
        if cls != "IntervalTier":
            continue
        tiers[name] = [(float(a), float(b), t.replace('""', '"')) for a, b, t in _IV.findall(body)]
    return tiers


def tier(tiers, suffix):
    """MFA names tiers 'words'/'phones' or '<speaker> - words'."""
    for name, ivs in tiers.items():
        if name == suffix or name.endswith(" - " + suffix):
            return ivs
    raise KeyError(f"no tier '{suffix}' in {list(tiers)}")


def write(path, xmax, tiers):
    """tiers: list of (name, [(s, e, label), ...]); gaps are filled with empty intervals."""
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
           "xmin = 0", f"xmax = {xmax}", "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for i, (name, ivs) in enumerate(tiers, 1):
        full, t = [], 0.0
        for s, e, lab in sorted(ivs):
            s, e = max(s, t), min(e, xmax)
            if s > t + 1e-6:
                full.append((t, s, ""))
            if e > s:
                full.append((s, e, lab))
                t = e
        if t < xmax - 1e-6:
            full.append((t, xmax, ""))
        out += [f"    item [{i}]:", '        class = "IntervalTier"', f'        name = "{name}"',
                "        xmin = 0", f"        xmax = {xmax}", f"        intervals: size = {len(full)}"]
        for j, (s, e, lab) in enumerate(full, 1):
            lab = lab.replace('"', '""')
            out += [f"        intervals [{j}]:", f"            xmin = {s:.4f}",
                    f"            xmax = {e:.4f}", f'            text = "{lab}"']
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
