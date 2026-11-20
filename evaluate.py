#!/usr/bin/env python3
"""Gold-standard templates and boundary evaluation.

1) Make Praat templates for annotators (pre-filled with ASR timings, so any
   anchoring bias favours the baseline, not MFA):
   python3 evaluate.py template --jsonl HR.mfa.jsonl --work work/HR --keys gold.list --out gold_todo/HR

   Annotators correct tier 'words' (one interval per token, labels unchanged) and
   tier 'chars' (one interval per letter; a multi-letter sound may be one interval
   labelled with all its letters, e.g. "sz"; a silent letter is merged into its
   neighbour's interval, e.g. "ni"). Then save as long-format TextGrid.

2) Score systems against gold:
   python3 evaluate.py score --jsonl HR.mfa.jsonl --work work/HR --gold gold_done/HR

Systems: mfa (words_mfa/chars_mfa), old (words_align/chars_align), asr (words).
"""
import argparse, json, os, statistics, sys
import textgrid

SKIP = {"punct", "fp", "annotation"}


def load(jsonl, work):
    meta = {}
    for l in open(f"{work}/meta/tokens.jsonl", encoding="utf-8"):
        m = json.loads(l); meta[m["key"]] = m
    recs = {}
    for l in open(jsonl, encoding="utf-8"):
        e = json.loads(l); recs[e["id"]] = e
    return meta, recs


def span(lst, cs, ce):
    """time span of entries in lst overlapping chars [cs, ce)."""
    ov = [w for w in (lst or []) if w and w["char_s"] < ce and cs < w["char_e"]]
    return (min(w["time_s"] for w in ov), max(w["time_e"] for w in ov)) if ov else None


def gold_tokens(m):
    return [t for t in m["tokens"] if t["kind"] not in SKIP]


def letters(e, m):
    out = []
    for t in gold_tokens(m):
        if t["kind"] == "word":
            for off in sorted(set(t["norm_offs"])):
                out.append((e["text"][off].lower(), off))
    return out


def flat_chars(cs):
    return [c for w in (cs or []) if w for c in w]


def cmd_template(a):
    meta, recs = load(a.jsonl, a.work)
    os.makedirs(a.out, exist_ok=True)
    for key in [k.strip() for k in open(a.keys) if k.strip()]:
        m = meta[key]; e = recs[m["id"]]
        wt, ct = [], []
        for t in gold_tokens(m):
            sp = span(e["words"], t["char_s"], t["char_e"])
            if not sp:
                continue
            wt.append((sp[0], sp[1], e["text"][t["char_s"]:t["char_e"]]))
            if t["kind"] == "word":
                offs = sorted(set(t["norm_offs"]))
                step = (sp[1] - sp[0]) / len(offs)
                ct += [(sp[0] + i * step, sp[0] + (i + 1) * step, e["text"][o]) for i, o in enumerate(offs)]
        textgrid.write(f"{a.out}/{key}.TextGrid", e["audio_length"], [("words", wt), ("chars", ct)])
    print("templates written to", a.out)


def report(name, errs):
    if not errs:
        return f"{name:5s}  n=0"
    ms = [x * 1000 for x in errs]
    w = lambda t: 100 * sum(x <= t + 1e-6 for x in ms) / len(ms)
    return (f"{name:5s}  n={len(ms):6d}  MAE={statistics.mean(ms):6.1f} ms  median={statistics.median(ms):5.1f}  "
            f"<=10ms {w(10):5.1f}%  <=20ms {w(20):5.1f}%  <=50ms {w(50):5.1f}%")


def cmd_score(a):
    meta, recs = load(a.jsonl, a.work)
    werr = {s: [] for s in ("mfa", "old", "asr")}
    cerr = {s: [] for s in ("mfa", "old")}
    nseg = 0
    for fn in sorted(os.listdir(a.gold)):
        if not fn.endswith(".TextGrid"):
            continue
        key = fn[:-9]; m = meta.get(key)
        if not m:
            print("unknown key", key, file=sys.stderr); continue
        e = recs[m["id"]]; tg = textgrid.read(os.path.join(a.gold, fn)); nseg += 1
        gw = [x for x in textgrid.tier(tg, "words") if x[2].strip()]
        toks = gold_tokens(m)
        if len(gw) != len(toks):
            print(f"{key}: {len(gw)} gold words vs {len(toks)} tokens, skipped", file=sys.stderr); continue
        mfa = {w["tok_idx"]: w for w in (e.get("words_mfa") or [])}
        for (gs, ge, _), t in zip(gw, toks):
            sy = {"mfa": (mfa[t["tok_idx"]]["time_s"], mfa[t["tok_idx"]]["time_e"]) if t["tok_idx"] in mfa else None,
                  "old": span(e.get("words_align"), t["char_s"], t["char_e"]),
                  "asr": span(e.get("words"), t["char_s"], t["char_e"])}
            for s, v in sy.items():
                if v:
                    werr[s] += [abs(v[0] - gs), abs(v[1] - ge)]
        if "chars" not in [n.split(" - ")[-1] for n in tg]:
            continue
        L = letters(e, m); pos = 0; groups = []
        for gs, ge, lab in textgrid.tier(tg, "chars"):
            lab = "".join(c for c in lab.lower() if c.isalpha())
            if not lab:
                continue
            offs = [o for _, o in L[pos:pos + len(lab)]]
            if "".join(c for c, _ in L[pos:pos + len(lab)]) != lab:
                print(f"{key}: chars tier does not match text at '{lab}', chars skipped", file=sys.stderr)
                groups = []; break
            groups.append((gs, ge, offs)); pos += len(lab)
        sysc = {"mfa": flat_chars(e.get("chars_mfa")), "old": flat_chars(e.get("chars_align"))}
        for gs, ge, offs in groups:
            for s, lst in sysc.items():
                v = span(lst, min(offs), max(offs) + 1)
                if v:
                    cerr[s] += [abs(v[0] - gs), abs(v[1] - ge)]
    print(f"segments: {nseg}\nWORD boundaries")
    for s in werr: print(" ", report(s, werr[s]))
    print("GRAPHEME boundaries")
    for s in cerr: print(" ", report(s, cerr[s]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template"); t.add_argument("--jsonl", required=True); t.add_argument("--work", required=True)
    t.add_argument("--keys", required=True); t.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--jsonl", required=True); s.add_argument("--work", required=True)
    s.add_argument("--gold", required=True)
    a = ap.parse_args()
    cmd_template(a) if a.cmd == "template" else cmd_score(a)


if __name__ == "__main__":
    main()
