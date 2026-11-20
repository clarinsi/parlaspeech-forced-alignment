#!/usr/bin/env python3
"""Turn MFA TextGrids back into ParlaSpeech JSONL fields.

Usage:
  python3 merge.py --jsonl HR.jsonl --work work/HR --aligned work/HR/aligned_final \
      --out work/HR/out/HR.mfa.jsonl --model hr_round2 [--analysis 'work/HR/aligned_final/**/alignment_analysis.csv']

Adds to every record:
  words_mfa          [{time_s, time_e, char_s, char_e, tok_idx, kind}]  one per spoken whitespace token
  units_mfa          per word: [{unit, time_s, time_e}]  (what MFA actually timed)
  chars_mfa          per word: [{text, time_s, time_e, char_s, char_e, derived}]  (like chars_align)
  filled_pauses_mfa  [{time_s, time_e}]
  silences_mfa       [{time_s, time_e}]  gaps between words (edges excluded)
  mfa_status         ok | missing | token_mismatch
  mfa_score, mfa_model
`derived` = letter time was set by rule (part of a multi-letter unit, or silent).
"""
import argparse, collections, csv, glob, json, os, re
import textgrid

SIL = {"", "sil", "sp", "<eps>", "silence"}
POS = re.compile(r"_[BIES]$")


def r3(x):
    return round(x, 3)


def load_scores(patterns):
    sc = {}
    for pat in patterns or []:
        for p in glob.glob(pat, recursive=True):
            for row in csv.DictReader(open(p, encoding="utf-8")):
                f = row.get("file") or row.get("utterance") or row.get("name") or ""
                ll = next((row[c] for c in row if "speech_log_likelihood" in c.lower()), None)
                ll = ll or next((row[c] for c in row if "log_likelihood" in c.lower()), None)
                try:
                    sc[os.path.splitext(os.path.basename(f))[0]] = float(ll)
                except (TypeError, ValueError):
                    pass
    return sc


def chars_from_units(tok, unit_times, text):
    per_off = collections.OrderedDict()
    for (u, idx, sil), (us, ue) in zip(tok["units"], unit_times):
        voiced = [k for k, s in enumerate(sil) if not s]
        step = (ue - us) / max(len(voiced), 1)
        derived = len(idx) > 1
        vi = 0
        for k, ni in enumerate(idx):
            if sil[k]:
                cs = ce = ue
            else:
                cs, ce = us + vi * step, us + (vi + 1) * step
                vi += 1
            off = tok["norm_offs"][ni]
            if off in per_off:
                p = per_off[off]
                p["time_s"], p["time_e"] = min(p["time_s"], cs), max(p["time_e"], ce)
                p["derived"] = True
            else:
                per_off[off] = {"text": text[off], "time_s": cs, "time_e": ce,
                                "char_s": off, "char_e": off + 1, "derived": derived or sil[k]}
    out = list(per_off.values())
    for c in out:
        c["time_s"], c["time_e"] = r3(c["time_s"]), r3(c["time_e"])
    return out


def align_record(e, meta, tg_path, stats):
    toks = [t for t in meta["tokens"] if t["kind"] != "punct" and not t.get("merged")]
    tiers = textgrid.read(tg_path)
    words = [w for w in textgrid.tier(tiers, "words") if w[2].strip().lower() not in SIL]
    phones = [(s, x, POS.sub("", l)) for s, x, l in textgrid.tier(tiers, "phones") if l.strip().lower() not in SIL]
    if len(words) != len(toks):
        stats["token_mismatch"] += 1
        return {"mfa_status": "token_mismatch"}
    res = {"words_mfa": [], "units_mfa": [], "chars_mfa": [], "filled_pauses_mfa": [], "silences_mfa": []}
    for (ws, we, lab), t in zip(words, toks):
        if t["kind"] == "fp":
            res["filled_pauses_mfa"].append({"time_s": r3(ws), "time_e": r3(we)})
            continue
        w = {"time_s": r3(ws), "time_e": r3(we), "char_s": t["char_s"], "char_e": t["char_e"],
             "tok_idx": t["tok_idx"], "kind": t["kind"]}
        ph = [p for p in phones if p[0] >= ws - 1e-4 and p[1] <= we + 1e-4]
        if t["kind"] == "word":
            expected = [u for u, _, _ in t["units"]]
            if [p[2] for p in ph] == expected:
                res["units_mfa"].append([{"unit": p[2], "time_s": r3(p[0]), "time_e": r3(p[1])} for p in ph])
                res["chars_mfa"].append(chars_from_units(t, [(p[0], p[1]) for p in ph], e["text"]))
                for p in ph:
                    stats["units"] += 1
                    stats["units_at_floor"] += (p[1] - p[0]) <= 0.0305
            else:
                stats["unit_mismatch"] += 1
                res["units_mfa"].append(None); res["chars_mfa"].append(None)
        else:
            w["pron"] = " ".join(p[2] for p in ph)      # which variant MFA chose
            res["units_mfa"].append(None); res["chars_mfa"].append(None)
        res["words_mfa"].append(w)
    for (a, b), (c, d) in zip([(x[0], x[1]) for x in words[:-1]], [(x[0], x[1]) for x in words[1:]]):
        if c - b >= 0.01:
            res["silences_mfa"].append({"time_s": r3(b), "time_e": r3(c)})
    res["mfa_status"] = "ok"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--aligned", required=True, nargs="+", help="MFA output dir(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--analysis", nargs="*", help="glob(s) of alignment_analysis.csv")
    a = ap.parse_args()

    tgs = {}
    for d in a.aligned:
        for p in glob.glob(os.path.join(d, "**", "*.TextGrid"), recursive=True):
            tgs[os.path.splitext(os.path.basename(p))[0]] = p
    scores = load_scores(a.analysis)
    meta_by_id = {}
    for line in open(f"{a.work}/meta/tokens.jsonl", encoding="utf-8"):
        m = json.loads(line); meta_by_id[m["id"]] = m
    stats = collections.Counter()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for line in open(a.jsonl, encoding="utf-8"):
            e = json.loads(line)
            m = meta_by_id.get(e["id"])
            add = {"mfa_status": "missing"}
            if m and m["key"] in tgs:
                add = align_record(e, m, tgs[m["key"]], stats)
                add["mfa_score"] = scores.get(m["key"])
            if add["mfa_status"] != "ok":
                for f in ("words_mfa", "units_mfa", "chars_mfa", "filled_pauses_mfa", "silences_mfa"):
                    add.setdefault(f, None)
            add["mfa_model"] = a.model
            stats["seg_" + add["mfa_status"]] += 1
            e.update(add)
            fo.write(json.dumps(e, ensure_ascii=False) + "\n")
    if stats["units"]:
        stats["share_units_at_30ms_floor"] = round(stats["units_at_floor"] / stats["units"], 4)
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
