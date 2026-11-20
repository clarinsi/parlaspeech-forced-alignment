#!/usr/bin/env python3
"""Build MFA corpora, dictionary and token metadata from a ParlaSpeech JSONL.

Usage:
  python3 prepare.py --jsonl HR.jsonl --lang hr --audio-root /data/parlaspeech/HR/audio \
      --out work/HR [--variants variants/hr.tsv] [--shard-by-year] [--no-fp]

Creates:
  out/corpus_train/<spk>/<key>.{flac,lab}   clean segments only (training)
  out/corpus_all[/<year>]/<spk>/<key>.{flac,lab}   all segments (final alignment)
  out/dict/<lang>.dict                     word<TAB>units
  out/meta/tokens.jsonl                    per-segment token table (needed by merge.py)
  out/meta/train.list, exclude.tsv, needs_variants.tsv, stats.json
Only the standard library is used.
"""
import argparse, collections, json, os, re, sys, unicodedata
from langs import ABBREV, normalize_word, split_units, safe_key

FP_TOKEN, FP_UNIT = "filledpause", "fp"
ANN_TOKEN = "annotationspan"           # not in dict -> MFA aligns it to spn
SYMBOLS = set("%§&@+=€$°/#*")
UPPER2 = re.compile(r"^[^\W\d_]{2,}")
ANN_RE = re.compile(r"\[\[.*?\]\]|/[^/\s]{3,}/|…/[^/]+/…")


def key_of(e):
    return re.sub(r"[^A-Za-z0-9_-]", "_", e["id"])


def speaker_of(e):
    si = e.get("speaker_info") or {}
    spk = si.get("Speaker_ID") or "UNK"
    return safe_key("", spk) or "UNK"


def tokens_of(e, lang, ann_spans):
    """Whitespace tokens of `text` with kind and key."""
    text, toks = e["text"], []
    for m in re.finditer(r"\S+", text):
        tok, s, t = m.group(), m.start(), m.end()
        a, b = 0, len(tok)
        while a < b and not tok[a].isalnum():
            a += 1
        while b > a and not tok[b - 1].isalnum():
            b -= 1
        core, cs, ce = tok[a:b], s + a, s + b
        rec = {"tok_idx": len(toks), "char_s": cs, "char_e": ce, "orig": tok}
        if any(x < t and s < y for x, y in ann_spans):
            rec.update(kind="annotation", key=ANN_TOKEN)
        elif not core:
            rec.update(kind="punct", key=None)
        elif any(ch.isdigit() for ch in core):
            ordinal = tok[b:b + 1] == "."
            rec.update(kind="numeral", key=safe_key("num_", core) + ("_o" if ordinal else ""))
        elif any(ch in SYMBOLS for ch in tok):
            rec.update(kind="symbol", key=safe_key("sym_", tok))
        elif UPPER2.match(core) and core[:2].isupper():
            rec.update(kind="acronym", key=safe_key("acr_", core))
        elif tok[b:b + 1] == "." and core.lower() in ABBREV.get(lang, ()):
            rec.update(kind="abbrev", key=safe_key("abr_", core))
        else:
            chars, offs = normalize_word(core, cs, lang)
            units = split_units(chars, lang)
            rec.update(kind="word", key="".join(chars), norm_offs=offs,
                       units=[[u, idx, sil] for u, idx, sil in units])
        toks.append(rec)
    return toks


def insert_fps(e, toks, min_dur):
    """Place filled pauses before the first token whose ASR start >= pause midpoint.
    ASR timings for a whitespace token = timings of ASR words overlapping its span."""
    fps = [f for f in (e.get("filled_pauses") or []) if f["time_e"] - f["time_s"] >= min_dur]
    if not fps:
        return toks
    starts = []
    for t in toks:
        ov = [w["time_s"] for w in e["words"] if w["char_s"] < t["char_e"] and t["char_s"] < w["char_e"]]
        starts.append(min(ov) if ov else None)
    out, fi = [], 0
    fps = sorted(fps, key=lambda f: f["time_s"])
    for t, st in zip(toks, starts):
        while fi < len(fps) and st is not None and st >= (fps[fi]["time_s"] + fps[fi]["time_e"]) / 2:
            out.append({"kind": "fp", "key": FP_TOKEN, "asr_s": fps[fi]["time_s"], "asr_e": fps[fi]["time_e"]})
            fi += 1
        out.append(t)
    for f in fps[fi:]:
        out.append({"kind": "fp", "key": FP_TOKEN, "asr_s": f["time_s"], "asr_e": f["time_e"]})
    return out


def link(src, dst):
    if os.path.lexists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def write_lab(path, words):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(words) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", help="TSV: key<TAB>spoken form (one line per variant)")
    ap.add_argument("--shard-by-year", action="store_true")
    ap.add_argument("--no-fp", action="store_true", help="do not insert filled pauses")
    ap.add_argument("--fp-min-dur", type=float, default=0.15)
    ap.add_argument("--min-len", type=float, default=1.0)
    ap.add_argument("--max-len", type=float, default=30.0)
    a = ap.parse_args()
    lang = a.lang.lower()
    os.makedirs(f"{a.out}/meta", exist_ok=True)
    os.makedirs(f"{a.out}/dict", exist_ok=True)

    lex = {}                                   # key -> set of unit strings
    lex[FP_TOKEN] = {FP_UNIT}
    need = collections.Counter(); need_ex = {}
    st = collections.Counter(); keys = set()
    ftok = open(f"{a.out}/meta/tokens.jsonl", "w", encoding="utf-8")
    ftrain = open(f"{a.out}/meta/train.list", "w")
    fexcl = open(f"{a.out}/meta/exclude.tsv", "w")

    for line in open(a.jsonl, encoding="utf-8"):
        e = json.loads(line)
        key, spk = key_of(e), speaker_of(e)
        if key in keys:
            sys.exit(f"duplicate key {key}")
        keys.add(key)
        ann = [(m.start(), m.end()) for m in ANN_RE.finditer(e["text"])]
        toks = tokens_of(e, lang, ann)
        if not a.no_fp:
            toks = insert_fps(e, toks, a.fp_min_dur)
        spoken = [t for t in toks if t["kind"] != "punct"]
        # collapse consecutive annotation tokens into one
        lab = []
        for t in spoken:
            if t["kind"] == "annotation" and lab and lab[-1] == ANN_TOKEN:
                t["merged"] = True
                continue
            lab.append(t["key"])
        for t in spoken:
            st[t["kind"]] += 1
            if t["kind"] == "word":
                lex.setdefault(t["key"], set()).add(" ".join(u for u, _, _ in t["units"]))
            elif t["kind"] in ("numeral", "acronym", "abbrev", "symbol"):
                need[t["key"]] += 1
                need_ex.setdefault(t["key"], t["orig"])
        kinds = {t["kind"] for t in spoken}
        reasons = sorted(kinds - {"word", "fp"})
        dur = e.get("audio_length") or 0
        if not (a.min_len <= dur <= a.max_len):
            reasons.append("length")
        if not lab:
            reasons.append("empty")
        shard = (e.get("speaker_info") or {}).get("Date", "0000")[:4] if a.shard_by_year else ""
        src = os.path.abspath(os.path.join(a.audio_root, e["audio"]))
        ext = os.path.splitext(e["audio"])[1] or ".flac"
        dirs = [os.path.join(a.out, "corpus_all", shard, spk)]
        if not reasons:
            dirs.append(os.path.join(a.out, "corpus_train", spk))
            ftrain.write(key + "\n"); st["seg_train"] += 1
        else:
            fexcl.write(f"{key}\t{','.join(reasons)}\n"); st["seg_excluded"] += 1
        if lab:
            for d in dirs:
                link(src, os.path.join(d, key + ext))
                write_lab(os.path.join(d, key + ".lab"), lab)
        st["seg_total"] += 1
        ftok.write(json.dumps({"key": key, "id": e["id"], "speaker": spk, "shard": shard,
                               "tokens": toks, "lab": lab}, ensure_ascii=False) + "\n")

    # variants for numerals / acronyms / abbreviations / symbols
    nvar = 0
    if a.variants:
        for line in open(a.variants, encoding="utf-8"):
            if not line.strip() or line.startswith("#"):
                continue
            k, spoken = line.rstrip("\n").split("\t", 1)
            units = []
            for w in spoken.split():
                chars, _ = normalize_word(w, 0, lang)
                units += [u for u, _, _ in split_units(chars, lang)]
            if units:
                lex.setdefault(k, set()).add(" ".join(units)); nvar += 1
    with open(f"{a.out}/dict/{lang}.dict", "w", encoding="utf-8") as f:
        for k in sorted(lex):
            for pron in sorted(lex[k]):
                f.write(f"{k}\t{pron}\n")
    with open(f"{a.out}/meta/needs_variants.tsv", "w", encoding="utf-8") as f:
        f.write("key\tcount\texample\thas_variant\n")
        for k, c in need.most_common():
            f.write(f"{k}\t{c}\t{need_ex[k]}\t{int(k in lex)}\n")
    st["dict_entries"] = sum(len(v) for v in lex.values()); st["variant_lines"] = nvar
    json.dump(dict(st), open(f"{a.out}/meta/stats.json", "w"), indent=1)
    print(json.dumps(dict(st), indent=1))


if __name__ == "__main__":
    main()
