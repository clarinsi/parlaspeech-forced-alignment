#!/usr/bin/env python3
"""Drop the worst-aligned training segments after round 1 and build corpus_train2.

Usage:
  python3 filter_round.py --work work/HR --analysis work/HR/aligned_r1/alignment_analysis.csv \
      [--drop-ll 0.05] [--drop-dev 0.05]

Reads MFA's alignment_analysis.csv (written by `mfa align`), drops the lowest
`drop-ll` share by speech log-likelihood and the highest `drop-dev` share by
phone-duration deviation, then symlinks the kept segments into corpus_train2/.
Column names are detected loosely, since they differ slightly between MFA versions.
"""
import argparse, csv, glob, os


def pick(cols, *cands):
    for c in cands:
        for col in cols:
            if col.lower() == c:
                return col
    for c in cands:
        for col in cols:
            if c in col.lower():
                return col
    raise SystemExit(f"none of {cands} in columns {cols}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--drop-ll", type=float, default=0.05)
    ap.add_argument("--drop-dev", type=float, default=0.05)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.analysis, encoding="utf-8")))
    cols = list(rows[0].keys())
    cfile = pick(cols, "file", "utterance", "name")
    cll = pick(cols, "speech_log_likelihood", "log_likelihood")
    cdev = pick(cols, "phone_duration_deviation", "duration_deviation")

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    recs = [(os.path.splitext(os.path.basename(r[cfile]))[0], num(r[cll]), num(r[cdev])) for r in rows]
    ok = [r for r in recs if r[1] == r[1] and r[2] == r[2]]
    bad = {r[0] for r in recs if not (r[1] == r[1] and r[2] == r[2])}
    by_ll = sorted(ok, key=lambda r: r[1])
    by_dev = sorted(ok, key=lambda r: -r[2])
    bad |= {r[0] for r in by_ll[:int(len(ok) * a.drop_ll)]}
    bad |= {r[0] for r in by_dev[:int(len(ok) * a.drop_dev)]}

    train = [l.strip() for l in open(f"{a.work}/meta/train.list")]
    aligned = {r[0] for r in recs}
    bad |= set(train) - aligned              # segments MFA failed to align
    keep = [k for k in train if k not in bad]
    src_root = f"{a.work}/corpus_train"
    index = {os.path.splitext(os.path.basename(p))[0]: os.path.dirname(p)
             for p in glob.glob(f"{src_root}/*/*.lab")}
    n = 0
    for k in keep:
        d = index.get(k)
        if not d:
            continue
        spk = os.path.basename(d)
        for p in glob.glob(f"{d}/{k}.*"):
            dst = f"{a.work}/corpus_train2/{spk}/{os.path.basename(p)}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(p), dst)   # audio -> original file, .lab -> corpus_train copy
        n += 1
    open(f"{a.work}/meta/train_round2.list", "w").write("\n".join(keep) + "\n")
    open(f"{a.work}/meta/dropped_round1.list", "w").write("\n".join(sorted(bad)) + "\n")
    print(f"train {len(train)}  aligned {len(aligned & set(train))}  dropped {len(bad)}  kept {n}")


if __name__ == "__main__":
    main()
