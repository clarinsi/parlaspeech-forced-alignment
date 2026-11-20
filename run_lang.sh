#!/usr/bin/env bash
# Full pipeline for one language.
# Usage: bash run_lang.sh <lang> <jsonl> <audio_root> <workdir> [jobs] [variants.tsv]
#   e.g. bash run_lang.sh hr /data/ParlaSpeech-HR.jsonl /data/ParlaSpeech-HR/audio work/HR 32 variants/hr.tsv
# Requires: conda env with MFA 3.x  (conda create -n mfa -c conda-forge montreal-forced-aligner)
# Check option names against `mfa train --help` / `mfa align --help` of your installed version.
set -euo pipefail
L=$1; JSONL=$2; AUDIO=$3; W=$4; J=${5:-16}; VAR=${6:-}
HERE=$(cd "$(dirname "$0")" && pwd)
STEP=${STEP:-all}     # run a single step with STEP=prepare|r1|filter|r2|final|merge

run() { [[ $STEP == all || $STEP == "$1" ]]; }

if run prepare; then
  python3 "$HERE/prepare.py" --jsonl "$JSONL" --lang "$L" --audio-root "$AUDIO" --out "$W" \
      --shard-by-year ${VAR:+--variants "$VAR"}
  mfa validate "$W/corpus_train" "$W/dict/$L.dict" -j "$J" --clean | tee "$W/meta/validate.log"
fi

if run r1; then      # round 1: train on clean segments, align them to get scores
  mfa train "$W/corpus_train" "$W/dict/$L.dict" "$W/models/${L}_round1.zip" -j "$J" --clean
  mfa align "$W/corpus_train" "$W/dict/$L.dict" "$W/models/${L}_round1.zip" "$W/aligned_r1" -j "$J" --clean
fi

if run filter; then  # drop worst 5% by likelihood and 5% by duration deviation
  python3 "$HERE/filter_round.py" --work "$W" --analysis "$W/aligned_r1/alignment_analysis.csv"
fi

if run r2; then      # round 2: retrain on cleaned data
  mfa train "$W/corpus_train2" "$W/dict/$L.dict" "$W/models/${L}_round2.zip" -j "$J" --clean
fi

if run final; then   # align everything, one shard (year) at a time
  for S in "$W"/corpus_all/*/; do
    Y=$(basename "$S")
    mfa align "$S" "$W/dict/$L.dict" "$W/models/${L}_round2.zip" "$W/aligned_final/$Y" -j "$J" --clean
  done
fi

if run merge; then
  python3 "$HERE/merge.py" --jsonl "$JSONL" --work "$W" --aligned "$W/aligned_final" \
      --analysis "$W/aligned_final/*/alignment_analysis.csv" --model "${L}_round2" \
      --out "$W/out/$(basename "${JSONL%.jsonl}").mfa.jsonl"
fi
