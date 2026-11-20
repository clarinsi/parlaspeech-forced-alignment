# ParlaSpeech: high-precision word and grapheme forced alignment — plan and cookbook

Goal: train one forced aligner per language on ParlaSpeech and align whitespace-separated words and their graphemes (letters) to the audio more precisely than the existing ASR alignment. Scripts: `parlaspeech_fa/` (Python standard library only, plus MFA).

---

## Part A — Plan

### A1. Starting point (from the 1k samples per language)

| Field | What it is | Limitation |
|---|---|---|
| `words` | ASR (CTC) word timings, character offsets into `text` | 20 ms grid; peaky/shifted timings. **In HR/RS these are not whitespace tokens**: punctuation splits them (`88,8` → 2 tokens, `2020.g.` → 2), some words are missing, and `[[Upadica: …]]` annotations are skipped. In SI they are whitespace tokens including punctuation. |
| `words_align`, `chars_align` | Earlier forced alignment (HR, RS only) | 10 ms grid, but words run back-to-back (no silences). In the samples it exists for 800 HR and 493 RS segments; SI has none. |
| `filled_pauses`, `silent_pauses` | Detected pauses linked to `words_idx` | Filled pauses are not written in `text`; their timings overlap ASR word timings. |

Because of the HR/RS tokenisation, the pipeline defines its own tokens: **whitespace tokens of `text`**, stored with character offsets, so every language is handled the same way and output can be joined to any existing field through character offsets.

### A2. Tool choice

- **Montreal Forced Aligner (MFA 3.x), trained from scratch for each language.** HMM-based aligners give more accurate boundaries than CTC- or Whisper-based ones. MFA uses 10 ms frames and adapts to each speaker (ParlaSpeech has speaker IDs). With a grapheme dictionary, letter boundaries come out directly and words are separated by real silences.
- **Optional later step:** a neural frame classifier at 10 ms (Charsiu-style, wav2vec2/XLS-R) trained on MFA alignments, or CTC fine-tuning with label priors (Huang et al., ICASSP 2024).
- **Not used as the final aligner:** plain CTC aligners (torchaudio/MMS_FA, NeMo Forced Aligner, WhisperX).

### A3. Evaluation

Not needed to run MFA; needed to claim better precision and to choose settings.

- About 30–50 segments per language with corrected **word** boundaries (~1,000 words, one to two days in Praat per language), and about 10 with corrected **grapheme** boundaries.
- Templates are pre-filled with ASR timings, so any anchoring bias favours the baseline, not MFA.
- Metrics: MAE, median, and share of boundaries within 10/20/50 ms, for MFA, the old alignment and the ASR timings.

### A4. Numerals, acronyms, abbreviations, symbols

- **Training:** segments containing them are left out. This barely affects the acoustic model, because number words use the same letters and contexts as the rest of the text.
- **Final alignment:** all segments are aligned. A token gets letter timings only if it has a dictionary entry; filling in `variants/<lang>.tsv` (all inflected forms of a numeral, spelled-out and read-as-word forms of an acronym) adds them. Tokens without variants are still aligned as a whole word (MFA's "unknown speech" model), so word boundaries around them are kept.
- Leaving them out of the final alignment would systematically drop procedural speech (votes, article numbers, dates, chairpersons) and skew per-speaker or per-role analyses.

Effect on the samples (segments eligible for training): **HR 644/1000, RS 705/1000, SI 806/1000.** Main exclusion reasons: numerals, acronyms, segment length outside 1–30 s.

### A5. Per-language expectations

| Language | Multi-letter units | Main issues | Grapheme precision |
|---|---|---|---|
| HR, SR | dž, lj, nj | Rare false digraphs (*nadživjeti*, *injekcija*) | High |
| CZ | ch | *ě* spreads into the previous consonant; colloquial forms vs. standard transcripts | High |
| SI | none | Spelling doesn't show vowel quality, inserted schwa, l/v → [u] | Medium–high |
| PL | cz, sz, rz, ch, dz, dź, dż; ci/si/zi/ni/dzi before a vowel (the *i* is silent) | Nasal vowels vary; compound numerals inflect part by part | Times inside a unit set by rule |

Word boundaries should be about equally good everywhere. Grapheme boundaries: HR/SR ≈ CZ > SL > PL.

---

## Part B — Cookbook

### B0. Install

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
mfa version            # check that train/align options below match your version (--help)
```

### B1. Files

| Script | Role |
|---|---|
| `langs.py` | Per-language unit rules (digraphs, Polish silent *i*), abbreviation lists, Serbian Cyrillic → Latin |
| `prepare.py` | JSONL → MFA corpora, dictionary, token table |
| `filter_round.py` | Drops the worst-aligned training segments after round 1 |
| `merge.py` | MFA TextGrids → new JSONL fields |
| `evaluate.py` | Gold templates for Praat; boundary scores for MFA / old / ASR |
| `textgrid.py` | Minimal TextGrid reader/writer |
| `run_lang.sh` | All steps for one language |

### B2. Data layout produced

```
work/<LANG>/
  corpus_train/<speaker>/<key>.flac|.lab        clean segments (training, round 1)
  corpus_train2/<speaker>/...                   after filtering (training, round 2)
  corpus_all/<year>/<speaker>/<key>.flac|.lab   all segments (final alignment, per year)
  dict/<lang>.dict                              token<TAB>units
  meta/tokens.jsonl                             per segment: tokens, kinds, char offsets, units
  meta/train.list  exclude.tsv  needs_variants.tsv  stats.json
  models/<lang>_round1.zip  <lang>_round2.zip
  aligned_r1/  aligned_final/<year>/            MFA output (TextGrid + alignment_analysis.csv)
  out/<LANG>.mfa.jsonl                          original records + new fields
```

- `key` = `id` with characters outside `[A-Za-z0-9_-]` replaced by `_`. Audio files are symlinks to the originals.
- `speaker` = `speaker_info.Speaker_ID` (ASCII-folded); segments without one go to a pooled `UNK` speaker.
- **Token kinds:** `word`, `numeral`, `acronym`, `abbrev`, `symbol`, `annotation` (`[[…]]`, `/…/`), `punct` (no letters or digits, e.g. `–`), `fp` (inserted filled pause).
- **`.lab` content:** one entry per spoken token, lowercase letters only (`SDP-a`-type tokens are acronyms; hyphens and apostrophes inside words are dropped). Non-words get keys such as `num_2020_o` (ordinal), `acr_eu`, `abr_dr`. Filled pauses (≥150 ms) are inserted as `filledpause`, before the first token whose ASR start is after the pause midpoint. Consecutive annotation tokens collapse into one `annotationspan` (aligned as unknown speech).
- **Dictionary:** `druge	d r u g e`; `baranju	b a r a nj u`; `filledpause	fp`. In Polish, `się	si ę`, where *i* in the unit `si` gets zero duration.

### B3. Steps

```bash
L=hr; J=32
bash parlaspeech_fa/run_lang.sh $L ParlaSpeech-HR.jsonl /data/ParlaSpeech-HR/audio work/HR $J [variants/hr.tsv]
# or step by step:  STEP=prepare|r1|filter|r2|final|merge bash parlaspeech_fa/run_lang.sh ...
```

1. **prepare** — `prepare.py` builds corpora, dictionary and token table; `mfa validate` reports OOVs and audio problems. Check `meta/stats.json` and `meta/exclude.tsv`.
2. **r1** — `mfa train` on `corpus_train` (a few hundred hours per language is enough; subsample the train list if bigger), then `mfa align` on the same data to get `alignment_analysis.csv`.
3. **filter** — `filter_round.py` drops the lowest 5% by speech log-likelihood and the highest 5% by phone-duration deviation (mostly transcript–audio mismatches), plus segments MFA failed on, and builds `corpus_train2`.
4. **r2** — `mfa train` on `corpus_train2`.
5. **final** — `mfa align` with the round-2 model, one year at a time.
6. **merge** — `merge.py` writes `out/<LANG>.mfa.jsonl`.

**Numeral and acronym variants (between steps 4 and 5, optional but recommended):** `meta/needs_variants.tsv` lists every non-word key by frequency with an example. Write `variants/<lang>.tsv` as `key<TAB>spoken form`, one line per variant:

```
num_2020_o	dvije tisuće dvadesete
num_2020_o	dvije tisuće dvadesetoj
num_2020_o	dvije tisuće dvadesetu
acr_hdz_a	hadezea
acr_hdz_a	ha de ze a
abr_dr	doktor
abr_dr	doktora
```

Then rerun `STEP=prepare` with the variants file (rewrites the dictionary) and continue with `final`. The acoustic model does not need retraining.

### B4. Output fields (added to each record)

| Field | Content |
|---|---|
| `words_mfa` | `[{time_s, time_e, char_s, char_e, tok_idx, kind}]`; non-words also carry `pron` (the variant MFA chose, or `spn`) |
| `units_mfa` | per token: `[{unit, time_s, time_e}]` as MFA timed them (`null` for non-words) |
| `chars_mfa` | per token: `[{text, time_s, time_e, char_s, char_e, derived}]`, like `chars_align`; `derived` = time split by rule (multi-letter unit or silent letter) |
| `filled_pauses_mfa`, `silences_mfa` | `[{time_s, time_e}]`; silences ≥ 10 ms between tokens |
| `mfa_status` | `ok`, `missing` (not aligned) or `token_mismatch` |
| `mfa_score`, `mfa_model` | speech log-likelihood from MFA; model name |

`merge.py` also prints `share_units_at_30ms_floor`: the share of units lasting ≤ 30 ms. A high value means MFA's minimum duration per unit is pushing boundaries; then try a shorter per-phone model topology in MFA's config and check against gold data.

### B5. Evaluation

```bash
# 1) pick 30–50 aligned segments per language (keys, one per line) -> gold.list
python3 parlaspeech_fa/evaluate.py template --jsonl work/HR/out/HR.mfa.jsonl --work work/HR --keys gold.list --out gold_todo/HR
# 2) annotators correct tiers in Praat and save to gold_done/HR (long text format)
python3 parlaspeech_fa/evaluate.py score --jsonl work/HR/out/HR.mfa.jsonl --work work/HR --gold gold_done/HR
```

**Annotation rules:**

- Tier `words`: one interval per token; don't change labels.
- Tier `chars`: one interval per letter. A multi-letter sound may be a single interval labelled with all its letters (`lj`, `sz`). A silent letter is merged into its neighbour's interval (`ni` in *nie*).
- Filled pauses and silences stay unlabelled.

### B6. Tuning checklist (decide with the gold set)

- Digraphs as one unit vs. two letters (`langs.py`, `MULTI`).
- `--fp-min-dur` and whether to insert filled pauses at all (`--no-fp`).
- Round-1 filter shares (`--drop-ll`, `--drop-dev`).
- MFA beam/retry-beam, per-phone topology, and a fine-tuning boundary option if your MFA version has one.
- For Czech: add common colloquial forms as extra dictionary pronunciations. For Polish: optionally a rule-based G2P if the cluster units underperform.

### B7. Status of the scripts

Tested on the 1k HR/RS/SI samples up to MFA itself: corpus/dictionary generation, merging, gold templates, scoring and filtering, using synthetic MFA-style output. The `mfa` commands themselves have not been run here; check option names against your installed version.
