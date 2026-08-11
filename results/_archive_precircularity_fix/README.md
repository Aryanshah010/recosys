# INVALID RESULTS — DO NOT CITE

These CSVs are the **pre-circularity-fix** results. They are retained only so the
correction is auditable. **No number in this directory belongs in the thesis.**

Recovered from git commit `618f90e` (2026-07-17), which was the last commit before the
fix. The contents are unchanged since `6249658` (2026-07-10) — the two commits store an
identical blob for every file here.

## Why they are invalid

`src/generate_synthetic_cohort.py` and `engine/hybrid_fusion.py` both imported the same
scoring functions from `src/localization_config.py`. The generator produced ground truth
with a `0.5·genre + 0.3·language + 0.1·popularity` base score — a normalised
genre:language ratio of 0.625:0.375 — and the localized scorer then ranked with
0.60:0.40 using those same functions.

The localized model was recovering the function that had generated its own ground truth.
H1 could not fail, so it was not a hypothesis.

The strongest evidence that this was circular is in the results themselves: the `mixed`
archetype was the only one whose generator zeroed the language coefficient, and it was
the only archetype where the localized model *lost*. Model performance tracked the
generator's coefficients exactly.

## The headline number these files produce

| Model | NDCG@10 |
|---|---|
| CF_ColdStart | 0.0445 |
| CBF | 0.1245 |
| NonLocal_Hybrid | 0.1316 |
| Localized_Hybrid | **0.1991** |

0.1316 → 0.1991 is **+51.3%**. That is the circular figure. The corrected,
non-circular result measured against real human ratings is **+18.2%** (0.1558 → 0.1841),
in `results/rq1_model_performance_real.csv`.

## Two further defects visible in these files

- `rq1_significance_tests.csv` has a bare `p_value` column with no multiple-comparison
  correction across six pairwise tests, and no Wilcoxon or win/tie/loss columns. Six
  uncorrected tests at α = 0.05 carry roughly a 26% family-wise false-positive rate.
- That same file shows `CBF` vs `NonLocal_Hybrid` at **p = 0.343, d = −0.048** — CF was
  contributing essentially nothing to the hybrid, and with no affinity-only ablation
  present there was no way to detect it.

## Structural difference from the current results

These files have no `_real` / `_synthetic` suffix because there was only one track: a
synthetic cohort whose ground truth came from the ranking formula. The current results
carry both a real proxy-cohort track (real human ratings, temporal splits, CF-excluded)
and a decoupled synthetic track, reported side by side.
