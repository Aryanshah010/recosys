# recosys — Hybrid Movie Recommender for Nepali Male IT Students

Research codebase for an undergraduate thesis (ST6000CEM, Coventry University /
Softwarica). It builds and empirically evaluates a hybrid recommender that adds
a **cohort-affinity** term (genre + language preference weighting) to a standard
CF + CBF hybrid, and measures the effect on both ranking accuracy (RQ1) and
exposure diversity (RQ2) for an under-represented user segment.

## What "localization" means here

It means **audience** localization, not content localization.

Nepali male IT undergraduates overwhelmingly consume Hollywood, Bollywood, anime
and Korean cinema rather than Nepali-language films, and the public catalogues
used here contain effectively no Nepali cinema — exactly **1 Nepali title out of
17,112** survives cleaning (*Manakamana*, 2013, a Documentary with 12 votes).

The localized signal is therefore the cohort's *multilingual consumption
profile*, which differs structurally from the English-dominant default that
global recommenders assume. The catalogue's near-total absence of Nepali content
is reported as a finding — a direct measurement of the structural
under-representation that motivates the study — not hidden as a limitation.

Supporting measurement: of 159,998 MovieLens users with ≥20 liked ratings,
**90.06% are English-dominant**; Japanese-inclined users are 0.30%,
Hindi-inclined 0.14%, Korean-inclined 0.09%. The target cohort is modelled as
~65% non-Hollywood, so it corresponds to roughly **0.5%** of the field's
standard benchmark population.

## Two evaluation tracks

| Track | Ground truth | Role |
|---|---|---|
| **real** | Real MovieLens users whose observed consumption matches the cohort profile, held out of CF training | Carries the empirical claim. Non-circular. |
| **synthetic** | Controlled cohort, archetype prevalence known by construction | Enables per-segment analysis the benchmark is too sparse to support. A simulation, and labelled as one. |

Both run through identical evaluation code. Reporting them side by side is the
point: if the localized model wins only in simulation, that is a finding about
the simulation.

## Data

Not committed (see `.gitignore`). Download and place as follows:

```
data/raw/ml-32m/          # MovieLens 32M — https://grouplens.org/datasets/movielens/32m/
    ratings.csv movies.csv links.csv
data/raw/tmdb/
    movies_metadata.csv   # TMDB metadata, joined via tmdbId
```

Both are publicly available research datasets. No API access or scraping is
required; everything runs desk-based from these two files.

## Running

```bash
uv sync
python main.py
```

Pipeline steps (order is load-bearing):

| Step | Script | Produces |
|---|---|---|
| 1 | `src/clean_data.py` | `movies_final.csv`, `ratings_final.csv` (timestamps retained) |
| 2 | `src/build_cbf_matrix.py` | TF-IDF over genres **and synopses** |
| 3 | `src/build_real_cohort.py` | real cohort, CF exclusion list, affinity tables |
| 4 | `src/generate_synthetic_cohort.py` | behaviour-seeded synthetic cohort |
| 5 | `engine/collaborative_filtering.py` | SVD, **excluding the evaluation cohort** |
| 6 | `evaluation/evaluation_metrics.py` | all RQ1/RQ2 tables, both tracks |

Step 3 must precede steps 4 and 5:

- Step 4 samples from per-archetype item affinity measured on real users, which
  is what keeps the synthetic generator independent of the scorer.
- Step 5 must exclude the evaluation cohort, or their holdout items leak into
  the factorisation.

Then:

```bash
python -m evaluation.sensitivity_analysis   # λ sweep + archetype-mix robustness
python -m evaluation.make_figures           # all 12 figures (descriptive + RQ1/RQ2)
uvicorn api.main:app --reload               # demo UI
```

## Models compared

| Model | Description |
|---|---|
| `Popularity` | Bayesian weighted-rating baseline |
| `CF_ColdStart` | SVD with averaged item factors (cohort is unseen by design) |
| `CBF` | TF-IDF content similarity |
| `Affinity_Only` | **Ablation** — cohort affinity alone, no CF or CBF |
| `NonLocal_Hybrid` | CF 0.625 / CBF 0.375 |
| `Localized_Hybrid` | CF 0.50 / CBF 0.30 / affinity 0.20 |

Both hybrids hold CF:CBF at 5:3, so the localized preset is exactly
`0.8 × standard + 0.2 × affinity`. Any difference between them is attributable
to the affinity term rather than to a reweighting of shared components.

`Affinity_Only` is essential rather than decorative: CF contributes little under
cold start, so without this ablation it is impossible to tell whether the hybrid
adds anything over the affinity term on its own.

## Metrics

- **RQ1** — Precision@10, Recall@10, **HR@10**, **MRR@10**, NDCG@10.
  Precision@10 is structurally capped at `|holdout|/10`; the attainable ceiling
  is reported alongside it, and HR/MRR are not subject to it.
- **RQ2** — language diversity, genre diversity, Filter Bubble Score (fraction
  of top-10 matching *both* preferred language and genre) and its complement
  Novelty@10.
- **Statistics** — paired t-test *and* Wilcoxon signed-rank (NDCG is heavily
  zero-inflated), Holm–Bonferroni correction across the comparison family,
  Cohen's d, 95% CIs, and explicit win/tie/loss counts.

## Repository layout

```
main.py                 pipeline orchestrator (six steps, artifact-checked)
seed_db.py              loads catalogue + cohort into SQLite for the demo

src/    cohort_spec.py            target-cohort specification (single source of truth)
        clean_data.py             ETL + MovieLens/TMDB fusion
        build_cbf_matrix.py       TF-IDF content features
        build_real_cohort.py      real proxy cohort + derived affinity tables
        generate_synthetic_cohort.py
        localization_config.py    cohort-affinity scoring configuration
        frames.py                 typed pandas aggregation helpers
        mappings.py               canonical genre/language vocabularies
engine/     collaborative_filtering.py, content_filter.py, hybrid_fusion.py
evaluation/ evaluation_metrics.py, sensitivity_analysis.py, make_figures.py
api/        FastAPI demo (dashboard, profile, metrics, bias pages)
tests/      methodological invariants
notebooks/  exploratory data analysis and processed-data validation
results/    generated tables and figures
            _archive_precircularity_fix/  INVALID pre-fix results, retained only
            so the correction is auditable — see its README, do not cite
```

## Quality gates

```bash
ruff check .     # lint          -> clean
ruff format .    # formatting    -> 25 files
pyright          # type checking -> clean
pytest           # 9 methodological invariants
```

`tests/` guards the properties the thesis argues for rather than ordinary
behaviour: that the synthetic generator never re-imports the scorer's functions
(which would make RQ1 circular), that no evaluation-cohort user reaches the SVD
trainset, that splits are genuinely temporal, and that preferences are inferable
from the train split alone.

## Known limitations

- The real cohort are MovieLens users **worldwide** with matching consumption
  profiles, not Nepali users. Real behavioural ground truth is bought at the
  cost of national specificity.
- Archetype prevalences (35/20/20/15/10) are a design assumption: no survey was
  possible within the desk-based scope. They are stress-tested in
  `evaluation/sensitivity_analysis.py` rather than presented as measured.
- Demographic weighting operates at **cohort level** — the archetype mix and
  language set were chosen because the users are Nepali/male/IT/19–26. Age,
  gender and education are recorded but are not per-user model features.

- The catalogue holds one Nepali title in 17,112, so no claim about
  Nepali-content surfacing is testable here. Supplementing it was ruled out: no
  cohort user prefers Nepali, no rating exists for any Nepali film so a title
  can never be a held-out positive, and language affinity is estimated as
  consumption lift, which pins Nepali at the floor at any catalogue size.
- All results are offline. Ranking gains are well known not to transfer
  reliably to user-perceived quality without online validation.
