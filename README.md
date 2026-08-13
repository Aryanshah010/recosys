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
17,070** survives cleaning (*Manakamana*, 2013, a Documentary with 12 votes).

The localized signal is therefore the cohort's *multilingual consumption
profile*, which differs structurally from the English-dominant default that
global recommenders assume. The catalogue's near-total absence of Nepali content
is reported as a finding — a direct measurement of the structural
under-representation that motivates the study — not hidden as a limitation.

Supporting measurement: of 143,319 MovieLens users with ≥20 liked ratings in
their training period, **90.52% are English-dominant**; Japanese-inclined users
are 0.29%, Hindi-inclined 0.13%, Korean-inclined 0.08%. The three
minority-language archetypes together are **0.49%** of the field's standard
benchmark population — that is the slice this cohort is drawn from.

## Two evaluation tracks

| Track | Ground truth | Role |
|---|---|---|
| **real** | Real MovieLens users whose observed consumption matches the cohort profile, held out of CF training | Carries the empirical claim. Non-circular. |
| **synthetic** | Controlled cohort, archetype prevalence known by construction | Enables per-segment analysis the benchmark is too sparse to support. A simulation, and labelled as one. |

Both run through identical evaluation code, but their ground truth is built
under different protocols: the real cohort uses a per-user **temporal** split,
the synthetic cohort a **random** one. A random split is the easier problem, so
absolute values are not comparable across tracks — compare the relative gains.

Reporting them side by side is the point: if the localized model wins only in
simulation, that is a finding about the simulation. It does not. On the real
track the NDCG@10 gain is **+21.0% (95% CI +15.1% to +27.4%)**; on the synthetic
track it is **+6.8% (95% CI +1.6% to +12.0%)** — same direction, and the
empirical track is the stronger of the two.

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

### Movies only

The raw TMDB dump labels a number of TV series and miniseries as movies, and
MovieLens links to them, so a plain join admits titles like *Band of Brothers*
and *Planet Earth* into what is meant to be a film catalogue. Runtime alone
cannot separate them — *Shoah* runs 566 minutes and is a film — so
`src/television_filter.py` defers to TMDB's own classification: an id belonging
to a TV record 404s against the `/3/movie/{id}` endpoint. Its output,
`data/processed/television_exclusions.json`, **is committed**, so the catalogue
reproduces without network access; regenerating it needs `TMDB_API_KEY`.

Made-for-TV *films* are kept — they are films — but TMDB's `TV Movie` genre is
no longer mapped into the taxonomy, so nothing in the catalogue or the UI is
labelled "TV". A small reviewed keep-list in the same module reinstates titles
whose Kaggle `tmdbId` is stale. `tests/test_methodology_invariants.py` pins all
of this.

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
| 7 | `evaluation/sensitivity_analysis.py` | λ sweep + archetype-mix robustness, both tracks |
| 8 | `evaluation/make_figures.py` | all 13 figures |

Step 3 must precede steps 4 and 5:

- Step 4 samples from per-archetype item affinity measured on real users, which
  is what keeps the synthetic generator independent of the scorer.
- Step 5 must exclude the evaluation cohort, or their holdout items leak into
  the factorisation.

Every step verifies its expected artifacts and halts if any are missing, so a
completed run is a complete set of thesis inputs.

Then, for the demo:

```bash
python seed_db.py               # load the cohort into SQLite (auto-runs on first boot)
uvicorn api.main:app --reload
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
  of top-10 matching *both* preferred language and genre) and its exact
  complement, **Preference_Escape@10**. That metric is deliberately *not* called
  novelty: novelty in the literature means self-information over item
  popularity, which is not what this measures.
- **Statistics** — paired t-test *and* Wilcoxon signed-rank (NDCG is heavily
  zero-inflated), Holm–Bonferroni correction across the comparison family,
  paired effect sizes (`Cohens_dz`, standardised on the SD of the paired
  differences — not an independent-samples *d*), and explicit win/tie/loss
  counts.
- **The headline effect** — `rq1_headline_effect_{track}.csv` reports the
  localized-vs-standard gain with a paired percentile bootstrap 95% interval
  (5,000 resamples). The per-model intervals in
  `rq1_confidence_intervals_*.csv` are *marginal*: they overlap and cannot be
  read as a test of the difference between two models scored on the same users.

## Repository layout

```
main.py                 pipeline orchestrator (eight steps, artifact-checked)
seed_db.py              loads the cohort into SQLite for the demo

src/    cohort_spec.py            target-cohort specification (single source of truth)
        clean_data.py             ETL + MovieLens/TMDB fusion
        television_filter.py      TMDB sweep that keeps the catalogue movies-only
        build_cbf_matrix.py       TF-IDF content features
        build_real_cohort.py      real proxy cohort + derived affinity tables
        generate_synthetic_cohort.py
        localization_config.py    cohort-affinity scoring configuration
        frames.py                 typed pandas aggregation helpers
        mappings.py               canonical genre/language vocabularies
engine/     collaborative_filtering.py, content_filter.py, hybrid_fusion.py
evaluation/ evaluation_metrics.py, sensitivity_analysis.py, make_figures.py
api/        FastAPI demo (see below)
tests/      methodological invariants
notebooks/  exploratory data analysis and processed-data validation
results/    generated tables and figures
```

## Quality gates

```bash
ruff check .          # lint          -> clean
ruff format --check . # formatting    -> clean
pyright               # type checking -> clean
pytest                # 13 methodological invariants
```

`tests/` guards the properties the thesis argues for rather than ordinary
behaviour: that the synthetic generator never re-imports the scorer's functions
(which would make RQ1 circular), that no evaluation-cohort user reaches the SVD
trainset, that splits are genuinely temporal, that preferences are inferable
from the train split alone, that no module hand-copies a weight
`localization_config` owns, and that **the model the demo serves is numerically
identical to the model the harness evaluates**.

## Demo

```bash
uvicorn api.main:app --reload
```

| Page | Shows |
|---|---|
| `/` | cohort user picker, archetype shares read from `cohort_spec` |
| `/dashboard/<id>` | all six models side by side, measured per-stage timings, live diversity metrics, before/after diff |
| `/profile/<id>` | inferred preferences and persisted session history |
| `/movie/<id>` | detail page; rating a title re-scores every model |
| `/cohort` | representation gap, sampling audit, prior-vs-derived affinity tables |
| `/metrics` | RQ1 tables with a real/synthetic track toggle |
| `/bias` | RQ2 tables, λ trade-off sweep, embedded figures |

The demo scores through `engine/hybrid_fusion.py`, and a test pins the served
models to that engine. It does **not** share a code path with the evaluation
harness, and the two are not expected to produce identical rankings: the harness
scores a sampled candidate set of ~1,000 items and min-max normalises within it,
while the demo ranks the full catalogue. What they do share — enforced by
`test_consumers_import_constants_rather_than_copying` — are the hybrid weights
and affinity tables, both imported from `localization_config` rather than
copied. So the demo illustrates the evaluated models; `results/` remains the
authority on their measured performance.

SQLite stores *interaction state only* (ratings, sessions, logs); the catalogue
is served from the model artifacts, so there is one source of truth.

Poster images are resolved once from TMDB when `TMDB_API_KEY` is set in `.env`
and cached to `data/processed/poster_cache.json`; after the first run the demo
needs no network, and titles without a poster fall back to a text tile.

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

- The catalogue holds one Nepali title in 17,070, so no claim about
  Nepali-content surfacing is testable here. Supplementing it was ruled out: no
  cohort user prefers Nepali, no rating exists for any Nepali film so a title
  can never be a held-out positive, and language affinity is estimated as
  consumption lift, which pins Nepali at the floor at any catalogue size.
- Ranking is measured against a **sampled** candidate set (~1,000 negatives per
  user), and the set grows with a user's holdout size, so the base rate is not
  constant across users. Sampled top-k metrics are known not to be consistent
  estimators of full-ranking metrics (Krichene & Rendle, KDD 2020). The
  popularity-stratified sampling removes the popularity confound but not this.
- `Affinity_Only` is a reference point, not a competitive baseline: it scores
  exactly the predicate `Filter_Bubble_Score` measures, so its bubble score is
  1.000 by construction rather than by measurement.
- All results are offline. Ranking gains are well known not to transfer
  reliably to user-perceived quality without online validation.

