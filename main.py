import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
ENGINE_DIR = PROJECT_ROOT / "engine"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

PIPELINE_STEPS = [
    (
        SRC_DIR / "clean_data.py",
        "Step 1/8: Data Cleaning & ETL (MovieLens + TMDb Fusion)",
        ["movies_final.csv", "ratings_final.csv"],
    ),
    (
        SRC_DIR / "build_cbf_matrix.py",
        "Step 2/8: Building CBF Matrix (TF-IDF over genres + synopses)",
        ["cbf_matrix.pkl", "cbf_metadata.pkl"],
    ),
    (
        SRC_DIR / "build_real_cohort.py",
        "Step 3/8: Selecting Real Proxy Cohort + Deriving Affinity Tables",
        [
            "real_cohort_users.csv",
            "real_cohort_ratings.csv",
            "cf_excluded_user_ids.json",
            "archetype_affinity.pkl",
            "cohort_affinity_tables.json",
            "benchmark_population_profile.csv",
            "derived_affinity_tables.csv",
            "real_cohort_sampling_audit.csv",
        ],
    ),
    (
        SRC_DIR / "generate_synthetic_cohort.py",
        "Step 4/8: Generating Synthetic Cohort (behaviour-seeded)",
        ["synthetic_users.csv", "synthetic_ratings.csv"],
    ),
    (
        ENGINE_DIR / "collaborative_filtering.py",
        "Step 5/8: Training SVD CF Model (evaluation cohort excluded)",
        ["svd_model.pkl"],
    ),
    (
        EVALUATION_DIR / "evaluation_metrics.py",
        "Step 6/8: Master Evaluation (real + synthetic tracks)",
        [
            "evaluation_user_level.csv",
            "rq1_model_performance_real.csv",
            "rq1_significance_tests_real.csv",
            "rq2_diversity_by_model_real.csv",
        ],
    ),
    (
        EVALUATION_DIR / "sensitivity_analysis.py",
        "Step 7/8: Sensitivity Analysis (lambda sweep + archetype mix)",
        [
            "sensitivity_affinity_tables.csv",
            "sensitivity_lambda_summary_real.csv",
            "sensitivity_lambda_summary_synthetic.csv",
        ],
    ),
    (
        EVALUATION_DIR / "make_figures.py",
        "Step 8/8: Generating Figures",
        [os.path.join("figures", "13_tradeoff_curve_synthetic.png")],
    ),
]


def print_banner(text: str, char: str = "=", width: int = 70):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def print_step_header(step_num: int, total: int, description: str):
    print(f"\n{'-' * 60}")
    print(f"  > [{step_num}/{total}] {description}")
    print(f"{'-' * 60}")


def check_artifacts(artifacts: list[str], context: str) -> bool:
    missing = []
    for artifact in artifacts:
        found = (
            (PROCESSED_DIR / artifact).exists()
            or (RESULTS_DIR / artifact).exists()
            or (PROJECT_ROOT / artifact).exists()
        )
        if not found:
            missing.append(artifact)

    if missing:
        print(f"\nERROR: Missing artifacts after {context}:")
        for m in missing:
            print(f"  x {m}")
        return False

    print(f"All {len(artifacts)} expected artifacts verified.")
    return True


def run_script(script_path: Path, description: str) -> bool:
    if not script_path.exists():
        print(f"\nFATAL: Script not found: {script_path}")
        return False

    print(f"\nRunning: {script_path.relative_to(PROJECT_ROOT)}")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-m", _module_name(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
            timeout=3600,
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"\nFAILED ({elapsed:.1f}s): {description}")
            print(f"  Exit code: {result.returncode}")
            return False

        print(f"\nCOMPLETED ({elapsed:.1f}s): {description}")
        return True

    except subprocess.TimeoutExpired:
        print(f"\nTIMEOUT: {description} exceeded 1-hour limit")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        return False


def _module_name(script_path: Path) -> str:
    rel = script_path.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(rel.parts)


def main():
    print_banner("HYBRID MOVIE RECOMMENDER - MASTER PIPELINE")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Python:       {sys.executable}")
    print(f"  Steps:        {len(PIPELINE_STEPS)}")

    if not SRC_DIR.exists():
        print(f"\nFATAL: src/ directory not found at {SRC_DIR}")
        sys.exit(1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_steps = len(PIPELINE_STEPS)
    completed_steps = 0
    pipeline_start = time.time()

    for i, (script_path, description, artifacts) in enumerate(PIPELINE_STEPS, 1):
        print_step_header(i, total_steps, description)

        success = run_script(script_path, description)

        if not success:
            print_banner("PIPELINE HALTED", "!")
            print(f"  Failed at Step {i}/{total_steps}: {description}")
            print("  Fix the error above and re-run: python main.py")
            sys.exit(1)

        if not check_artifacts(artifacts, description):
            print_banner("PIPELINE HALTED", "!")
            print(f"  Step {i}/{total_steps} did not produce its expected artifacts.")
            sys.exit(1)
        completed_steps += 1

    total_elapsed = time.time() - pipeline_start
    print_banner("PIPELINE COMPLETE")
    print(f"  Steps completed:  {completed_steps}/{total_steps}")
    print(f"  Total time:       {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("\n  The evaluation cohort is excluded from SVD training by construction.")
    print("  RQ1 tables: rq1_*_{real,synthetic}.csv")
    print("  RQ2 tables: rq2_*_{real,synthetic}.csv")
    print("  Demo:       python seed_db.py && uvicorn api.main:app")
    print()


if __name__ == "__main__":
    main()
