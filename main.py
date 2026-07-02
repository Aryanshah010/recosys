"""
Single entry point that executes the full sequential pipeline:
1. Data Cleaning & ETL (MovieLens + TMDb fusion)
2. CBF Matrix Generation (TF-IDF + Cosine Similarity)
3. Synthetic Cohort Generation (Nepali IT Male Students)
4. Collaborative Filtering Training (SVD on MovieLens real ratings only)
5. Master Evaluation (synthetic support/holdout + bias metrics)
"""

import subprocess
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
ENGINE_DIR=PROJECT_ROOT / "engine"
EVALUATION_DIR=PROJECT_ROOT / "evaluation"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

PIPELINE_STEPS = [
    (
        SRC_DIR / "clean_data.py",
        "Step 1/5: Data Cleaning & ETL (MovieLens + TMDb Fusion)",
        ["movies_final.csv", "ratings_final.csv", "cbf_items.csv"]
    ),
    (
        SRC_DIR / "02_build_cbf_matrix.py",
        "Step 2/5: Building CBF Matrix (TF-IDF + Cosine Similarity)",
        ["cbf_matrix.pkl", "cbf_metadata.pkl"]
    ),
    (
        SRC_DIR / "generate_synthetic_cohort.py",
        "Step 3/5: Generating Synthetic Nepali IT Male Cohort",
        ["synthetic_user_profiles.csv", "synthetic_interactions.csv"]
    ),
    (
        ENGINE_DIR / "collaborative_filter.py",
        "Step 4/5: Training SVD CF Model (MovieLens real ratings only)",
        ["svd_model.pkl"]
    ),
    (
        EVALUATION_DIR / "evaluation_metrics.py",
        "Step 5/5: Running Master Evaluation (Synthetic Support/Holdout)",
        ["thesis_evaluation_metrics.csv"]
    ),
]


def print_banner(text: str, char: str = "=", width: int = 70):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")

def print_step_header(step_num: int, total: int, description: str):
    print(f"\n{'─' * 60}")
    print(f"  ▶ [{step_num}/{total}] {description}")
    print(f"{'─' * 60}")


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
        print(f"\nWARNING: Missing artifacts after {context}:")
        for m in missing:
            print(f" ✗ {m}")
        return False
    
    print(f"All {len(artifacts)} expected artifacts verified.")
    return True


def run_script(script_path: Path, description: str) -> bool:
    if not script_path.exists():
        print(f"\nFATAL: Script not found: {script_path}")
        print(f"   Ensure '{script_path.name}' exists in the src/ directory.")
        return False

    print(f"\n Running: {script_path.relative_to(PROJECT_ROOT)}")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,  
            text=True,
            timeout=3600  
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"\n FAILED ({elapsed:.1f}s): {description}")
            print(f"   Exit code: {result.returncode}")
            return False

        print(f"\n  COMPLETED ({elapsed:.1f}s): {description}")
        return True

    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT: {description} exceeded 1-hour limit")
        return False
    except Exception as e:
        print(f"\n ERROR: {e}")
        return False

def main():
    print_banner("HYBRID MOVIE RECOMMENDER — MASTER PIPELINE")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Python:       {sys.executable}")
    print(f"  Steps:        {len(PIPELINE_STEPS)}")

    if not SRC_DIR.exists():
        print(f"\n FATAL: src/ directory not found at {SRC_DIR}")
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
            print("  Completed steps will be skipped via artifact checks.\n")
            sys.exit(1)

        check_artifacts(artifacts, description)
        completed_steps += 1

    # Final summary
    total_elapsed = time.time() - pipeline_start
    print_banner("PIPELINE COMPLETE ✓")
    print(f"  Steps completed:  {completed_steps}/{total_steps}")
    print(f"  Total time:       {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results saved to: {RESULTS_DIR / 'thesis_evaluation_metrics.csv'}")
    print("\n Open the results CSV to extract tables for Chapter 4.")
    print("  SVD was trained on MovieLens only; synthetic users were evaluation-only.")
    print("  Use Filter_Bubble_Score column for Chapter 5 (RQ2/Ethics).")
    print()


if __name__ == "__main__":
    main()
