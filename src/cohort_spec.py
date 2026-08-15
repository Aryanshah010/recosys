from __future__ import annotations

from typing import TypedDict


class ArchetypeSpec(TypedDict):
    name: str
    target_share: float
    language: str | None
    min_share: float
    priority: int


ARCHETYPE_SPECS: tuple[ArchetypeSpec, ...] = (
    {
        "name": "anime",
        "target_share": 0.20,
        "language": "Japanese",
        "min_share": 0.25,
        "priority": 1,
    },
    {
        "name": "bollywood",
        "target_share": 0.20,
        "language": "Hindi",
        "min_share": 0.10,
        "priority": 2,
    },
    {
        "name": "korean",
        "target_share": 0.15,
        "language": "Korean",
        "min_share": 0.10,
        "priority": 3,
    },
    {
        "name": "hollywood",
        "target_share": 0.35,
        "language": "English",
        "min_share": 0.85,
        "priority": 4,
    },
    {
        "name": "mixed",
        "target_share": 0.10,
        "language": None,
        "min_share": 0.0,
        "priority": 5,
    },
)

ARCHETYPE_NAMES: tuple[str, ...] = tuple(a["name"] for a in ARCHETYPE_SPECS)
ARCHETYPE_BY_NAME: dict[str, ArchetypeSpec] = {a["name"]: a for a in ARCHETYPE_SPECS}

COHORT_LANGUAGES: tuple[str, ...] = (
    "English",
    "Hindi",
    "Japanese",
    "Korean",
    "Nepali",
)

COHORT_AGE_RANGE = (19, 26)
COHORT_GENDER = "Male"
COHORT_EDUCATION_WEIGHTS: dict[str, float] = {
    "BSc Computing": 0.60,
    "BIT": 0.20,
    "BCA": 0.15,
    "BE Software": 0.05,
}

N_COHORT_USERS = 400
SYNTHETIC_USER_ID_OFFSET = 900_000

LIKE_THRESHOLD = 3.5
MIN_LIKED_RATINGS = 20

N_INFERRED_GENRES = 4
MIN_INFERRED_LANGUAGE_SHARE = 0.10
MAX_INFERRED_LANGUAGES = 3

TRAIN_SPLIT_RATIO = 0.8
RANDOM_STATE = 42

COHORT_DESIGN_NOTE: str = (
    "Proxy Cohort Design\n"
    "-------------------\n"
    f"  Target users    : {N_COHORT_USERS}\n"
    f"  Age range       : {COHORT_AGE_RANGE[0]}–{COHORT_AGE_RANGE[1]}\n"
    f"  Gender          : {COHORT_GENDER}\n"
    f"  Languages       : {', '.join(COHORT_LANGUAGES)}\n"
    f"  Like threshold  : ≥{LIKE_THRESHOLD} stars, ≥{MIN_LIKED_RATINGS} liked ratings\n"
    f"  Train/test split: {TRAIN_SPLIT_RATIO:.0%} / {1 - TRAIN_SPLIT_RATIO:.0%}\n"
    "  Archetypes:\n"
    + "".join(
        f"    [{a['priority']}] {a['name']:12s} target={a['target_share']:.0%}  "
        f"min_share={a['min_share']:.0%}  lang={a['language'] or 'any'}\n"
        for a in ARCHETYPE_SPECS
    )
)


def assert_valid() -> None:
    total = sum(a["target_share"] for a in ARCHETYPE_SPECS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Archetype target_share must sum to 1.0, got {total}.")

    priorities = [a["priority"] for a in ARCHETYPE_SPECS]
    if len(set(priorities)) != len(priorities):
        raise ValueError("Archetype priorities must be unique.")

    edu_total = sum(COHORT_EDUCATION_WEIGHTS.values())
    if abs(edu_total - 1.0) > 1e-6:
        raise ValueError(f"Education weights must sum to 1.0, got {edu_total}.")

    fallbacks = [a for a in ARCHETYPE_SPECS if a["language"] is None]
    if len(fallbacks) != 1:
        raise ValueError(
            "Exactly one archetype must be the language-agnostic fallback."
        )


assert_valid()


def target_counts(n_users: int = N_COHORT_USERS) -> dict[str, int]:
    counts = {
        a["name"]: int(round(a["target_share"] * n_users)) for a in ARCHETYPE_SPECS
    }
    drift = n_users - sum(counts.values())
    if drift:
        largest = max(ARCHETYPE_SPECS, key=lambda a: a["target_share"])["name"]
        counts[largest] += drift
    return counts


def label_from_language_shares(shares: dict[str, float]) -> str:
    for spec in sorted(ARCHETYPE_SPECS, key=lambda a: a["priority"]):
        lang = spec["language"]
        if lang is None:
            continue
        if shares.get(lang, 0.0) >= spec["min_share"]:
            return spec["name"]
    return next(a["name"] for a in ARCHETYPE_SPECS if a["language"] is None)
