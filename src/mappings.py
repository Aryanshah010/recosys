from __future__ import annotations

CANONICAL_GENRE_MAP: dict[str, str] = {
    "Science Fiction": "Sci-Fi",
    "Sci-Fi": "Sci-Fi",
    "TV Movie": "TV",
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Documentary": "Documentary",
    "Drama": "Drama",
    "Family": "Family",
    "Fantasy": "Fantasy",
    "History": "History",
    "Horror": "Horror",
    "Music": "Music",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western",
    "Film-Noir": "Thriller",
    "IMAX": "Other",
    "(no genres listed)": "Other",
}

CANONICAL_GENRES: tuple[str, ...] = (
    "Sci-Fi",
    "TV",
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Thriller",
    "War",
    "Western",
)

UNKNOWN_GENRE_LABEL = "Other"

CANONICAL_LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "hi": "Hindi",
    "ko": "Korean",
    "zh": "Chinese",
    "cn": "Chinese",
    "sv": "Swedish",
    "pt": "Portuguese",
    "fi": "Finnish",
    "nl": "Dutch",
    "da": "Danish",
    "pl": "Polish",
    "tr": "Turkish",
    "cs": "Czech",
    "el": "Greek",
    "ne": "Nepali",
}


UNKNOWN_LANGUAGE_LABEL = "Other"
MISSING_LANGUAGE_LABEL = "Unknown"


def map_genre(raw_genre: str) -> str:

    return CANONICAL_GENRE_MAP.get(raw_genre.strip(), UNKNOWN_GENRE_LABEL)


def map_language(raw_code: str) -> str:

    if not isinstance(raw_code, str) or not raw_code.strip():
        return MISSING_LANGUAGE_LABEL
    return CANONICAL_LANGUAGE_MAP.get(raw_code.strip().lower(), UNKNOWN_LANGUAGE_LABEL)
