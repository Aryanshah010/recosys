import os
import pickle
import pandas as pd
import numpy as np
from surprise import dump
import warnings

warnings.filterwarnings("ignore")


class HybridFusionEngine:
    def __init__(self):
        self.PROCESSED_DIR = "data/processed"
        self.is_ready = False
        self.missing_artifacts = []

        self.GENRE_MAPPING = {
            "Sci-Fi": "ScienceFiction",
            "Science Fiction": "ScienceFiction",
            "Anime": "Animation",
            "Children's": "Family",
            "Childrens": "Family",
            "Bollywood": "Drama",
            "Tollywood": "Action",
        }
        self.LANGUAGE_MAPPING = {
            "HINDI": "HI",
            "ENGLISH": "EN",
            "NEPALI": "NE",
            "JAPANESE": "JA",
            "KOREAN": "KO",
            "FRENCH": "FR",
            "SPANISH": "ES",
            "GERMAN": "DE",
            "CHINESE": "ZH",
        }

        self._load_artifacts()

    def _load_artifacts(self):
        print("Loading Recommender Artifacts into Memory...")
        required = [
            "svd_model.pkl",
            "cbf_matrix.pkl",
            "cbf_metadata.pkl",
            "movies_final.csv",
        ]
        self.missing_artifacts = [
            name
            for name in required
            if not os.path.exists(os.path.join(self.PROCESSED_DIR, name))
        ]
        if self.missing_artifacts:
            print(
                "Recommender artifacts missing. Run 'uv run python main.py' first: "
                + ", ".join(self.missing_artifacts)
            )
            return

        _, self.svd_model = dump.load(os.path.join(self.PROCESSED_DIR, "svd_model.pkl"))

        with open(os.path.join(self.PROCESSED_DIR, "cbf_matrix.pkl"), "rb") as f:
            self.cosine_sim = pickle.load(f)
        with open(os.path.join(self.PROCESSED_DIR, "cbf_metadata.pkl"), "rb") as f:
            self.cbf_meta = pickle.load(f)

        self.cbf_indices = pd.Series(
            self.cbf_meta.index, index=self.cbf_meta["movieId"]
        )
        self.cbf_indices = self.cbf_indices[
            ~self.cbf_indices.index.duplicated(keep="first")
        ]

        self.movies_catalog = pd.read_csv(
            os.path.join(self.PROCESSED_DIR, "movies_final.csv")
        )

        max_votes = self.movies_catalog["vote_count"].max()
        if max_votes > 0:
            self.movies_catalog["pop_norm"] = (
                self.movies_catalog["vote_count"] / max_votes
            )
        else:
            self.movies_catalog["pop_norm"] = 0.0

        self.popularity_dict = dict(
            zip(self.movies_catalog["movieId"], self.movies_catalog["pop_norm"])
        )

        self.is_ready = True
        print("Hybrid Fusion Engine Ready.")

    def _normalize_preferences(self, user_profile):
        raw_langs = str(user_profile.get("preferred_languages", "")).split("|")
        raw_genres = str(user_profile.get("preferred_genres", "")).split("|")

        pref_langs = set()
        for l in raw_langs:
            l_clean = l.strip().upper()
            if l_clean:
                pref_langs.add(self.LANGUAGE_MAPPING.get(l_clean, l_clean))

        pref_genres = set()
        for g in raw_genres:
            g_clean = g.strip()
            if g_clean:
                mapped = self.GENRE_MAPPING.get(
                    g_clean, g_clean.replace(" ", "").replace("-", "")
                )
                pref_genres.add(mapped)

        return pref_langs, pref_genres

    def _get_preference_score_and_reason(self, movie_row, pref_langs, pref_genres):
        score = 0.0
        matched_attrs = []

        m_lang = (
            str(movie_row.get("language", "")).upper()
            if pd.notna(movie_row.get("language"))
            else ""
        )
        m_genres_raw = movie_row.get("clean_genres", "")
        m_genres = set(str(m_genres_raw).split()) if pd.notna(m_genres_raw) else set()

        if m_lang and m_lang in pref_langs:
            score += 0.6
            matched_attrs.append(m_lang)

        genre_overlap = m_genres.intersection(pref_genres)
        if genre_overlap:
            score += 0.4
            matched_attrs.extend(list(genre_overlap))

        return score, matched_attrs

    def _compute_adaptive_weights(self, user_history_len, is_cold_start):
        if is_cold_start:
            return {"cf": 0.0, "cbf": 0.7, "pref": 0.3}
        elif user_history_len < 5:  # Very cold
            return {"cf": 0.2, "cbf": 0.5, "pref": 0.3}
        elif user_history_len < 15:  # Warm
            return {"cf": 0.4, "cbf": 0.4, "pref": 0.2}
        else:  # Warm user
            return {"cf": 0.6, "cbf": 0.2, "pref": 0.2}

    def _diversify_recommendations(self, recommendations, diversity_lambda=0.3):
        final_recs = []
        selected_languages = set()
        selected_genres = set()

        for rec in recommendations:
            lang_bonus = 1.0 if rec["language"] not in selected_languages else 0.7
            rec_genres = set(str(rec["genres"]).split()) if rec["genres"] else set()
            genre_bonus = 1.0 if not rec_genres.issubset(selected_genres) else 0.7

            diversity_score = (lang_bonus + genre_bonus) / 2
            adjusted = (1 - diversity_lambda) * rec[
                "score"
            ] + diversity_lambda * diversity_score

            final_recs.append({**rec, "adjusted_score": round(adjusted, 4)})
            if rec["language"]:
                selected_languages.add(rec["language"])
            selected_genres.update(rec_genres)

        return sorted(final_recs, key=lambda x: x["adjusted_score"], reverse=True)

    def _require_ready(self):
        if not self.is_ready:
            missing = ", ".join(self.missing_artifacts) or "processed artifacts"
            raise RuntimeError(
                "Recommender artifacts are missing. Run 'uv run python main.py' first. "
                f"Missing: {missing}"
            )

    def _get_matrix_idx(self, movie_id):
        idx_val = self.cbf_indices.get(movie_id)
        if idx_val is None:
            return None
        if isinstance(idx_val, pd.Series):
            return int(idx_val.iloc[0])
        return int(idx_val)

    def _build_candidates(self, valid_history, user_history, count=500):
        candidates = set()

        if valid_history:
            safe_idxs = []
            for movie_id in valid_history:
                idx_val = self._get_matrix_idx(movie_id)
                if idx_val is not None:
                    safe_idxs.append(idx_val)

            if safe_idxs:
                unique_idxs = np.unique(safe_idxs)
                sim_matrix_slice = self.cosine_sim[unique_idxs, :]
                sim_scores = np.mean(sim_matrix_slice, axis=0)

                top_k_count = min(count, len(sim_scores))
                top_cbf_idx = np.argsort(sim_scores)[-top_k_count:]
                candidates.update(self.cbf_meta.iloc[top_cbf_idx]["movieId"].tolist())

        candidates.update(
            self.movies_catalog.nlargest(count, "vote_count")["movieId"].tolist()
        )
        return list(candidates - set(user_history))

    def _score_candidates(self, user_id, user_profile, user_history):
        pref_langs, pref_genres = self._normalize_preferences(user_profile)
        valid_history = [m for m in user_history if m in self.cbf_indices.index]
        candidates = self._build_candidates(valid_history, user_history)

        scored_by_model = {
            "CF": [],
            "CBF": [],
            "NonLocal_Hybrid": [],
            "Localized_Hybrid": [],
        }
        is_cold_start = len(user_history) == 0
        weights = self._compute_adaptive_weights(len(user_history), is_cold_start)

        history_idxs = [
            self._get_matrix_idx(history_movie) for history_movie in valid_history
        ]
        history_idxs = [idx for idx in history_idxs if idx is not None]

        for mid in candidates:
            matrix_idx = self._get_matrix_idx(mid)
            if matrix_idx is None:
                continue

            movie_row = self.cbf_meta.loc[matrix_idx]

            try:
                cf_pred = self.svd_model.predict(str(user_id), mid).est / 5.0
            except Exception:
                cf_pred = 0.0

            cbf_score = 0.0
            if history_idxs:
                cbf_score = float(
                    np.mean([self.cosine_sim[h_idx][matrix_idx] for h_idx in history_idxs])
                )

            pref_score, matched_attrs = self._get_preference_score_and_reason(
                movie_row, pref_langs, pref_genres
            )

            pop_score = self.popularity_dict.get(mid, 0.0)
            localized_score = (
                (weights["cf"] * cf_pred)
                + (weights["cbf"] * cbf_score)
                + (weights["pref"] * pref_score)
            )

            if is_cold_start:
                localized_score += 0.01 * pop_score
                if matched_attrs:
                    localized_reason = (
                        "Cold-Start: Matches your preference for "
                        + ", ".join(matched_attrs[:3])
                    )
                else:
                    localized_reason = (
                        "Cold-Start: Popular recommendation for your demographic profile"
                    )
            else:
                localized_reason = (
                    "Hybrid: Recommended from SVD, content similarity, and preferences"
                )

            base_rec = {
                "movieId": int(mid),
                "title": movie_row.get("title_x", movie_row.get("title", "Unknown")),
                "genres": movie_row.get("clean_genres", ""),
                "language": movie_row.get("language", ""),
            }

            scored_by_model["CF"].append(
                {
                    **base_rec,
                    "score": round(cf_pred, 4),
                    "explanation": "CF: SVD prediction from real MovieLens ratings only",
                }
            )
            scored_by_model["CBF"].append(
                {
                    **base_rec,
                    "score": round(cbf_score, 4),
                    "explanation": "CBF: Similar to the synthetic user's support history",
                }
            )
            scored_by_model["NonLocal_Hybrid"].append(
                {
                    **base_rec,
                    "score": round(0.625 * cf_pred + 0.375 * cbf_score, 4),
                    "explanation": "Non-local hybrid: fixed SVD + CBF blend",
                }
            )
            scored_by_model["Localized_Hybrid"].append(
                {
                    **base_rec,
                    "score": round(localized_score, 4),
                    "explanation": localized_reason,
                }
            )

        return scored_by_model

    def recommend(self, user_id, user_profile, user_history=None, k=10):
        self._require_ready()

        if user_history is None:
            user_history = []

        scored_movies = self._score_candidates(user_id, user_profile, user_history)[
            "Localized_Hybrid"
        ]

        scored_movies.sort(key=lambda x: x["score"], reverse=True)
        diversified_movies = self._diversify_recommendations(
            scored_movies[: max(k * 2, 50)], diversity_lambda=0.3
        )

        for movie in diversified_movies:
            movie["score"] = movie.pop("adjusted_score")

        return diversified_movies[:k]

    def compare_models(self, user_id, user_profile, user_history=None, k=10):
        self._require_ready()

        if user_history is None:
            user_history = []

        scored_by_model = self._score_candidates(user_id, user_profile, user_history)
        compared = {}
        for model_name, scored_movies in scored_by_model.items():
            scored_movies.sort(key=lambda x: x["score"], reverse=True)
            if model_name == "Localized_Hybrid":
                model_recs = self._diversify_recommendations(
                    scored_movies[: max(k * 2, 50)], diversity_lambda=0.3
                )
                for movie in model_recs:
                    movie["score"] = movie.pop("adjusted_score")
                compared[model_name] = model_recs[:k]
            else:
                compared[model_name] = scored_movies[:k]

        return compared


if __name__ == "__main__":
    engine = HybridFusionEngine()

    test_profile = {
        "preferred_languages": "English|Japanese|Hindi",
        "preferred_genres": "Sci-Fi|Action|Animation",
    }

    print("\nGenerating Cold-Start Recommendations")
    recs = engine.recommend(
        user_id="999999", user_profile=test_profile, user_history=[], k=5
    )
    for r in recs:
        print(f"[{r['score']}] {r['title']} ({r['language']}) - {r['explanation']}")
