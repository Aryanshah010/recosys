import os
import pickle
import pandas as pd
import numpy as np
from surprise import dump
import warnings
warnings.filterwarnings('ignore')

class HybridFusionEngine:
    """
    Real-time inference engine for the Hybrid Movie Recommender.
    Implements the thesis formula: 0.5*CF + 0.3*Content + 0.2*Preference
    with Cold-Start renormalization (0.6*Content + 0.4*Preference).
    """
    def __init__(self):
        self.PROCESSED_DIR = "data/processed"
        
        # Cross-Dataset Ontology Mappings (Synthetic Profile -> TMDb Cleaned Tokens)
        self.GENRE_MAPPING = {
            'Sci-Fi': 'ScienceFiction', 'Science Fiction': 'ScienceFiction',
            'Anime': 'Animation', "Children's": 'Family', 'Childrens': 'Family',
            'Bollywood': 'Drama', 'Tollywood': 'Action'
        }
        self.LANGUAGE_MAPPING = {
            'HINDI': 'HI', 'ENGLISH': 'EN', 'NEPALI': 'NE',
            'JAPANESE': 'JA', 'KOREAN': 'KO', 'FRENCH': 'FR',
            'SPANISH': 'ES', 'GERMAN': 'DE', 'CHINESE': 'ZH'
        }
        
        self._load_artifacts()

    def _load_artifacts(self):
        print("🔄 Loading Recommender Artifacts into Memory...")
        # 1. Load SVD (CF)
        _, self.svd_model = dump.load(os.path.join(self.PROCESSED_DIR, "svd_model.pkl"))
        
        # 2. Load CBF Matrix & Metadata
        with open(os.path.join(self.PROCESSED_DIR, "cbf_matrix.pkl"), 'rb') as f:
            self.cosine_sim = pickle.load(f)
        with open(os.path.join(self.PROCESSED_DIR, "cbf_metadata.pkl"), 'rb') as f:
            self.cbf_meta = pickle.load(f)
            
        # Ensure strict 1:1 mapping and handle any residual duplicates safely
        self.cbf_indices = pd.Series(self.cbf_meta.index, index=self.cbf_meta['movieId'])
        self.cbf_indices = self.cbf_indices[~self.cbf_indices.index.duplicated(keep='first')]
        
        # 3. Load Full Catalog for Popularity Fallback & Tie-breaking
        self.movies_catalog = pd.read_csv(os.path.join(self.PROCESSED_DIR, "movies_final.csv"))
        
        # Pre-calculate global popularity percentiles for tie-breaking
        max_votes = self.movies_catalog['vote_count'].max()
        if max_votes > 0:
            self.movies_catalog['pop_norm'] = self.movies_catalog['vote_count'] / max_votes
        else:
            self.movies_catalog['pop_norm'] = 0.0
            
        self.popularity_dict = dict(zip(self.movies_catalog['movieId'], self.movies_catalog['pop_norm']))
        
        print("✅ Hybrid Fusion Engine Ready.")

    def _normalize_preferences(self, user_profile):
        """Parses and maps user preferences to TMDb matrix tokens."""
        raw_langs = str(user_profile.get('preferred_languages', '')).split('|')
        raw_genres = str(user_profile.get('preferred_genres', '')).split('|')
        
        pref_langs = set()
        for l in raw_langs:
            l_clean = l.strip().upper()
            if l_clean:
                pref_langs.add(self.LANGUAGE_MAPPING.get(l_clean, l_clean))
                
        pref_genres = set()
        for g in raw_genres:
            g_clean = g.strip()
            if g_clean:
                mapped = self.GENRE_MAPPING.get(g_clean, g_clean.replace(' ', '').replace('-', ''))
                pref_genres.add(mapped)
                
        return pref_langs, pref_genres

    def _get_preference_score_and_reason(self, movie_row, pref_langs, pref_genres):
        """Calculates the Preference Match Score and returns matched attributes for explanation."""
        score = 0.0
        matched_attrs = []
        
        m_lang = str(movie_row.get('language', '')).upper() if pd.notna(movie_row.get('language')) else ""
        m_genres_raw = movie_row.get('clean_genres', '')
        m_genres = set(str(m_genres_raw).split()) if pd.notna(m_genres_raw) else set()
        
        if m_lang and m_lang in pref_langs: 
            score += 0.6
            matched_attrs.append(m_lang)
            
        genre_overlap = m_genres.intersection(pref_genres)
        if genre_overlap: 
            score += 0.4
            matched_attrs.extend(list(genre_overlap))
            
        return score, matched_attrs

    def recommend(self, user_id, user_profile, user_history=None, k=10):
        """
        Generates Top-K explainable recommendations.
        """
        if user_history is None:
            user_history = []
            
        is_cold_start = len(user_history) == 0
        pref_langs, pref_genres = self._normalize_preferences(user_profile)
        
        # 1. Generate Candidate Pool (Optimized)
        candidates = set()
        valid_history = [m for m in user_history if m in self.cbf_indices.index]
        
        if valid_history:
            # Safe index extraction to prevent inhomogeneous array errors
            safe_idxs = []
            for m in valid_history:
                idx_val = self.cbf_indices.get(m)
                if isinstance(idx_val, pd.Series):
                    safe_idxs.append(int(idx_val.iloc[0]))
                else:
                    safe_idxs.append(int(idx_val))
            
            unique_idxs = np.unique(safe_idxs)
            sim_matrix_slice = self.cosine_sim[unique_idxs, :]
            sim_scores = np.mean(sim_matrix_slice, axis=0)
            
            top_k_count = min(500, len(sim_scores))
            top_cbf_idx = np.argsort(sim_scores)[-top_k_count:]
            candidates.update(self.cbf_meta.iloc[top_cbf_idx]['movieId'].tolist())
            
        # Add Popularity Candidates
        candidates.update(self.movies_catalog.nlargest(500, 'vote_count')['movieId'].tolist())
        
        # Remove already watched
        candidates = list(candidates - set(user_history))
        
        # 2. Score Candidates
        scored_movies = []
        
        for mid in candidates:
            if mid not in self.cbf_indices.index:
                continue
                
            matrix_idx = self.cbf_indices[mid]
            if isinstance(matrix_idx, pd.Series): matrix_idx = int(matrix_idx.iloc[0])
            else: matrix_idx = int(matrix_idx)
                
            movie_row = self.cbf_meta.loc[matrix_idx]
            
            # A. CF Score (Normalized 0-1)
            try:
                cf_pred = self.svd_model.predict(str(user_id), mid).est / 5.0
            except Exception:
                cf_pred = 0.0 
                
            # B. CBF Score
            cbf_score = 0.0
            if valid_history:
                cbf_score = float(np.mean([self.cosine_sim[self.cbf_indices[h]][matrix_idx] for h in valid_history]))
                
            # C. Preference Score & Explainability
            pref_score, matched_attrs = self._get_preference_score_and_reason(movie_row, pref_langs, pref_genres)
            
            # Popularity Tie-breaker (0.0 to 1.0)
            pop_score = self.popularity_dict.get(mid, 0.0)
            
            # D. Hybrid Fusion & Explainability
            if is_cold_start:
                # Cold Start Renormalization: 0.6 CBF + 0.4 Preference + tiny popularity tie-breaker
                final_score = (0.6 * cbf_score) + (0.4 * pref_score) + (0.01 * pop_score)
                
                if matched_attrs:
                    reason = f"Cold-Start: Matches your preference for {', '.join(matched_attrs[:3])}"
                else:
                    reason = "Cold-Start: Popular recommendation for your demographic profile"
            else:
                # Returning User: 0.5 CF + 0.3 CBF + 0.2 Preference
                final_score = (0.5 * cf_pred) + (0.3 * cbf_score) + (0.2 * pref_score)
                reason = "Hybrid: Recommended based on similar user behavior and your genre preferences"
                
            scored_movies.append({
                'movieId': int(mid),
                'title': movie_row.get('title_x', movie_row.get('title', 'Unknown')),
                'genres': movie_row.get('clean_genres', ''),
                'language': movie_row.get('language', ''),
                'score': round(final_score, 4),
                'explanation': reason
            })
            
        # 3. Sort and Return Top-K
        scored_movies.sort(key=lambda x: x['score'], reverse=True)
        return scored_movies[:k]

# --- QUICK TEST ---
if __name__ == "__main__":
    engine = HybridFusionEngine()
    
    # Simulate a Cold-Start Nepali IT Student
    # Notice we use full names here; the engine's Ontology Mapping will translate them to TMDb tokens
    test_profile = {
        'preferred_languages': 'English|Japanese|Hindi', 
        'preferred_genres': 'Sci-Fi|Action|Animation'     
    }
    
    print("\n--- Generating Cold-Start Recommendations ---")
    recs = engine.recommend(user_id="999999", user_profile=test_profile, user_history=[], k=5)
    for r in recs:
        print(f"[{r['score']}] {r['title']} ({r['language']}) - {r['explanation']}")