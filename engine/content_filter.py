import pandas as pd
import numpy as np
import pickle
import re

GENRE_MAPPING = {
    'Sci-Fi': 'ScienceFiction',
    'Science Fiction': 'ScienceFiction',
    'Children\'s': 'Family',
    'Childrens': 'Family',
    'Film-Noir': 'Crime',   # TMDb lacks Film-Noir, map to closest equivalent
    'Anime': 'Animation',   # Anime is not a TMDb genre, it's Animation + JA
    'Bollywood': 'Drama',   # Fallback proxy
    'Tollywood': 'Action'   # Fallback proxy
}

LANGUAGE_MAPPING = {
    'HINDI': 'HI', 'ENGLISH': 'EN', 'NEPALI': 'NE', 
    'JAPANESE': 'JA', 'KOREAN': 'KO', 'FRENCH': 'FR',
    'SPANISH': 'ES', 'GERMAN': 'DE', 'ITALIAN': 'IT',
    'CHINESE': 'ZH', 'RUSSIAN': 'RU', 'ARABIC': 'AR',
    'BENGALI': 'BN', 'TELUGU': 'TE', 'TAMIL': 'TA',
    'KANNADA': 'KN', 'MALAYALAM': 'ML', 'MARATHI': 'MR',
    'PUNJABI': 'PA', 'URDU': 'UR'
}

def normalize_user_preferences(user_profile):
    """
    Parses pipe-separated CSV strings and maps them to TMDb's exact format.
    """
    # 1. Parse and Map Languages (e.g., "Hindi|English" -> {'HI', 'EN'})
    raw_langs = str(user_profile.get('preferred_languages', '')).split('|')
    pref_langs = set()
    for l in raw_langs:
        l_clean = l.strip().upper()
        pref_langs.add(LANGUAGE_MAPPING.get(l_clean, l_clean))
        
    # 2. Parse and Map Genres (e.g., "Sci-Fi|Action" -> {'ScienceFiction', 'Action'})
    raw_genres = re.split(r'\||,', str(user_profile.get('preferred_genres', '')))
    pref_genres = set()
    for g in raw_genres:
        g_clean = g.strip()
        if g_clean in GENRE_MAPPING:
            pref_genres.add(GENRE_MAPPING[g_clean])
        else:
            # Fallback: remove spaces/hyphens to match TMDb clean_genres logic
            pref_genres.add(g_clean.replace('-', '').replace(' ', ''))
            
    return pref_langs, pref_genres

class CBFEngine:
    def __init__(self, matrix_path='data/processed/cbf_matrix.pkl', meta_path='data/processed/cbf_metadata.pkl'):
        with open(matrix_path, 'rb') as f:
            self.cosine_sim = pickle.load(f)
        with open(meta_path, 'rb') as f:
            self.df = pickle.load(f)
            
        # Map movieId to matrix index for O(1) lookups
        self.indices = pd.Series(self.df.index, index=self.df['movieId']).drop_duplicates()
        
        # Pre-calculate popularity percentiles for Popularity Bias mitigation
        self.df['popularity_score'] = self.df['vote_count'] / self.df['vote_count'].max()

    def _calculate_preference_score(self, movie_row, user_profile):
        """
        Models CONFIRMATION BIAS (RQ2).
        Boosts movies that match the synthetic cohort's assumed cultural/genre traits.
        """
        score = 0.0
        pref_genres = set(user_profile.get('preferred_genres', []))
        pref_langs = set([l.upper() for l in user_profile.get('preferred_languages', [])])
        
        # Language Match (High weight for localized hypothesis)
        if movie_row['language'] in pref_langs:
            score += 0.6 
            
        # Genre Match
        movie_genres = set(movie_row['clean_genres'].split())
        if not movie_genres.isdisjoint(pref_genres):
            score += 0.4
            
        return score

    def recommend_for_user(self, user_profile, liked_movie_ids, k=10, apply_debiasing=False):
        """
        Generates recommendations. 
        Handles Cold-Start via Weight Renormalization (Project Context Rule).
        """
        # 1. Identify valid liked movies in our matrix
        valid_liked_ids = [m for m in liked_movie_ids if m in self.indices.index]
        
        has_cf_signal = len(valid_liked_ids) > 0
        
        # Initialize scores array
        scores = np.zeros(len(self.df))
        
        # 2. Calculate Content Similarity Score (0.3 base weight)
        if has_cf_signal:
            idxs = [self.indices[m] for m in valid_liked_ids]
            content_scores = np.mean(self.cosine_sim[idxs], axis=0)
        else:
            # Cold Start: No interaction history, content score is uniform/zero
            content_scores = np.zeros(len(self.df)) 

        # 3. Calculate Preference Score (0.2 base weight)
        pref_scores = np.array([
            self._calculate_preference_score(row, user_profile) 
            for _, row in self.df.iterrows()
        ])

        # 4. Hybrid Fusion & Renormalization (Thesis Constraint)
        # Base Weights: CF=0.5, Content=0.3, Preference=0.2
        if has_cf_signal:
            # Note: CF score would be added here in the full Hybrid Engine. 
            # For pure CBF evaluation, we just use Content + Preference
            w_content, w_pref = 0.6, 0.4 # Renormalized 0.3 and 0.2
        else:
            # COLD START: CF is unavailable. Renormalize Content (0.3) and Preference (0.2)
            # 0.3 / (0.3+0.2) = 0.6 | 0.2 / (0.3+0.2) = 0.4
            w_content, w_pref = 0.6, 0.4 

        final_scores = (w_content * content_scores) + (w_pref * pref_scores)
        
        # 5. Popularity Bias Mitigation (Optional Debiasing)
        if apply_debiasing:
            # Penalize top 10% most popular movies to increase Catalog Coverage (RQ2)
            popularity_penalty = np.where(self.df['popularity_score'] > 0.9, 0.8, 1.0)
            final_scores = final_scores * popularity_penalty

        # 6. Format and Filter Results
        results = self.df[['movieId', 'title_x', 'clean_genres', 'language']].copy()
        results['final_score'] = final_scores
        
        # Filter out already liked movies
        results = results[~results['movieId'].isin(valid_liked_ids)]
        
        # Sort and return Top-K
        results = results.sort_values(by='final_score', ascending=False).head(k)
        return results.rename(columns={'title_x': 'title'})

# --- THESIS TEST SCENARIO ---
if __name__ == "__main__":
    engine = CBFEngine()
    
    # Synthetic Cohort: Nepali IT Male Student
    synthetic_user = {
        'user_id': 'synth_nepali_it_01',
        'preferred_genres': ['ScienceFiction', 'Action', 'Animation', 'Thriller'],
        'preferred_languages': ['en', 'hi', 'ja'] # English, Hindi, Japanese (Anime)
    }
    
    # Scenario A: Cold Start (No history)
    print("\n--- COLD START (Preference Driven) ---")
    cold_recs = engine.recommend_for_user(synthetic_user, liked_movie_ids=[], k=5)
    print(cold_recs[['title', 'language', 'clean_genres']])
    
    # Scenario B: Returning User (Liked The Matrix - movieId 2571)
    print("\n--- RETURNING USER (Content + Preference) ---")
    returning_recs = engine.recommend_for_user(synthetic_user, liked_movie_ids=[2571], k=5, apply_debiasing=True)
    print(returning_recs[['title', 'language', 'clean_genres']])