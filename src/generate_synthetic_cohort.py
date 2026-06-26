import pandas as pd
import numpy as np
import os
import random

CATALOG_PATH = 'data/processed/movies_final.csv'
PROCESSED_DIR = 'data/processed'

print("Loading validated movie catalog...")
movies = pd.read_csv(CATALOG_PATH) 

movies['clean_genres'] = movies['clean_genres'].fillna('').astype(str)
movies['language'] = movies['language'].fillna('').astype(str).str.upper()

archetypes = [
    {
        'name': 'A_Techie', 'count': 50,
        'langs': ['EN', 'JA'], 
        'genres': ['ScienceFiction', 'Action', 'Thriller', 'Documentary']
    },
    {
        'name': 'B_Mainstream', 'count': 50,
        'langs': ['HI', 'EN'], 
        'genres': ['Action', 'Drama', 'Romance', 'Comedy']
    },
    {
        'name': 'C_AnimeFan', 'count': 50,
        'langs': ['JA', 'KO', 'EN'], 
        'genres': ['Animation', 'Fantasy', 'ScienceFiction']
    },
    {
        'name': 'D_Localist', 'count': 50,
        'langs': ['NE', 'HI', 'EN'], 
        'genres': ['Drama', 'Romance', 'Musical']
    }
]

synthetic_users = []
synthetic_ratings = []
user_id_counter = 1000000 

print("Generating Synthetic Cohort & Validated Interactions...")
for arch in archetypes:
    mask_lang = movies['language'].isin(arch['langs'])
    mask_genre = movies['clean_genres'].apply(lambda x: any(g in x for g in arch['genres']))
    preferred_movies = movies[mask_lang | mask_genre]
    
    if len(preferred_movies) < 50:
        preferred_movies = movies.head(500) 
        
    for i in range(arch['count']):
        uid = user_id_counter
        user_id_counter += 1
        
        is_cold_start = random.random() < 0.15 
        
        synthetic_users.append({
            'user_id': uid, 'country': 'Nepal', 'gender': 'Male', 
            'occupation': 'IT Student', 'cohort_group': arch['name'],
            'preferred_languages': '|'.join(arch['langs']),
            'preferred_genres': '|'.join(arch['genres']),
            'is_cold_start': is_cold_start
        })
        
        if not is_cold_start:
            num_pos = random.randint(15, 30)
            watched_pos = preferred_movies.sample(n=min(num_pos, len(preferred_movies)))
            for _, movie in watched_pos.iterrows():
                synthetic_ratings.append({
                    'userId': uid, 'movieId': movie['movieId'],
                    'rating': random.choice([4.0, 4.5, 5.0]), 'is_synthetic': True
                })
                
            num_neg = random.randint(5, 10)
            watched_neg = movies.sample(n=num_neg)
            for _, movie in watched_neg.iterrows():
                if movie['movieId'] not in watched_pos['movieId'].values:
                    synthetic_ratings.append({
                        'userId': uid, 'movieId': movie['movieId'],
                        'rating': random.choice([2.0, 2.5, 3.0]), 'is_synthetic': True
                    })

os.makedirs(PROCESSED_DIR, exist_ok=True)
pd.DataFrame(synthetic_users).to_csv(f'{PROCESSED_DIR}/synthetic_user_profiles.csv', index=False)
pd.DataFrame(synthetic_ratings).to_csv(f'{PROCESSED_DIR}/synthetic_interactions.csv', index=False)

print(f"Success! Generated {len(synthetic_users)} users and {len(synthetic_ratings)} valid interactions.")