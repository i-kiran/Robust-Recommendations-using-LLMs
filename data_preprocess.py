#imports 
import pandas as pd
import numpy as np

#load data
def load_data():
    ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')
    users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users_df = pd.read_csv('ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')

    items_cols = ['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
    'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items_df = pd.read_csv('ml-100k/u.item', sep='|', names=items_cols, encoding='latin-1')
    items_df['movie_title'] = items_df['movie_title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
    return ratings_df, users_df, items_df

ratings_df, users_df, items_df = load_data()