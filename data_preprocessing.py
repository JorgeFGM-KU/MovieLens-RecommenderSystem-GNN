from typing import List, Dict
import pandas as pd

class get_data():
    def __init__(self):
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.user = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.rating = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
        m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        self.movie = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5), encoding='latin-1')
        self.movie_rating = pd.merge(self.movie, self.rating)
        self.lens = pd.merge(self.movie_rating, self.user)

    def users(self):
        return self.user

    def ratings(self):
        return self.rating

    def movies(self):
        return self.movie

    def all(self):
        #return one merged DataFrame of users, ratings and movies
        return self.lens