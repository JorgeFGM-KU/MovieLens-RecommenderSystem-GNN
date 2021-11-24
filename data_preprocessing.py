from typing import List, Dict
import pandas as pd

class get_data():
    def __init__(self):
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.user = pd.read_csv('data/raw/u.user',
                                sep='|',
                                names=u_cols,
                                usecols=["user_id", "age", "sex", "occupation"],
                                encoding='latin-1')

        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.rating = pd.read_csv('data/raw/u.data',
                                  sep='\t',
                                  names=r_cols,
                                  usecols=["user_id", "movie_id", "rating"],
                                  encoding='latin-1')
        m_cols = [
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            # features from here:
            "unknown",
            "action",
            "adventure",
            "animation",
            "childrens",
            "comedy",
            "crime",
            "documentary",
            "drama",
            "fantasy",
            "film_noir",
            "horror",
            "musical",
            "mystery",
            "romance",
            "sci_fi",
            "thriller",
            "war",
            "western",
        ]
        self.movie = pd.read_csv('data/raw/u.item',
                                 sep='|',
                                 names=m_cols,
                                 usecols=["movie_id"] + m_cols[5:],
                                 encoding='latin-1')
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
