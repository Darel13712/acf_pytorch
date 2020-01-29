import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from torch.utils.data import Dataset
from numpy.random import randint

from download_movielens import download

class MovieLens(Dataset):
    """
    Handles loading of movielens.
    Also provides sampling of positive and negative items for user.
    """
    def __init__(self, dataset: str, threshold=4, unknown=3.0):
        if not os.path.exists(dataset):
            download(dataset)

        self.movie_labels = LabelEncoder()
        self.user_labels = LabelEncoder()
        self.threshold = threshold
        self.unknown = unknown

        movies  = pd.read_csv(os.path.join(dataset, 'movies.csv'), index_col='movieId')
        ratings = pd.read_csv(os.path.join(dataset, 'ratings.csv'))

        movies.index = self.movie_labels.fit_transform(movies.index)
        ratings['userId'] = self.user_labels.fit_transform(ratings['userId'])
        ratings['movieId'] = self.movie_labels.transform(ratings['movieId'])

        self.movies = movies
        self.total_movies = len(movies)
        self.genres = self._extract_genres(movies)
        self.set_scope(ratings)

    def set_scope(self, ratings):
        self.ratings = ratings
        self.users = self.ratings.userId.unique()
        self.positive = self.ratings.loc[self.ratings['rating'] >= self.threshold].index
        self.negative = self.ratings.loc[self.ratings['rating'] < self.threshold].index
        self.gr_users_pos = ratings.loc[self.positive].groupby('userId')
        self.gr_users = ratings.groupby('userId')

    def pos_data(self, user):
        return self.ratings.loc[self.gr_users_pos.groups[user]]

    @staticmethod
    def _extract_genres(movies):
        genres = movies.genres.apply(lambda x: x.split('|'))
        mlb = MultiLabelBinarizer()
        data = mlb.fit_transform(genres)
        return pd.DataFrame(data, columns=mlb.classes_, index=movies.index)

    @property
    def feature_dim(self):
        return self.genres.shape[1]

    def __getitem__(self, index):
        row = self.ratings.loc[self.positive[index]]
        user, pos, pos_score, _ = row
        neg, neg_score = self.get_negative(user)
        return user, pos, pos_score, neg, neg_score

    def _neg_score(self, x):
        """
        Get score for negative item <x>
        """
        if x in self.negative:
            return self.ratings.loc[x, 'rating']
        else:
            return self.unknown

    def get_negative(self, user):
        """
        Sample negative item for <user>

        Parameters
        ----------
        user: int
            user id

        Returns
        -------
        array, array
            negative item id and corresponding explicit rating
        """
        candidate = randint(self.total_movies)
        seen = self.pos_data(user)
        while candidate in seen.movieId:
            candidate = randint(self.total_movies)
        # unseen = self.not_liked_movies(user)
        # negative = np.random.choice(unseen, 1)[0]
        neg_score = self._neg_score(candidate)

        return candidate, neg_score

    def not_liked_movies(self, user):
        if user in self.gr_users_pos.groups.keys():
            return self.movies.index[~self.movies.index.isin(self.pos_data(user).movieId)]
        else:
            return self.movies.index

    def get_positive(self, user, limit=-1):
        """
        Sample no more than <limit> positive items for <user>
        Used to enrich user embedding

        Parameters
        ----------
        user: int
            user id

        limit: int
            if -1 returns all positive items, else <limit> best of them.

        Returns
        -------
        array
            positive item ids
        """
        seen = self.pos_data(user)
        if limit > 0:
            seen = seen.head(limit)
        return seen.movieId.values

    def get_features(self, ids):
        return self.genres.loc[ids].values

    def __len__(self):
        return len(self.positive)