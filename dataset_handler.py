import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from download_movielens import download


class MovieLens(Dataset):
    """
    Handles loading of movielens.
    Also provides sampling of positive and negative items for user.
    """
    def __init__(self, dataset, threshold=4, unknown=3):
        if not os.path.exists(dataset):
            download(dataset)

        self.threshold = threshold
        self.unknown = unknown
        self.movies  = pd.read_csv(os.path.join(dataset, 'movies.csv'), index_col='movieId')
        self.genres = self._extract_genres()
        ratings = pd.read_csv(os.path.join(dataset, 'ratings.csv'))
        self.set_scope(ratings)


    def set_scope(self, ratings):
        self.ratings = ratings
        self.users = self.ratings.userId.unique()
        self.positive = self.ratings.loc[self.ratings['rating'] >= self.threshold]
        self.negative = self.ratings.loc[self.ratings['rating'] < self.threshold]



    def _extract_genres(self):

        genres = self.movies.genres.apply(lambda x: x.split('|'))

        mlb = MultiLabelBinarizer()
        data = mlb.fit_transform(genres)

        return pd.DataFrame(data, columns=mlb.classes_, index=self.movies.index)

    @property
    def feature_dim(self):
        return self.genres.shape[1]

    def __getitem__(self, index):
        return self.positive.iloc[index]

    def _get_neg_score(self, negative):
        return np.array([self._neg_score(n) for n in negative])


    def _neg_score(self, x):
        """
        Get score for negative item <x>
        """
        if x in self.negative.index:
            return self.negative.loc[x, 'rating']
        else:
            return self.unknown

    def get_negative(self, user, batch=30):
        """
        Sample <batch> negative items for <user>

        Parameters
        ----------
        user: int
            user id

        batch: int
            number of negative items to return

        Returns
        -------
        array, array
            negative item ids and corresponding explicit ratings
        """

        seen = self.positive.loc[self.positive.userId == user, 'movieId']
        unseen = self.movies.index[~self.movies.index.isin(seen)]

        negative = np.random.choice(unseen, batch)
        neg_score = self._get_neg_score(negative)

        return negative, neg_score

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
        seen = self.positive.loc[self.positive.userId == user]
        seen = seen.sort_values('rating', ascending=False)
        if limit > 0:
            seen = seen.head(limit)
        return seen.movieId

    def get_features(self, ids):
        return self.genres.loc[ids]

    def get_movies(self, ids):
        return self.movies.loc[ids]

    def shuffle(self):
        return np.random.permutation(len(self.positive))


    def __len__(self):
        return len(self.positive)