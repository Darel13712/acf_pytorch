from copy import deepcopy

from logger import Log
from tqdm import tqdm
import torch
import numpy as np

from metrics import ndcg_score


def negative_head(g, n):
    """
    Let a be an array
    If tail returns mask of 0 and 1 to select elements from array
    >>> tail(a, 2)
    [0, 0, 0, 1, 1]

    Then negative head would be
    >>> negative_head(a, 2)
    [1, 1, 1, 0, 0]

    Not a real return of the function, just explanation.

    g in function stays because it is supposed to be casted on grouped objects
    """
    return g._selected_obj[g.cumcount(ascending=False) >= n]


class Trainer():
    def __init__(self, model, dataset, loss, optimizer, run_name, device=None, test_size=10):
        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.optimizer = optimizer
        self.logger = Log(run_name)
        self.train_test_split(test_size)
        self.best_loss = float("inf")
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

    def train_test_split(self, num=10):
        """
        Leave <num> last items for each user for test
        """
        ratings = self.dataset.ratings.sort_values('timestamp')
        r = ratings.groupby('userId')

        train_r = negative_head(r, num)
        test_r = r.tail(num)

        train = deepcopy(self.dataset)
        test = deepcopy(self.dataset)

        train.set_scope(train_r)
        test.set_scope(test_r)

        self.train = train
        self.test = test

    @staticmethod
    def sort(items, score):
        order = np.argsort(score.cpu().numpy())[::-1]
        top = np.take(items, order)
        return top

    def score(self):
        ratings = self.train.ratings
        movies = self.dataset.movies.index
        score = []

        for userId, df in self.test.positive.groupby('userId'):
            watched_movies = ratings.loc[ratings.userId == userId]
            unseen = movies[~movies.isin(watched_movies.movieId)]
            user = self.get_user_embedding(userId)
            pred = self.get_predictions(user, unseen)

            top = self.sort(unseen, pred)
            gain = df.set_index('movieId').loc[top, 'rating'].fillna(0)
            best = df.sort_values('rating')['rating']

            score.append(ndcg_score(best, gain, k=10))

        return np.mean(score)

    @property
    def state(self):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state


    def fit(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):

            for phase in ['train', 'val']:
                self.logger.epoch(epoch, phase)

                self.model.train(phase == 'train')

                loss = 0

                if phase == 'train':
                    for idx in self.train.shuffle():
                        row = self.train[idx]
                        self.optimizer.zero_grad()
                        cur_loss  = self.training_step(row)
                        self.optimizer.step()
                        loss += cur_loss

                    loss /= len(self.train)
                    self.logger.metrics(loss, 0, epoch, phase)
                else:
                    with torch.no_grad():
                        for idx in self.test.shuffle():
                            row = self.test[idx]
                            cur_loss = self.validation_step(row)
                            loss += cur_loss

                        loss /= len(self.test)
                        score = self.score()
                        self.logger.metrics(loss, score, epoch, phase)

                        if loss < self.best_loss:
                            self.best_loss = loss
                            self.logger.save(self.state, epoch)


    def get_user_embedding(self, user):
        items = self.dataset.get_positive(user)
        feats = self.dataset.get_features(items).values
        user = self.model(user, items, feats)
        return user


    def get_predictions(self, user, items):
        item_embeddings = self.model.get_items(items)
        prediction = self.model.score(user, item_embeddings)
        return prediction


    def training_step(self, row):

        user, pos, pos_score, _ = row
        neg, neg_score = self.train.get_negative(user)

        user = self.get_user_embedding(user)

        pos_pred = self.get_predictions(user, [pos])
        neg_pred = self.get_predictions(user, neg)

        loss = self.loss(pos_pred, pos_score, neg_pred, neg_score, device=self.device)
        loss.backward()

        return loss.item()

    def validation_step(self, row):
        user, pos, pos_score, _ = row
        neg, neg_score = self.dataset.get_negative(user)

        user = self.get_user_embedding(user)

        pos_pred = self.get_predictions(user, [pos])
        neg_pred = self.get_predictions(user, neg)

        loss = self.loss(pos_pred, pos_score, neg_pred, neg_score, self.device)
        return loss.item()

