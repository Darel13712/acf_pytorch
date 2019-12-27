from copy import deepcopy

from torch.utils.data import DataLoader

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
    """
    Handles training process
    """
    def __init__(self, model, dataset, loss, optimizer, run_name, batch_size=100, device=None, test_size=10):
        """
        Parameters
        ----------
        model: initialized UserNet
        dataset: initialized MovieLens
        loss: one of the warp functions
        optimizer: torch.optim
        run_name: directory to save results
        batch_size: number of samples to process for one update
        device: gpu or cpu
        test_size: number of tail items for each user to leave for test
        """
        self.loss = loss
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.logger = Log(run_name)
        self.best_loss = float("inf")
        self.device = self.get_device(device)
        self.train_test_split(test_size, batch_size)

    @staticmethod
    def get_device(device):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return device

    def train_test_split(self, num=10, batch_size=100):
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

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True) #todo move dataloaders and self.= to init
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

        self.train = train
        self.test = test

        self.train_loader = train_loader
        self.test_loader = test_loader

    @staticmethod
    def sort(items, score):
        order = np.argsort(score.cpu().numpy())[::-1]
        top = np.take(items.numpy(), order)
        return top

    def score(self):
        ratings = self.train.ratings
        movies = self.dataset.movies.index
        score = []

        for userId, df in self.test.positive.groupby('userId'):
            watched_movies = ratings.loc[ratings.userId == userId]
            unseen = movies[~movies.isin(watched_movies.movieId)]
            unseen = torch.tensor(unseen, device=self.device)
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
                    for batch in self.train_loader:
                        self.optimizer.zero_grad()
                        cur_loss  = self.training_step(batch)
                        self.optimizer.step()
                        loss += cur_loss

                    loss /= len(self.train)
                    self.logger.metrics(loss, 0, epoch, phase)
                else:
                    with torch.no_grad():
                        for batch in self.test_loader:
                            cur_loss = self.validation_step(batch)
                            loss += cur_loss

                        loss /= len(self.test)
                        score = self.score()
                        self.logger.metrics(loss, score, epoch, phase)

                        if loss < self.best_loss:
                            self.best_loss = loss
                            self.logger.save(self.state, epoch)

    def user_embeddings(self, users):
        unique_users = np.unique(users)
        embeddings = dict()
        for user in unique_users:
            embeddings[user] = self.get_user_embedding(user)
        res = [embeddings[user] for user in users.numpy()]
        return torch.stack(res)

    def get_user_embedding(self, user):
        items = self.dataset.get_positive(user)
        feats = self.dataset.get_features(items)
        items = torch.tensor(items, device=self.device)
        feats = torch.tensor(feats, device=self.device)
        user = torch.tensor(user, device=self.device)
        user = self.model(user, items, feats)
        return user


    def get_predictions(self, user, items):
        item_embeddings = self.model.item_embedding(items)
        prediction = self.model.score(user, item_embeddings)
        return prediction


    def training_step(self, batch):
        user, pos, pos_score, neg, neg_score = batch

        user = self.user_embeddings(user)
        pos_pred = self.get_predictions(user, pos)
        neg_pred = self.get_predictions(user, neg)

        pos_score = pos_score.float()
        neg_score = neg_score.float()
        loss = self.loss(pos_pred, pos_score, neg_pred, neg_score, device=self.device)
        loss.backward()
        return loss.item()

    def validation_step(self, batch):
        user, pos, pos_score, neg, neg_score = batch

        user = self.user_embeddings(user)
        pos_pred = self.get_predictions(user, pos)
        neg_pred = self.get_predictions(user, neg)

        pos_score = pos_score.float()
        neg_score = neg_score.float()
        loss = self.loss(pos_pred, pos_score, neg_pred, neg_score, device=self.device)
        return loss.item()

