from copy import deepcopy
from typing import Sequence

from torch.utils.data import DataLoader
from torch import tensor

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

def get_device(device=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    elif isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    else:
        device = torch.device(device)
    return device


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
        self.best_loss = np.inf
        self.loss = loss
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.logger = Log(run_name)
        self.device = get_device(device)
        self.train, self.test = self.train_test_split(test_size)
        self.test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=True)
        self.train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)

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

        return train, test

    def score(self, k=10):
        """
        Calculate mean NDCG for users in test
        """
        score = []
        for userId, df in self.test.positive.groupby('userId'):
            not_watched = tensor(self.train.not_liked_movies(userId), device=self.device)
            order = self.predict(userId, not_watched).argsort(descending=True)
            top = torch.take(not_watched, order).cpu().numpy()
            gain = df.set_index('movieId').loc[top, 'rating'].fillna(0)
            best = df.sort_values('rating')['rating']
            score.append(ndcg_score(best, gain, k=k))
        return np.mean(score)

    @property
    def state(self):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state

    def fit(self, num_epochs, k=10):
        num_train_batches = len(self.train) / self.batch_size
        num_test_batches = len(self.test) / self.batch_size
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
                    loss /= num_train_batches
                    self.logger.metrics(loss, 0, epoch, phase)
                else:
                    with torch.no_grad():
                        for batch in self.test_loader:
                            cur_loss = self.validation_step(batch)
                            loss += cur_loss
                        loss /= num_test_batches
                        self.logger.metrics(loss, self.score(k=k), epoch, phase)

                        if loss < self.best_loss:
                            self.best_loss = loss
                            self.logger.save(self.state, epoch)

    def user_embeddings(self, users: tensor):
        """Get embeddings for every user without extra calculations for same users"""
        unique_users = np.unique(users)
        embeddings = dict()
        for user in unique_users:
            embeddings[user] = self.get_user_embedding(user)
        res = [embeddings[user] for user in users.numpy()]
        return torch.stack(res)

    def get_user_embedding(self, user: int):
        """Run UserNet to get embedding for <user>"""
        items = self.dataset.get_positive(user)
        feats = self.dataset.get_features(items)
        items = tensor(items, device=self.device)
        feats = tensor(feats, device=self.device)
        user = tensor(user, device=self.device)
        user = self.model(user, items, feats)
        return user

    def get_predictions(self,
                        user: tensor,
                        items: Sequence[int]):
        items = items.to(self.device)
        item_embeddings = self.model.item_embedding(items)
        prediction = self.model.score(user, item_embeddings)
        return prediction

    def predict(self,
                user_id: int,
                item_ids: Sequence[int]):
        user = self.get_user_embedding(user_id)
        pred = self.get_predictions(user, item_ids)
        return pred

    def training_step(self, batch):
        user, pos, pos_score, neg, neg_score = batch

        user = self.user_embeddings(user)
        pos_pred = self.get_predictions(user, pos)
        neg_pred = self.get_predictions(user, neg)

        pos_score = pos_score.float().to(self.device)
        neg_score = neg_score.float().to(self.device)
        loss = self.loss(pos_pred, pos_score, neg_pred, neg_score, device=self.device)
        loss.backward()
        return loss.item()

    def validation_step(self, batch):
        user, pos, pos_score, neg, neg_score = batch

        user = self.user_embeddings(user)
        pos_pred = self.get_predictions(user, pos)
        neg_pred = self.get_predictions(user, neg)

        pos_score = pos_score.float().to(self.device)
        neg_score = neg_score.float().to(self.device)
        loss = self.loss(pos_pred, pos_score, neg_pred, neg_score, device=self.device)
        return loss.item()

