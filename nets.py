import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    """
    Process auxiliary item features into latent space.
    All items for user can be processed in batch.
    """
    def __init__(self, emb_dim, feature_dim):
        super(FeatureNet, self).__init__()

        self.l1 = nn.Linear(emb_dim + feature_dim, emb_dim)
        self.l2 = nn.Linear(emb_dim, emb_dim)

        self._kaiming_(self.l1)
        self._kaiming_(self.l2)

    def _kaiming_(self, layer):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(layer.bias)

    def forward(self, user, components):

        if len(components.shape) > 1:
            user = user.expand(components.shape[0], -1)

        x = torch.cat([user, components], dim=-1)

        x = self.l1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = F.relu(x)

        return x


class UserNet(nn.Module):
    """
    Get user embedding accounting to surpassed items
    """

    def __init__(self, users, items, emb_dim=128, feature_dim=0, device=None):
        super(UserNet, self).__init__()

        self.emb_dim = emb_dim
        num_users = max(users) + 1
        num_items = max(items) + 1

        self.feats = FeatureNet(emb_dim, feature_dim) if feature_dim > 0 else None

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        f = 1 if self.feats is not None else 0
        self.l1 = nn.Linear((2 + f) * emb_dim, emb_dim)
        self.l2 = nn.Linear(emb_dim, 1)

        self._kaiming_(self.l1)
        self._kaiming_(self.l2)

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

    def _kaiming_(self, layer):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(layer.bias)

    def _get_user(self, user_id):
        return torch.tensor(self.user_dict[user_id], device=self.device)

    def forward(self, user_ids, item_ids, features=None):

        user = self.user_embedding(user_ids)
        items = self.item_embedding(item_ids)

        if self.feats is not None:
            components = self.feats(user, torch.tensor(features, dtype=torch.float32, device=self.device))
        else:
            components = torch.tensor([], device=self.device)

        usern = user.expand(items.shape[0], -1)

        x = torch.cat([usern, items, components], dim=-1)

        x = self.l1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = F.softmax(x, 0)

        x = user + (items * x).sum(0)

        return x

    def score(self, user, items):
        return (user * items).sum(1) / self.emb_dim

