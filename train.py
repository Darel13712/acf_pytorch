import torch

from dataset_handler import MovieLens
from losses import ewarp_loss
from nets import UserNet
from train_handler import Trainer, get_device


dataset = 'ml-20m'
device = get_device()
ml = MovieLens(dataset)
net = UserNet(ml.users, ml.movies.index, feature_dim=ml.feature_dim, device=device).to(device)
optimizer = torch.optim.Adam(net.params)
t = Trainer(net, ml, ewarp_loss, optimizer, '20m', device=device)
t.fit(300)
