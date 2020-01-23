import torch

from dataset_handler import MovieLens
from losses import ewarp_loss
from nets import UserNet
from train_handler import Trainer, get_device


dataset = 'ml-20m'
device = get_device()
ml = MovieLens(dataset)
ml.set_scope(ml.ratings.head(3000))
net = UserNet(ml.users, ml.movies.index, feature_dim=ml.feature_dim, device=device).to(device)
optimizer = torch.optim.Adam(net.params)
t = Trainer(net, ml, ewarp_loss, optimizer, '20m_100e', device=device, batch_size=420)
t.fit(5)
