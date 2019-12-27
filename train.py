import torch
from torch.utils.data import DataLoader

from dataset_handler import MovieLens
from losses import ewarp_loss
from nets import UserNet
from train_handler import Trainer

def get_params(model):
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    return params_to_update

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = 'ml-latest-small'
ml = MovieLens(dataset)
net = UserNet(ml.users, ml.movies.index, feature_dim=ml.feature_dim, device=device).to(device)
optimizer = torch.optim.Adam(get_params(net))
t = Trainer(net, ml, ewarp_loss, optimizer, 'small', device=device)
t.fit(300)
