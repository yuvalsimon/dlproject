import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn
import sklearn.cluster
from math import ceil

def cluster(data, k, temp, num_iter, init = None, cluster_temp=5):
    '''
    pytorch (differentiable) implementation of soft k-means clustering.
    '''
    #normalize x so it lies on the unit sphere
    data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
    #use kmeans++ initialization if nothing is provided
    if init is None:
        data_np = data.detach().numpy()
        norm = (data_np**2).sum(axis=1)
        init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
        init = torch.tensor(init, requires_grad=True)
        if num_iter == 0: return init
    mu = init.to('cuda') # TODO: change to some variable
    n = data.shape[0]
    d = data.shape[1]
#    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
    for t in range(num_iter):
        #get distances between all data points and cluster centers
#        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
        dist = data @ mu.t()
        #cluster responsibilities via softmax
        r = torch.softmax(cluster_temp*dist, 1)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp*dist, 1)
    return mu, r, dist

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, norm_eps=0.001, n_slope=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, norm_eps),
            nn.LeakyReLU(n_slope),
            nn.Dropout(dropout),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, norm_eps),
            nn.LeakyReLU(n_slope),
            nn.Dropout(dropout),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels, norm_eps),
            nn.LeakyReLU(n_slope),
            nn.Dropout(dropout),
        )
        self.pool_factor = 8

    def forward(self, x):
        return self.encoder(x)

class KMeansClassifier(nn.Module):
    def __init__(self, in_dims, out_channels, K, cluster_temp):
        super().__init__()
        if(len(in_dims) == 2):
            in_channels = 1
            in_x, in_y = in_dims
        else:
            in_x, in_y, in_channels = in_dims
        self.encoder = Encoder(in_channels, out_channels)
        self.encoder_features_num = int(ceil(in_x / self.encoder.pool_factor)
            * ceil(in_y / self.encoder.pool_factor) * out_channels)
        self.K = K
        self.cluster_temp = cluster_temp
        self.init =  torch.rand(self.K, self.encoder_features_num)

    def forward(self, x, num_iter=1):
        embeds = self.encoder(x)
        embeds = embeds.view(-1, self.encoder_features_num)
        mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = self.cluster_temp, init = self.init)
        mu, r, dist = cluster(embeds, self.K, 1, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
        return r
    
class LinearClassifier(nn.Module):
    def __init__(self, in_dims, out_channels, K):
        super().__init__()
        if(len(in_dims) == 2):
            in_channels = 1
            in_x, in_y = in_dims
        else:
            in_x, in_y, in_channels = in_dims
        self.encoder = Encoder(in_channels, out_channels)
        self.encoder_features_num = int(ceil(in_x / self.encoder.pool_factor)
            * ceil(in_y / self.encoder.pool_factor) * out_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_features_num, 256),
            nn.ReLU(),
            nn.Linear(256, K),
            nn.Sigmoid()
        )

    def forward(self, x):
        embeds = self.encoder(x)
        embeds = embeds.view(-1, self.encoder_features_num)
        return self.classifier(embeds)