import torch.nn as nn
import torch.nn.functional as F
# from pygcn.layers import GraphConvolution
import torch
import sklearn
import sklearn.cluster
    
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

# class GCNClusterNet(nn.Module):
#     '''
#     The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
#     The output is the cluster means mu and soft assignments r, along with the 
#     embeddings and the the node similarities (just output for debugging purposes).
    
#     The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
#     adjacency matrix. The optional parameter num_iter determines how many steps to 
#     run the k-means updates for.
#     '''
#     def __init__(self, nfeat, nout, dropout, K, cluster_temp):
#         super(GCNClusterNet, self).__init__()

#         self.feature_extractor = nn.Sequential(
#             nn.Linear(nfeat, nout),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#         self.distmult = nn.Parameter(torch.rand(nout))
#         self.sigmoid = nn.Sigmoid()
#         self.K = K
#         self.cluster_temp = cluster_temp
#         self.init =  torch.rand(self.K, nout)
        
#     def forward(self, x, num_iter=1):
#         embeds = self.feature_extractor(x)
#         mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = self.cluster_temp, init = self.init)
#         mu, r, dist = cluster(embeds, self.K, 1, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
#         return mu, r, dist

class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, K, cluster_temp):
        super().__init__()
        dropout = 0.2
        norm_eps = 0.001
        negative_slope = 0.1
        modules = [
            nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128, norm_eps),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
            nn.Conv2d(128, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels, norm_eps),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
#             nn.Conv2d(256, out_channels, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm2d(out_channels, norm_eps),
#             nn.LeakyReLU(negative_slope),
#             nn.Dropout(dropout),
        ]
        self.encoder = nn.Sequential(*modules)
        self.encoder_features_num = int((28 / 4) ** 2 * out_channels)
        self.K = K
        self.cluster_temp = cluster_temp
        self.init =  torch.rand(self.K, self.encoder_features_num)

    def forward(self, x, num_iter=1):
        embeds = self.encoder(x)
        embeds = embeds.view(-1, self.encoder_features_num)
        mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = self.cluster_temp, init = self.init)
        mu, r, dist = cluster(embeds, self.K, 1, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
        return mu, r, dist