import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
import sys
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, DBSCAN
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics.pairwise import cosine_similarity

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # define: encoder
        self.encoder = nn.Sequential(
          nn.Conv2d(3, 64, 3, 1, 1),
          nn.ReLU(True),
          nn.BatchNorm2d(64), 
          nn.MaxPool2d(2,2),
          nn.Conv2d(64, 128, 3, 1, 1),
          nn.ReLU(True),
          nn.BatchNorm2d(128),
          nn.MaxPool2d(2,2),
          nn.Conv2d(128, 256, 3, 1, 1),
          nn.ReLU(True),
          nn.BatchNorm2d(256),
          nn.MaxPool2d(2,2),
        )

        # define: decoder
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(256, 128, 2, 2),
          nn.ReLU(True),
          nn.ConvTranspose2d(128, 64, 2, 2),
          nn.ReLU(True),
          nn.ConvTranspose2d(64, 3, 2, 2),
          nn.Tanh(),
        )

    def forward(self, x):
        # (3, 32, 32)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Total AE: return latent & reconstruct
        return encoded, decoded

if __name__ == '__main__':

    # detect is gpu available.
    use_gpu = torch.cuda.is_available()

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('./model.npy'))
    
    # load data and normalize to [-1, 1]
    trainX = np.load(sys.argv[1])
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)

    # if use_gpu, send model / data to GPU.
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()

    # Dataloader: train shuffle = True
    train_dataloader = DataLoader(trainX, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(trainX, batch_size=32, shuffle=False)


    # We set criterion : L1 loss (or Mean Absolute Error, MAE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)


    # Collect the latents and stdardize it.
    latents = []
    reconstructs = []
    for x in test_dataloader:

        latent, reconstruct = autoencoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())

    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)


    # Use PCA to lower dim of latents and use K-means to clustering.
    latents = PCA(n_components=32, whiten=True).fit_transform(latents)
    latents = TSNE(n_components=2, perplexity=50, init='pca', verbose=1).fit_transform(latents)
    # latents = TSNE(n_components=8, perplexity=50, method='exact').fit_transform(latents)
    
    result = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels="discretize").fit(latents).labels_

    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
    
    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(sys.argv[2],index=False)
