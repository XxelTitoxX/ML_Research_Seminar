import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from src.models.fm_wrapper import FlowMatching
from src.models.base_models import MLP

from .utils import *



def main(d=2, device='cpu'):

    device = torch.device(device)

    
    d = d

    pi1 = 0.4
    pi2 = 0.6

    # Means
    mu1 = np.array([0.0, 0.0, 0.0])[:d]
    mu2 = np.array([5.0, -10.0, 1.0])[:d]

    # Covariances
    Sigma1 = np.array([[0.5, 0.1, 0.0],
                    [0.1, 0.3, 0.0],
                    [0.0, 0.0, 0.2]])[:d, :d]

    Sigma2 = np.array([[1, 0.2, 0.1],
                    [0.2, 1.0, 0.3],
                    [0.1, 0.3, 1]])[:d, :d]


    weights = [pi1, pi2]
    means = [mu1, mu2]
    covs = [Sigma1, Sigma2]

    

    train_data_length = 1024
    train_data = torch.from_numpy(mixture_samples(train_data_length, d, weights, means, covs)).to(torch.float)

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    n_epochs = 500
    lr = 0.001
    obs_dim = (d,)
    fm = FlowMatching(MLP(d, 128, 5), obs_dim).to(device)
    optimizer = torch.optim.AdamW(fm.parameters(), lr=lr)

    total_loss = []
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        for i, x1 in enumerate(train_loader):
            x1 = x1.to(device)
            t = torch.rand(x1.shape[0]).to(device)
            x0 = torch.randn_like(x1).to(device)
            loss = fm.criterion(t, x0, x1)
            epoch_loss += loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss.append(epoch_loss/(i+1))
        if epoch %100 == 0:
            print(f"Total Loss Epoch {epoch+1}: ", total_loss[-1])

    fm = fm.to(device if torch.cuda.is_available() else 'cpu')
    samples_fm = fm.sample(train_data_length)

    df = pd.DataFrame(train_data)
    df['sample'] = 'ground truth'
    df_fm = pd.DataFrame(samples_fm,)
    df_fm['sample'] = 'flow matching'
    title = "Comparison: True Data vs Flow Matching"
    if d > 2:
        fig = px.scatter_3d(pd.concat([df, df_fm], axis=0), x=0, y=1, z=2,
                            color='sample',
                            size_max=1, opacity=1,
                            title=title)
    else:
        fig = px.scatter(pd.concat([df, df_fm], axis=0), x=0, y=1,
                            color='sample',
                            size_max=5, opacity=1,
                            title=title)
    fig.update_traces(marker_size=2)
    fig.show()



    # x1 = train_data[:256].to(device if torch.cuda.is_available() else 'cpu')
    # logp = fm.logp(x1, n_samples=50, atol=1e-4, rtol=1e-5)
    # logp_true = compute_gmm_logp(x1, weights, means, covs)

    # sns.histplot(abs(logp.numpy() - logp_true))
    # plt.title('Absolute error in logp')
    # plt.show()

    # if d==2:
        # plot_density_interactive(fm, weights, means, covs, device=device if torch.cuda.is_available() else 'cpu', range_lim=15, n_grid=32)

if __name__ == '__main__':
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    device = str(sys.argv[2]) if len(sys.argv) > 2 else 'cpu'
    main(d, device)
