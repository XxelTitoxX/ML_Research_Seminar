import torch 

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

def plot_density_interactive(fm, weights, means, covs, device='cpu', range_lim=15, n_grid=60, rtol=1e-4, atol=1e-4):
        x = np.linspace(-range_lim, range_lim, n_grid)
        y = np.linspace(-range_lim, range_lim, n_grid)
        X, Y = np.meshgrid(x, y)
        grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            log_probs = fm.logp(grid_points_tensor, rtol=rtol, atol=atol) 
            Z = torch.exp(log_probs).cpu().numpy().reshape(X.shape)
            
        log_probs_true = compute_gmm_logp(grid_points, weights, means, covs)
        Z_true = np.exp(log_probs_true).reshape(X.shape)

        fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.1, 
            specs=[[{'type': 'contour'}], [{'type': 'surface'}]],
            subplot_titles=("2D Interactive Contour (FM vs True GMM)", "3D Density Surface (Flow Matching)")
        )

        fig.add_trace(go.Contour(
            z=Z, x=x, y=y, 
            colorscale='Viridis',
            name='Flow Matching',
            contours_coloring='heatmap',
            colorbar=dict(len=0.4, y=0.8, title="Prob")
        ), row=1, col=1)

        fig.add_trace(go.Contour(
            z=Z_true, x=x, y=y,
            contours=dict(coloring='none', showlabels=False),
            line=dict(color='white', width=1.5, dash='dash'),
            name='True GMM'
        ), row=1, col=1)

        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y, 
            colorscale='Viridis',
            showscale=False,
            name='FM Surface'
        ), row=2, col=1)

        fig.update_layout(
            title=f"Flow Matching Density Analysis (range={range_lim})",
            width=900, 
            height=1000, 
            template='none',
            paper_bgcolor='rgba(245,245,245,1)', 
            plot_bgcolor='rgba(245,245,245,1)'   
        )
        
        fig.update_scenes(
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            xaxis_title='X1', yaxis_title='X2', zaxis_title='Density'
        )

        fig.show()

def mixture_samples(n_samples, d, weights, means, covs):

    choices = np.random.rand(n_samples) < weights[0]  # True = Gaussian 1, False = Gaussian 2
    samples = np.zeros((n_samples, d))

    samples[choices] = np.random.multivariate_normal(means[0], covs[0], size=np.sum(choices))
    samples[~choices] = np.random.multivariate_normal(means[1], covs[1], size=np.sum(~choices))

    return samples

def compute_gmm_logp(x, weights, means, covs):

    log_probs = []
    for i in range(len(weights)):
        lp = multivariate_normal.logpdf(x, mean=means[i], cov=covs[i])
        log_probs.append(np.log(weights[i]) + lp)
    
    return logsumexp(np.array(log_probs), axis=0)
