import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint 
import torch.nn.functional as F

class CatFlow(nn.Module):
    def __init__(self, model, obs_dim=(16,), n_samples=10):
        super().__init__()
        self.model = model
        self.obs_dim = obs_dim
        self.n_samples = n_samples
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.eps_ = 1e-10
        self.softmax = nn.Softmax(-1)
    
    def process_timesteps(self, t, x):
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])
        if t.shape[0] != x.shape[0] or len(t.shape) != 1:
            raise ValueError("Timesteps shape should be (batch_size, )")
        return t

    def conditional_flow(self, t, x0, x1):
        dims = [1] * (len(x0.shape) - 1)
        t = t.view(-1, *dims)
        return t * x1 + (1 - t) * x0

    def approx_div(self, f_x, x, retain_graph=True):
        z = torch.randint(low=0, high=2, size=x.shape).to(x) * 2 - 1
        e_dzdx = torch.autograd.grad(f_x, x, z, create_graph=True, retain_graph=retain_graph)[0]
        return (e_dzdx * z).view(z.shape[0], -1).sum(dim=1)

class LlamaCatFlow(CatFlow):
    def __init__(self, model, vq_model, obs_dim=(16,), temperature=0.8):
        super().__init__(model, obs_dim)
        self.temperature = temperature
        self.vq_model = vq_model
        
        if hasattr(self.vq_model, 'quantize') and hasattr(self.vq_model.quantize, 'embedding'):
            self.embed_dim = self.vq_model.quantize.embedding.weight.shape[1]
        elif hasattr(self.vq_model, 'codebook'):
            self.embed_dim = self.vq_model.codebook.shape[1]
        else:
            self.embed_dim = 8 

    def get_codebook(self):
        if hasattr(self.vq_model, 'quantize') and hasattr(self.vq_model.quantize, 'embedding'):
            return self.vq_model.quantize.embedding.weight
        elif hasattr(self.vq_model, 'codebook'):
            return self.vq_model.codebook
        else:
            raise AttributeError("Codebook not found")

    def criterion(self, t, x0, x1, x1_indices, cond_idx=None):
        t = self.process_timesteps(t, x0)
        x_t = self.conditional_flow(t, x0, x1)
        logits = self.model(t, x_t)
        return self.cross_entropy(logits.transpose(1, 2), x1_indices)

    def velocity(self, t, x, is_inference=False, cond_idx=None):
        t = self.process_timesteps(t, x)
        dims = [1] * (len(x.shape) - 1)
        
        logits = self.model(t, x)
        if is_inference:
            logits = logits / self.temperature
            
        probs = self.softmax(logits)
        codebook = self.get_codebook()
        
        mu_t = torch.matmul(probs, codebook)
        
        velocity = (mu_t - x) / (1 - t.view(-1, *dims) + self.eps_)
        return velocity

    def sample(self, n_samples, cond_idx=None, n_steps=100, method='midpoint'):
        self.device = next(self.parameters()).device
        seq_len = self.obs_dim[0]

        codebook = self.get_codebook()
        cb_std = codebook.std().item()
        x = torch.randn(n_samples, seq_len, self.embed_dim, device=self.device) 
        
        ts = torch.linspace(0, 1.0, n_steps, device=self.device)
        dt = ts[1] - ts[0]
        
        with torch.no_grad():
            for i, t in enumerate(ts[:-1]):
                t_batch = t.expand(n_samples)
                
                if method == 'euler':
                    v = self.velocity(t_batch, x, is_inference=True, cond_idx=cond_idx)
                    x = x + dt * v
                    
                elif method == 'midpoint':
                    v1 = self.velocity(t_batch, x, is_inference=True, cond_idx=cond_idx)
                    x_mid = x + (dt / 2) * v1
                    t_mid = (t + dt / 2).expand(n_samples)
                    v2 = self.velocity(t_mid, x_mid, is_inference=True, cond_idx=cond_idx)
                    x = x + dt * v2
                    
                elif method == 'rk4':
                    v1 = self.velocity(t_batch, x, is_inference=True, cond_idx=cond_idx)
                    v2 = self.velocity((t + dt/2).expand(n_samples), x + dt/2 * v1, is_inference=True, cond_idx=cond_idx)
                    v3 = self.velocity((t + dt/2).expand(n_samples), x + dt/2 * v2, is_inference=True, cond_idx=cond_idx)
                    v4 = self.velocity((t + dt).expand(n_samples), x + dt * v3, is_inference=True, cond_idx=cond_idx)
                    x = x + (dt / 6) * (v1 + 2*v2 + 2*v3 + v4)
        
        return x

    def generate(self, n_samples, cond_idx=None, method='midpoint', n_steps=100):
            x_final = self.sample(n_samples, cond_idx=cond_idx, method=method, n_steps=n_steps)
            
            codebook = self.vq_model.quantize.embedding.weight
            
            x_final_flat = x_final.reshape(-1, codebook.shape[-1])
            
            d = torch.sum(x_final_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook ** 2, dim=1) - 2 * \
                torch.matmul(x_final_flat, codebook.t())
                
            indices = torch.argmin(d, dim=1)
            
            block_size = self.obs_dim[0]
            h = int(block_size**0.5)
            w = h
            indices = indices.reshape(n_samples, h, w)

            decoded_img = self.vq_model.decode_code(indices, shape=(n_samples, -1, h, w))
            return decoded_img