import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from .ode import solve_ode
import math


def pack_state(x, f):
    b = x.shape[0]
    f = f.reshape(b, -1)
    return torch.cat([x.reshape(b, -1), f], dim=1)

def unpack_state(z, d, k):
    x_dim = d * k
    x_flat, f = z[:, :x_dim], z[:, x_dim:].contiguous()
    x = x_flat.view(z.shape[0], d, k)
    return x, f

class CatFlow(nn.Module):
    def __init__(self, model, obs_dim=(2,), loss="kld", sigma_min=0.0, n_samples=10, prior_eps=1e-4):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.obs_dim = obs_dim
        self.n_samples = n_samples
        self.prior_eps = prior_eps
        self.loss = loss # "kld" or "mse"
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')
        self.eps_ = 1e-6
        self.clamp_t: float = 0.05 # from Categorical Flow Map repository
        self.softmax = nn.Softmax(-1)

    @staticmethod
    def sample_simplex(*sizes, device="cpu", eps=1e-4):
        """Uniform sample on simplex via normalized exponential variables."""
        x = torch.empty(*sizes, device=device, dtype=torch.float).exponential_(1)
        p = x / x.sum(dim=-1, keepdim=True)
        p = p.clamp(eps, 1 - eps)
        return p / p.sum(dim=-1, keepdim=True)

    def sample_prior(self, *size, device=None, eps=None):
        if eps is None:
            eps = self.prior_eps
        if device is None:
            device = next(self.parameters()).device
        return self.sample_simplex(*size, device=device, eps=eps)

    @staticmethod
    def prior_logp0(p0, eps=1e-4):
        """
        Log-density of uniform prior on simplex.
        p0: (B1, B2, ..., D, K) tensor of probabilities (last dim sums to 1)
        """
        batch_shape = p0.shape[:-2]
        D, K = p0.shape[-2:]
        return torch.full(batch_shape, D * math.lgamma(K), device=p0.device, dtype=p0.dtype)
    
    def process_timesteps(self, t, x):
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])
        if t.shape[0]!=x.shape[0] or len(t.shape)!=1:
            raise ValueError("Timesteps shape should (batch_size, )")
        return t

    def forward(self, t, x0, x1):
        """ 
        Computes velocity v from the equation dphi(t, x) = v(t, phi(t, x))dt. 
        """
        t = self.process_timesteps(t, x0)
        x = self.conditional_flow(t, x0, x1)
        return self.model(t, x)

    def velocity(self, t, x):
        t = self.process_timesteps(t, x)
        dims = [1]*(len(x.shape)-1)
        if self.loss == "mse":
            velocity = self.model(t, x)
        elif self.loss == "kld":
            velocity = (self.model(t, x, return_probs=True) - x)/(1-t.view(-1, *dims) + self.eps_)
        return velocity
    
    def reversed_velocity_with_div(self, t, state):
        # Reverse-time integration uses model-time s = 1 - t.
        # Clamp away from s=1 where velocity has a 1/(1-s) singularity.
        s = (1 - t).clamp(max=1.0 - self.clamp_t)
        x, logp = state
        x_ = x.detach().clone().requires_grad_(True)
        div_estimates = []
        with torch.set_grad_enabled(True):
            for i in range(self.n_samples):
                v = self.velocity(s, x_)
                div_estimates.append(
                    self.approx_div(v, x_, retain_graph=False)
                )
        
        mean_div = torch.stack(div_estimates).mean(dim=0)
        print(f"Mean div: {mean_div.mean().item():.4f}, s: {s[0].item():.4f}")
        return ((-v).detach(), mean_div)

    def conditional_flow(self, t, x, x1):
        """
        Computes \phi(t,x) = \sigma(t, x1)x + \mu(t, x1), where \phi(0,x) = x = x0
        
        :param t: timestep. Float in [0,1].
        :param x0: starting point sampled from N(0, I).
        :param x1: observation
        """
        dims = [1]*(len(x.shape)-1)
        t = t.view(-1, *dims)
        values = t.expand(-1, x1.shape[1], 1)
        x_scaled = x * (1 - (1 - self.sigma_min) * t)
        return x_scaled.scatter_add(2, x1.unsqueeze(-1), values.to(x.dtype))
    
    def conditional_velocity(self, t, x, x1, eps=1e-7):
        """
        Computes (x1-x)/(1-t) =  - (x-x1)/(1-t) = - (x/(1-t) - x1/(1-t))
        """
        dims = [1]*(len(x.shape)-1)
        t = t.view(-1, *dims)
        denom = 1-t.expand(-1, x1.shape[1], 1) + eps
        values = - 1.0/denom
        x_scaled = x/denom
        return -x_scaled.scatter_add(2, x1.unsqueeze(-1), values.to(x.dtype))

    def criterion(self, t, x0, x1):
        output = self.forward(t, x0, x1) # shape: (batch_size, seq_len, codebook_size)

        if self.loss == "mse":
            target = torch.nn.functional.one_hot(x1, num_classes=x0.shape[-1]).to(x0.dtype) - x0
            dim = tuple(torch.arange(1, len(x0.shape)))
            return torch.mean((output - target).pow(2).sum(dim=dim))
        elif self.loss == "kld":
            # Normalize by number of categorical variables for token-level CE.
            return self.cross_entropy(output.transpose(1, 2), x1) / (x0.shape[0] * x0.shape[1])
        else:
            raise ValueError(f"Unknown loss type: {self.loss}")

    
    def sample(self, n_samples, method='euler', n_steps=10, rtol=1e-5, atol=1e-5):
        self.device = next(self.parameters()).device
        x0 = self.sample_prior(n_samples, *list(self.obs_dim), device=self.device)
        t = torch.linspace(0,1-self.clamp_t,n_steps, device=self.device)
        with torch.no_grad():
            # return odeint(self.velocity, x0, t, rtol=rtol, atol=atol, method=method, adjoint_params = self.model.parameters())[-1,:,:]
            return solve_ode(self.velocity, x0, t, method=method)
        

    def approx_div(self, f_x, x, retain_graph=False):
        z = torch.randint(low=0, high=2, size=x.shape).to(x) * 2 - 1
        e_dzdx = torch.autograd.grad(f_x, x, z, create_graph=False, retain_graph=retain_graph)[0]
        return (e_dzdx*z).view(z.shape[0], -1).sum(dim=1)

        
    def logp(self, x1, n_samples=50, rtol=1e-05, atol=1e-05):
        B, D, K = x1.shape
        self.device = next(self.parameters()).device
        self.n_samples = n_samples
        # Start slightly above 0 to avoid evaluating reverse model-time at s=1.
        t = torch.linspace(self.clamp_t, 1.0, 20, device=self.device)
        z0 = pack_state(x1, torch.zeros((x1.shape[0], 1), device=x1.device, dtype=x1.dtype))
        def vel_packed(t, z):
            x, f = unpack_state(z, D, K)
            dx, df = self.reversed_velocity_with_div(t, (x, f))
            return pack_state(dx, df)
        zT = solve_ode(vel_packed, z0, t, method='midpoint')
        phi, f = unpack_state(zT, D, K)
        print(f"Mean f: {f.mean().item():.4f}")
        logp_noise = self.prior_logp0(phi, eps=self.prior_eps)
        logp = logp_noise - f
        print(f"Mean logp: {logp.mean().item():.4f}")
        return logp.reshape(-1)
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.model.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class CodebookCatFlow(CatFlow):
    def __init__(self, model, vq_model, obs_dim=(16,), temperature=0.8):
        super().__init__(model, obs_dim=obs_dim)
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
        logits = self.model(t, x_t, cond_idx=cond_idx)
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = 1e-4 * torch.mean(log_z**2)
            
        return self.cross_entropy(logits.transpose(1, 2), x1_indices) + z_loss

    def velocity(self, t, x, is_inference=False):
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

    def sample(self, n_samples, n_steps=100, method='midpoint'):
        self.device = next(self.parameters()).device
        seq_len = self.obs_dim[0]

        codebook = self.get_codebook()
        cb_std = codebook.std().item()
        print(f"Sampling with observed dim: {self.obs_dim}")
        x0 = self.sample_prior(n_samples, *list(self.obs_dim), device=self.device)
        
        ts = torch.linspace(0, 1.0-self.clamp_t, n_steps, device=self.device)
        dt = ts[1] - ts[0]
        
        with torch.no_grad():
            x = solve_ode(self.velocity, x0, ts, method=method)
        
        return x

    def generate(self, n_samples, method='midpoint', n_steps=100):
            x_final = self.sample(n_samples, method=method, n_steps=n_steps)
            
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

class LlamaCatFlow(CatFlow):
    def __init__(self, model, vq_model, obs_dim=(16,), temperature=0.8):
        super().__init__(model, obs_dim=obs_dim)
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
        logits = self.model(t, x_t, cond_idx=cond_idx)
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = 1e-4 * torch.mean(log_z**2)
            
        return self.cross_entropy(logits.transpose(1, 2), x1_indices) + z_loss

    def velocity(self, t, x, is_inference=False, cond_idx=None):
        """
        Computes the velocity in codebook space.
        x : (batch_size, seq_len, embed_dim)
        """
        t = self.process_timesteps(t, x)
        dims = [1] * (len(x.shape) - 1)
        
        logits = self.model(t, x, cond_idx=cond_idx)
        if is_inference:
            logits = logits / self.temperature
            
        probs = self.softmax(logits)
        codebook = self.get_codebook()
        
        mu_t = torch.matmul(probs, codebook)
        
        velocity = (mu_t - x) / (1 - t.view(-1, *dims) + self.eps_)
        return velocity
    
    def prob_velocity(self, t, x, is_inference=False, cond_idx=None):
        """
        Computes the velocity in probability space.
        x : (batch_size, seq_len, codebook_size)
        """
        t = self.process_timesteps(t, x)
        dims = [1] * (len(x.shape) - 1)

        codebook_vec_x = torch.matmul(x, self.get_codebook())
        
        logits = self.model(t, codebook_vec_x, cond_idx=cond_idx)
        if is_inference:
            logits = logits / self.temperature
            
        probs = self.softmax(logits)
        
        velocity = (probs - x) / (1 - t.view(-1, *dims) + self.eps_)
        return velocity
    
    def sample_gauss_prior(self, n_samples, device):
        return torch.randn(n_samples, self.obs_dim[0], self.embed_dim, device=device)
    
    def gauss_prior_log_density(self, x):
        # x shape: (Bx, Bp, seq_len, codebook_dim)
        assert (x.shape[-1] == self.embed_dim and x.shape[-2] == self.obs_dim[0]), "Input shape does not match expected dimensions"
        D = x.shape[-1] * x.shape[-2]
        return -0.5 * torch.sum(x**2, dim=(-1, -2)) - 0.5 * D * math.log(2 * math.pi)

    def sample(self, n_samples, cond_idx=None, n_steps=100, method='midpoint'):
        self.device = next(self.parameters()).device

        x = self.sample_gauss_prior(n_samples, device=self.device)
        
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
            print(f"Observed dim: {self.obs_dim}")
            x_final = self.sample(n_samples, cond_idx=cond_idx, method=method, n_steps=n_steps)
            print(f"Sampled latent shape: {x_final.shape}")
            
            codebook = self.vq_model.quantize.embedding.weight
            
            x_final_flat = x_final.reshape(-1, codebook.shape[-1])
            print(f"Codebook shape: {codebook.shape}, x_final_flat shape: {x_final_flat.shape}")
            
            d = torch.sum(x_final_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook ** 2, dim=1) - 2 * \
                torch.matmul(x_final_flat, codebook.t())
                
            indices = torch.argmin(d, dim=1)
            print(f"Indices shape: {indices.shape}")
            
            block_size = self.obs_dim[0]
            h = int(block_size**0.5)
            w = h
            indices = indices.reshape(n_samples, h, w)
            print(f"Reshaped indices shape: {indices.shape}")

            decoded_img = self.vq_model.decode_code(indices, shape=(n_samples, -1, h, w))
            return decoded_img