import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class CatFlow(nn.Module):
    def __init__(self, model, p0, obs_dim=(2,), sigma_min=1e-10, n_samples=10):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.obs_dim = obs_dim
        self.n_samples = n_samples
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')
        self.eps_ = 1e-10
        self.softmax = nn.Softmax(-1)
        self.p0 = p0
    
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
        return (self.model(t, x, return_probs=True) - x)/(1-t.view(-1, *dims) + self.eps_)
    
    def reversed_velocity_with_div(self, t, state):
        s = 1-t
        x, logp = state
        x_ = x.detach().clone().requires_grad_(True)
        div_estimates = []
        with torch.set_grad_enabled(True):
            for i in range(self.n_samples):
                v = self.velocity(s, x_)
                is_last = (i == self.n_samples - 1)
                div_estimates.append(
                    self.approx_div(v, x_, retain_graph=not is_last)
                )
        
        mean_div = torch.stack(div_estimates).mean(dim=0)
        return ((-v).detach(), mean_div)

    def conditional_flow(self, t, x, x1):
        """
        Computes \\phi(t,x) = \\sigma(t, x1)x + \\mu(t, x1), where \\phi(0,x) = x = x0
        
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
    
    # def target_velocity(self, t, x, x1):
    #     return self.conditional_velocity(t, self.conditional_flow(t, x, x1), x1)
    
    # def criterion(self, t, x0, x1):
    #    v = self.forward(t, x0, x1)
    #    target = self.target_velocity(t, x0, x1)
    #    dim = tuple(torch.arange(1, len(x0.shape)))
    #    return torch.mean((v - target).pow(2).sum(dim=dim))

    def criterion(self, t, x0, x1):
        output = self.forward(t, x0, x1)
        return self.cross_entropy(output.transpose(1, 2), x1)/x0.shape[0]

    
    def sample(self, n_samples, method='midpoint', rtol=1e-5, atol=1e-5):
        self.device = next(self.parameters()).device
        x0 = self.p0.sample([n_samples]+list(self.obs_dim)).to(self.device)
        t = torch.linspace(0,1-self.eps_,100, device=self.device)
        with torch.no_grad():
            return odeint(self.velocity, x0, t, rtol=rtol, atol=atol, method=method, adjoint_params = self.model.parameters())[-1,:,:]
        

    def approx_div(self, f_x, x, retain_graph=True):
        z = torch.randint(low=0, high=2, size=x.shape).to(x) * 2 - 1
        e_dzdx = torch.autograd.grad(f_x, x, z, create_graph=True, retain_graph=retain_graph)[0]
        return (e_dzdx*z).view(z.shape[0], -1).sum(dim=1)

        
    def logp(self, x1, n_samples=50, rtol=1e-05, atol=1e-05):
        self.device = next(self.parameters()).device
        self.n_samples = n_samples
        t = torch.linspace(0, 1, 100, device=self.device )
        phi, f = odeint(
            self.reversed_velocity_with_div, 
            (x1, torch.zeros((x1.shape[0], 1))), 
            t, 
            rtol=rtol,
            atol=atol,
            adjoint_params = self.model.parameters(),
            )
        phi, f = phi[-1].detach().cpu(), f[-1].detach().cpu().flatten()
        logp_noise = self.p0.log_prob(phi)
        return logp_noise - f
    
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
    

class LlamaCatFlow(CatFlow):
    def __init__(self, model, vq_model, p0=None, obs_dim=(16,)):
        '''
        obs_dim: number of tokens (16 if using cifar-10 with patch size of 8)
        '''
        
        num_classes = vq_model.config.codebook_size
        if not p0:
            p0 = torch.distributions.MultivariateNormal(torch.zeros(num_classes), torch.eye(num_classes))
        super().__init__(model, p0, obs_dim)
 
        self.vq_model = vq_model

    def velocity(self, t, x, cond_idx=None):
        '''
        t: timestep
        x: x_t, current point
        '''
        t = self.process_timesteps(t, x)
        dims = [1]*(len(x.shape)-1)
        logits,_ = self.model(x, t, cond_idx)
        return (self.softmax(logits) - x)/(1-t.view(-1, *dims) + self.eps_)

    def generate(self,n_samples,top_p,top_k, method='midpoint', rtol=1e-5, atol=1e-5):
        logits = self.sample(n_samples, method=method, rtol=rtol, atol=atol)

        indices = torch.argmax(logits, dim=-1)
        block_size = self.obs_dim[0]
        h = int(block_size**0.5)
        w = h
        indices = indices.reshape(n_samples,h,w)

        decoded_img = self.vq_model.decode_code(indices,shape=(n_samples,-1,h,w))
        return decoded_img
        

    