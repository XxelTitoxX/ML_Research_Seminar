import torch


def solve_ode(velocity_fn, x0 : torch.Tensor, ts : torch.Tensor, method='euler'):
    x = x0
    n_samples = x0.shape[0]
    for i, t in enumerate(ts[:-1]):
        dt = ts[i+1] - t
        t_batch = t.expand(n_samples)
        
        if method == 'euler':
            v = velocity_fn(t_batch, x)
            x = x + dt * v
            
        elif method == 'midpoint':
            v1 = velocity_fn(t_batch, x)
            x_mid = x + (dt / 2) * v1
            t_mid = (t + dt / 2).expand(n_samples)
            v2 = velocity_fn(t_mid, x_mid)
            x = x + dt * v2
            
        elif method == 'rk4':
            v1 = velocity_fn(t_batch, x)
            v2 = velocity_fn((t + dt/2).expand(n_samples), x + dt/2 * v1)
            v3 = velocity_fn((t + dt/2).expand(n_samples), x + dt/2 * v2)
            v4 = velocity_fn((t + dt).expand(n_samples), x + dt * v3)
            x = x + (dt / 6) * (v1 + 2*v2 + 2*v3 + v4)
    
    return x