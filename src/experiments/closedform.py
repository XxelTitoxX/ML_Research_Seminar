import torch

def get_lambda_block(t, xt, x1prime, prior_log_density, eps=1e-6):
    """
    t:       (Bx, 1) or (Bx,)
    xt:      (Bx, 256, 32)
    x1prime: (Bp, 256, 32)

    returns:
        (Bx, Bp)
    """
    t = t.reshape(-1)                       # (Bx,)
    xt = xt[:, None, :, :]                  # (Bx, 1, 256, 32)
    x1prime = x1prime[None, :, :, :]        # (1, Bp, 256, 32)

    x0 = (xt - t[:, None, None, None] * x1prime) / (1 - t[:, None, None, None] + eps)  # (Bx, Bp, 256, 32)
    logp0 = prior_log_density(x0)           # should return (Bx, Bp)
    return logp0


def u_star(t, xt, x1prime_loader, prior_log_density, eps=1e-6):
    """
    Parameters:
    t: batch_size * 1
    xt: batch_size * size_img
    x1prime_loader: iterable that yields batches of x1prime (batch_size_mean * size_img)
    prior_log_density: function that computes log-density of the prior at given x0

    Returns:
        torch array: (batch_size * size_img)
    -------
    """
    assert isinstance(x1prime_loader.sampler, torch.utils.data.SequentialSampler), "x1prime_loader must be a sequential dataloader for correct batching"

    all_priors = []
    processed = 0
    for x1prime in x1prime_loader:
        lambda_batch = get_lambda_block(t, xt, x1prime.to(xt.device), prior_log_density, eps) # (Bx, Bp)
        all_priors.append(lambda_batch)
        processed += lambda_batch.shape[1]
    all_priors = torch.cat(all_priors, dim=1)  # (Bx, total_Bp)
    all_lambdas = torch.softmax(all_priors, dim=1)  # (Bx, Loaderx1prime)
    u_tot = torch.zeros_like(xt)
    processed = 0
    for x1prime in x1prime_loader:
        x1prime = x1prime.to(xt.device)
        batch_size_mean = x1prime.shape[0]
        lambda_batch = all_lambdas[:, processed:processed + batch_size_mean]  # (Bx, batch_size_mean)
        lambda_batch = lambda_batch[:, :, None, None]  # (Bx, batch_size_mean, 1, 1)
        ucond = (x1prime[None, :, :, :] - xt[:, None, :, :]) / (1 - t[:, None, None] + eps)  # (Bx, batch_size_mean, 256, 32)
        u_tot += (lambda_batch * ucond).sum(dim=1)  # sum over batch_size_mean
        processed += batch_size_mean
    return u_tot


def u_star_with_labels(t, xt, x1prime_loader, labels, prior_log_density, eps=1e-6):
    """
    Parameters:
    t: batch_size * 1
    xt: batch_size * size_img
    x1prime_loader: iterable that yields batches of (x1prime, label) where x1prime is (batch_size_mean * size_img) and label is (batch_size_mean,)
    prior_log_density: function that computes log-density of the prior at given x0
    labels: (batch_size,) tensor of labels corresponding to each sample in xt

    Returns:
        torch array: (batch_size * size_img)
    -------
    """
    assert isinstance(x1prime_loader.sampler, torch.utils.data.SequentialSampler), "x1prime_loader must be a sequential dataloader for correct batching"

    all_priors = []
    processed = 0
    for x1prime_batch, label_batch in x1prime_loader:
        label_batch = label_batch.to(labels.device)  # Ensure labels are on the same device
        lambda_batch = get_lambda_block(t, xt, x1prime_batch.to(xt.device), prior_log_density, eps) # (Bx, Bp)
        # Assuming label_batch is of shape (batch_size_mean,) and labels is of shape (batch_size,)
        # We need to create a mask to zero out contributions from mismatched labels
        label_mask = (labels[:, None] == label_batch[None, :]).float()  # (Bx, batch_size_mean)
        lambda_batch = lambda_batch * label_mask  # Zero out contributions from mismatched labels
        all_priors.append(lambda_batch)
        processed += lambda_batch.shape[1]
    all_priors = torch.cat(all_priors, dim=1)  # (Bx, total_Bp)
    all_lambdas = torch.softmax(all_priors, dim=1)  # (Bx, Loaderx1prime)
    u_tot = torch.zeros_like(xt)
    processed = 0
    for x1prime_batch, label_batch in x1prime_loader:
        x1prime_batch = x1prime_batch.to(xt.device)
        batch_size_mean = x1prime_batch.shape[0]
        lambda_batch = all_lambdas[:, processed:processed + batch_size_mean]  # (Bx, batch_size_mean)
        lambda_batch = lambda_batch[:, :, None, None]  # (Bx, batch_size_mean, 1, 1)
        ucond = (x1prime_batch[None, :, :, :] - xt[:, None, :, :]) / (1 - t[:, None, None] + eps)  # (Bx, batch_size_mean, 256, 32)
        u_tot += (lambda_batch * ucond).sum(dim=1)  # sum over batch_size_mean
        processed += batch_size_mean
    return u_tot

    
    
