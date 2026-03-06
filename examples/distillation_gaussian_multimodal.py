import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import mixture_samples
from src.models.base_models import MLP
import src.distillation.scfm as scfm
import matplotlib.pyplot as plt
import sys

def main(teacher_net_path, method="vanilla", d=2, device="cpu"):

    pi1 = 0.4
    pi2 = 0.6

    mu1 = np.array([0.0, 0.0, 0.0])[:d]
    mu2 = np.array([5.0, -10.0, 1.0])[:d]

    Sigma1 = np.array([[0.5, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.0, 0.2]])[:d, :d]

    Sigma2 = np.array([[1, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1]])[:d, :d]

    weights = [pi1, pi2]
    means = [mu1, mu2]
    covs = [Sigma1, Sigma2]

    train_data_length = 1024
    train_data = torch.from_numpy(
        mixture_samples(train_data_length, d, weights, means, covs)
    ).to(torch.float)

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    teacher_net = scfm.SCFMWrapper(
        torch.load(teacher_net_path, weights_only=False), "reverse"
    ).to(device)
    student_net = scfm.SCFMWrapper(MLP(d, 128, 5), "reverse").to(device)
    slow_ema_net = torch.optim.swa_utils.AveragedModel(
        student_net,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
        device=device,
    )
    if method == "dual":
        fast_ema_net = torch.optim.swa_utils.AveragedModel(
            student_net,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99),
            device=device,
        )
        fast_ema_net.eval()
    teacher_net.eval()
    student_net.train()
    slow_ema_net.eval()
    optimizer = torch.optim.AdamW(student_net.parameters(), lr=5e-4)
    n_steps = 4

    if method == "vanilla":
        scfm.vanilla_scfm(
            teacher_net,
            student_net,
            slow_ema_net,
            None,
            train_loader,
            500,
            batch_size,
            n_steps,
            0.4,
            (1.0, 2.0),
            device,
            optimizer,
            False,
            0,
            None,
        )
    else:
        scfm.dual_ema_scfm(
            teacher_net,
            student_net,
            slow_ema_net,
            fast_ema_net,
            None,
            train_loader,
            500,
            batch_size,
            n_steps,
            0.4,
            (1.0, 2.0),
            device,
            optimizer,
            None,
        )

    student_net.eval()
    x_t = scfm.sample(student_net, train_data_length, d, n_steps, 1.5, device)

    train_data = train_data.detach().cpu().numpy()
    x_t = x_t.detach().cpu().numpy()

    plt.scatter(train_data[:, 0], train_data[:, 1])
    plt.scatter(x_t[:, 0], x_t[:, 1])
    plt.show()


if __name__ == "__main__":
    teacher_net_path = str(sys.argv[1])
    method = str(sys.argv[1]) if len(sys.argv) > 2 else "vanilla"
    d = int(sys.argv[2]) if len(sys.argv) > 3 else 2
    device = str(sys.argv[3]) if len(sys.argv) > 4 else "cpu"
    main(teacher_net_path, d, device)
