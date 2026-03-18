import torch
import wandb
import math
import datasets
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
EPS = 1e-8

from models.llama_models import GPT_B
from models.vq_model import VQ_Cifar_L
from models.vfm_wrapper import LlamaCatFlow


def load_cifar10(batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = dict(
        vocab_size=512,
        num_classes=10,
        block_size=256,
        batch_size=64,
        n_epochs=200,
        sample_every_steps=500,
        save_every_epochs=5,
        lr=1e-4,
        grad_clip=5.0,
        slow_mu=0.999,
        fast_mu=0.99,
        s_range=[1.0, 1.5],
        nb_teacher_steps=8,
        ratio_teacher_samples=0.4,
        resume_checkpoint=None,
    )

    wandb.init(project="distillation-catflow-cifar10", config=config)

    # 1. Load VQ-VAE and Codebook
    vq_model = VQ_Cifar_L().to(device)
    vq_model.load_state_dict(
        torch.load("checkpoints/vq_cifar_epoch_20.pt", map_location=device)[
            "model_state_dict"
        ]
    )
    vq_model.eval()

    # 2. Setup Models
    # Teacher (Frozen)
    teacher_model = GPT_B(
        vocab_size=config["vocab_size"],
        num_classes=config["num_classes"],
        block_size=config["block_size"],
    ).to(device)
    teacher_model.load_state_dict(
        torch.load("checkpoints/llamagen_epoch_250.pt", map_location=device)[
            "model_state_dict"
        ]
    )
    teacher_model.eval()

    # Student (Trainable)
    student_model = GPT_B(
        vocab_size=config["vocab_size"],
        num_classes=config["num_classes"],
        block_size=config["block_size"],
    ).to(device)
    student_model.load_state_dict(
        torch.load("checkpoints/llamagen_epoch_250.pt", map_location=device)[
            "model_state_dict"
        ]
    )
    student_model.train()

    # 3. Initialize VFM Wrapper
    # We use the student model as the primary model for the wrapper
    vfm = LlamaCatFlow(student_model, vq_model, obs_dim=(config["block_size"],))
    codebook = vfm.get_codebook()

    # 4. Setup EMA Models
    slow_ema_model = torch.optim.swa_utils.AveragedModel(
        student_model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config["slow_mu"]),
        device=device,
    )
    fast_ema_model = torch.optim.swa_utils.AveragedModel(
        student_model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config["fast_mu"]),
        device=device,
    )

    opt = torch.optim.AdamW(
        student_model.parameters(), lr=config["lr"], weight_decay=0.05
    )
    dataloader = load_cifar10(config["batch_size"])
    timesteps = torch.linspace(0, 1, config["nb_teacher_steps"] + 1, device=device)

    # Functional helper to get probs and velocity using the wrapper's logic but varying the model
    def get_probs_vel_logits(model_instance, t, x, cond):
        # We temporarily point vfm.model to the instance we want to query
        orig_model = vfm.model
        vfm.model = model_instance

        t_proc = vfm.process_timesteps(t, x)
        logits = vfm.model(t_proc, x, cond_idx=cond)
        probs = F.softmax(logits, dim=-1)
        mu_t = torch.matmul(probs, codebook)
        vel = (mu_t - x) / (1 - t_proc.view(-1, 1, 1) + vfm.eps_)

        vfm.model = orig_model  # Restore student
        return probs, vel, logits

    global_step = 0
    for epoch in range(1, config["n_epochs"] + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config["n_epochs"]}")

        for images, cond_idx in pbar:
            images, cond_idx = images.to(device), cond_idx.to(device)

            with torch.no_grad():
                _, _, (_, _, idx) = vq_model.encode(images)
                x_1_indices = idx.view(images.shape[0], -1)
                x_1 = F.embedding(x_1_indices, codebook)

            x_0 = torch.randn_like(x_1).to(device)
            s = torch.empty(config["batch_size"], device=device).uniform_(
                *config["s_range"]
            )

            # --- SCFM Indices Logic ---
            nb_teacher = int(config["ratio_teacher_samples"] * config["batch_size"])
            nb_student = config["batch_size"] - nb_teacher

            # Part A: Teacher
            j_T = torch.randint(
                0, config["nb_teacher_steps"] - 1, (nb_teacher,), device=device
            )
            # Part B: Student
            max_pow = int(math.log2(config["nb_teacher_steps"])) - 1
            powers = torch.randint(1, max_pow + 1, (nb_student,), device=device)
            strides = 2**powers
            j_S = (
                torch.rand(nb_student, device=device)
                * (config["nb_teacher_steps"] - 2 * strides + 1)
            ).long()

            t1_idx = torch.cat([j_T, j_S])
            t2_idx = torch.cat([j_T + 1, j_S + strides])
            t3_idx = torch.cat([j_T + 2, j_S + 2 * strides])

            # Time Shifting
            def get_t(idx):
                raw = timesteps[idx]
                return (s * raw) / (1 + (s - 1) * raw)

            t1, t2, t3 = (
                get_t(t1_idx).view(-1, 1, 1),
                get_t(t2_idx).view(-1, 1, 1),
                get_t(t3_idx).view(-1, 1, 1),
            )
            x_t1 = (1 - t1) * x_0 + t1 * x_1

            # Consistency Coefficients
            alpha = ((t2 - t1) / (t3 - t1 + EPS)) * ((1 - t3) / (1 - t2 + EPS))
            beta = ((t3 - t2) / (t3 - t1 + EPS)) * ((1 - t1) / (1 - t2 + EPS))

            with torch.no_grad():
                # Teacher path
                P1_T, V1_T, _ = get_probs_vel_logits(
                    teacher_model,
                    t1[:nb_teacher],
                    x_t1[:nb_teacher],
                    cond_idx[:nb_teacher],
                )
                x_t2_T = x_t1[:nb_teacher] + (t2[:nb_teacher] - t1[:nb_teacher]) * V1_T
                P2_T, _, _ = get_probs_vel_logits(
                    slow_ema_model, t2[:nb_teacher], x_t2_T, cond_idx[:nb_teacher]
                )
                P_target_T = alpha[:nb_teacher] * P1_T + beta[:nb_teacher] * P2_T

                # Student Consistency path
                P1_S, V1_S, _ = get_probs_vel_logits(
                    fast_ema_model,
                    t1[nb_teacher:],
                    x_t1[nb_teacher:],
                    cond_idx[nb_teacher:],
                )
                x_t2_S = x_t1[nb_teacher:] + (t2[nb_teacher:] - t1[nb_teacher:]) * V1_S
                P2_S, _, _ = get_probs_vel_logits(
                    slow_ema_model, t2[nb_teacher:], x_t2_S, cond_idx[nb_teacher:]
                )
                P_target_S = alpha[nb_teacher:] * P1_S + beta[nb_teacher:] * P2_S

                P_target = torch.cat([P_target_T, P_target_S], dim=0)

            # --- Student Update ---
            _, _, student_logits = get_probs_vel_logits(
                student_model, t1, x_t1, cond_idx
            )
            log_probs = F.log_softmax(student_logits, dim=-1)

            # Variational Cross Entropy
            loss_per_token = torch.sum(-P_target * log_probs, dim=-1)
            loss = loss_per_token.mean()

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), max_norm=config["grad_clip"]
            )
            opt.step()

            slow_ema_model.update()
            fast_ema_model.update()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % config["sample_every_steps"] == 0:
                student_model.eval()
                imgs = vfm.generate(
                    n_samples=4,
                    cond_idx=torch.randint(0, 10, (4,), device=device),
                    n_steps=config["nb_teacher_steps"],
                    method="euler",
                )
                grid = utils.make_grid((imgs.clamp(-1, 1) + 1) / 2, nrow=4)
                wandb.log({"samples": wandb.Image(grid)}, step=global_step)
                student_model.train()

            wandb.log(
                {"train/loss": loss.item(), "train/grad_norm": grad_norm},
                step=global_step,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
