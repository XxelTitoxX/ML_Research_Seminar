import torch
import wandb
import math
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import DataLoader
import os
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)


def main():
    scaler = torch.amp.GradScaler("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = dict(
        vocab_size=512,
        num_classes=10,
        block_size=256,
        batch_size=64,
        n_epochs=50,
        sample_every_steps=500,
        save_every_epochs=5,
        lr=5e-5,
        grad_clip=5.0,
        slow_mu=0.999,
        fast_mu=0.99,
        s_range=[1.0, 1.2],
        nb_teacher_steps=8,
        ratio_teacher_samples=0.6,
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
    student_model = torch.compile(student_model)
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

    
    dataloader = load_cifar10(config["batch_size"])
    timesteps = torch.linspace(0, 1, config["nb_teacher_steps"] + 1, device=device)

    opt = torch.optim.AdamW(
        student_model.parameters(), lr=config["lr"], weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["n_epochs"] * len(dataloader))

    # Functional helper to get probs and velocity using the wrapper's logic but varying the model
    def get_probs_vel_logits(model_instance, t, x, cond):
        # We temporarily point vfm.model to the instance we want to query
        orig_model = vfm.model
        vfm.model = model_instance

        t_proc = vfm.process_timesteps(t.view(-1), x)
        logits = vfm.model(t_proc, x, cond_idx=cond)
        probs = F.softmax(logits, dim=-1)
        mu_t = torch.matmul(probs, codebook)
        vel = (mu_t - x) / (1 - t_proc.view(-1, 1, 1) + vfm.eps_)

        vfm.model = orig_model 
        return probs, vel, logits

    global_step = 0
    for epoch in range(1, config["n_epochs"] + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config["n_epochs"]}")

        for images, cond_idx in pbar:
            opt.zero_grad()
            images, cond_idx = images.to(device), cond_idx.to(device)
            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    _, _, (_, _, idx) = vq_model.encode(images)
                    x_1_indices = idx.view(images.shape[0], -1)
                    x_1 = F.embedding(x_1_indices, codebook)

                x_0 = torch.randn_like(x_1).to(device)
                s = torch.empty(config["batch_size"], device=device).uniform_(
                    *config["s_range"]
                )

                nb_teacher = int(config["ratio_teacher_samples"] * config["batch_size"])
                nb_student = config["batch_size"] - nb_teacher

                j_T = torch.randint(
                    0, config["nb_teacher_steps"] - 1, (nb_teacher,), device=device
                )
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

                def get_t(idx):
                    raw = timesteps[idx]
                    return (s * raw) / (1 + (s - 1) * raw)

                t1, t2, t3 = (
                    get_t(t1_idx).view(-1, 1, 1),
                    get_t(t2_idx).view(-1, 1, 1),
                    get_t(t3_idx).view(-1, 1, 1),
                )

                
                x_t1 = (1 - t1) * x_0 + t1 * x_1

                alpha = ((t2 - t1) / (t3 - t1 + EPS)) * ((1 - t3) / (1 - t2 + EPS))
                beta = ((t3 - t2) / (t3 - t1 + EPS)) * ((1 - t1) / (1 - t2 + EPS))

                with torch.no_grad():
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
                    target_idx = P_target.argmax(dim=-1)

                _, _, student_logits = get_probs_vel_logits(
                    student_model, t1, x_t1, cond_idx
                )
                loss_per_token = F.cross_entropy(student_logits.transpose(1, 2), target_idx, reduction='none')
                loss_per_image = loss_per_token.mean(dim=1)
                distillation_loss = loss_per_image[:nb_teacher].mean()     
                self_consistency_loss = loss_per_image[nb_teacher:].mean()
                loss = (distillation_loss * nb_teacher + self_consistency_loss * nb_student) / config["batch_size"]

            with torch.no_grad():
                student_probs = F.softmax(student_logits, dim=-1)
                student_entropy = -torch.sum(student_probs * torch.log(student_probs + EPS), dim=-1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=config["grad_clip"])
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            slow_ema_model.update_parameters(student_model)
            fast_ema_model.update_parameters(student_model)

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

            if global_step % 10 == 0:
                wandb.log({
                        "train/total_loss": loss.item(),
                        "train/distillation_loss": distillation_loss.item(),
                        "train/self_consistency_loss": self_consistency_loss.item(),
                        "train/grad_norm": grad_norm,
                        "train/student_entropy": student_entropy.item()
                    }, step=global_step)
        avg_loss = epoch_loss / len(dataloader)
        wandb.log({
            "epoch/avg_loss": avg_loss,
            "epoch": epoch,
        }, step=global_step)
        print(f"Epoch {epoch} | avg_loss {avg_loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        if epoch % config["save_every_epochs"] == 0:
            # Use getattr for robustness
            raw_model = getattr(student_model, '_orig_mod', student_model)
            slow_ema_inner = getattr(slow_ema_model.module, '_orig_mod', slow_ema_model.module)

            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "ema_state_dict": slow_ema_inner.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            }, f"checkpoints/distilled_llamagen_epoch_{epoch}.pt")

    wandb.finish()


if __name__ == "__main__":
    main()
