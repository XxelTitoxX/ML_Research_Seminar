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
from torch_fidelity import calculate_metrics

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
EPS = 1e-8

from models.llama_models import GPT_B
from models.vq_model import VQ_Cifar_L
from models.vfm_wrapper import LlamaCatFlow

class FIDDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    def __len__(self):
        return self.tensor.shape[0]
    def __getitem__(self, idx):
        return self.tensor[idx]

@torch.no_grad()
def sample(model, vq_vae, codebook, device, cond_idx, steps):
    model.eval()
    batch_size = cond_idx.shape[0]
    z_dim = codebook.shape[-1]
    
    x = torch.randn(batch_size, 256, z_dim, device=device)
    
    # Create exact timesteps:[0.0, 0.125, 0.25, ..., 1.0]
    ts = torch.linspace(0, 1.0, steps + 1, device=device)

    for i in range(steps):
        t_val = ts[i]
        t_next = ts[i+1]
        dt = t_next - t_val

        t = torch.full((batch_size,), t_val, device=device)
        logits = model(t, x, cond_idx)

        # Step using the exact soft math the model was trained on
        probs = F.softmax(logits, dim=-1)
        mu = torch.matmul(probs, codebook)

        vel = (mu - x) / (1.0 - t_val + EPS)
        x = x + vel * dt

    # x is now fully integrated to t = 1.0
    # FIX: Project the final continuous vector to the nearest codebook tokens
    x_flat = x.reshape(-1, z_dim)
    
    # Compute L2 distance to codebook
    d = torch.sum(x_flat ** 2, dim=1, keepdim=True) + \
        torch.sum(codebook ** 2, dim=1) - 2 * \
        torch.matmul(x_flat, codebook.t())
        
    final_indices = torch.argmin(d, dim=1)
    final_indices = final_indices.view(batch_size, 16, 16)
    
    images = vq_vae.decode_code(final_indices, shape=(batch_size, -1, 16, 16))

    return (images.clamp(-1, 1) + 1) / 2

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
        save_every_epochs=1,
        lr=5e-5,
        grad_clip=5.0,
        slow_mu=0.999,
        fast_mu=0.99,
        s_range=[1.0, 1.2],
        nb_teacher_steps=8,
        ratio_teacher_samples=0.6,
        resume_checkpoint=None,
        fid_samples=1024
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
    vfm = LlamaCatFlow(student_model, vq_model, obs_dim=(config["block_size"],), temperature=1.0)
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
    fid_samples = config["fid_samples"]

    opt = torch.optim.AdamW(
        student_model.parameters(), lr=config["lr"], weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["n_epochs"] * len(dataloader))

    # Functional helper to get probs and velocity using the wrapper's logic but varying the model
    def get_probs_vel_logits(model_instance, t, x, cond):
        orig_model = vfm.model
        vfm.model = model_instance

        t_proc = vfm.process_timesteps(t.view(-1), x)
        logits = vfm.model(t_proc, x, cond_idx=cond)
        
        # 1. Pure, soft probabilities
        probs = F.softmax(logits, dim=-1)
        
        # 2. Pure, soft expected vector (Crucial for the math to cancel)
        mu_soft = torch.matmul(probs, codebook)
        vel_soft = (mu_soft - x) / (1 - t_proc.view(-1, 1, 1) + EPS)

        vfm.model = orig_model 
        return probs, vel_soft, logits

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

                _, _, student_logits = get_probs_vel_logits(
                    student_model, t1, x_t1, cond_idx
                )
                log_probs = F.log_softmax(student_logits, dim=-1)
                loss_per_token = F.kl_div(log_probs, P_target, reduction='none').sum(dim=-1)
                
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
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    # 2. Visual grid for inspection
                    sample_cond = torch.randint(0, 10, (4,), device=device)
                    visual_imgs = sample(slow_ema_model.module, vq_model, codebook, device, sample_cond, config["nb_teacher_steps"])
                    grid = utils.make_grid(visual_imgs, nrow=4)

                    # 3. Collect samples for FID
                    fid_tensors = []
                    pbar_fid = tqdm(range(0, config["fid_samples"], 256), desc="FID Sampling", leave=False)
                    for _ in pbar_fid:
                        c = torch.randint(0, 10, (256,), device=device)
                        out = sample(slow_ema_model.module, vq_model, codebook, device, c, config["nb_teacher_steps"])
                        # Fidelity expects uint8 [0, 255]
                        out = (out * 127.5).to(torch.uint8)
                        fid_tensors.append(out.cpu())
                    
                    full_fid_tensor = torch.cat(fid_tensors, dim=0)
                    
                    # 4. Wrap tensor in Dataset for torch-fidelity
                    fid_input = FIDDataset(full_fid_tensor)

                    # 5. Compute Metrics
                    metrics = calculate_metrics(
                        input1=fid_input,
                        input2='cifar10-train',
                        cuda=True,
                        fid=True,
                        verbose=False,
                    )

                wandb.log({
                    "samples": wandb.Image(grid),
                    "eval/fid": metrics['frechet_inception_distance']
                }, step=global_step)

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
