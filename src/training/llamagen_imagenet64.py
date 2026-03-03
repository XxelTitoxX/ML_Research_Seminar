import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import sys
from pathlib import Path
from tqdm import tqdm
import wandb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.llama_models import GPT_B
from models.vq_model import VQ_8
from models.vfm_wrapper import LlamaCatFlow

def load_imagenet64(batch_size):
    dataset = load_dataset("student/ImageNet-64", split="validation")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def transform_batch(examples):
        examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["png"]]
        return examples

    dataset.set_transform(transform_batch)

    def collate_fn(batch):
        images = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([0 for item in batch])
        return images, labels

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

def to_one_hot(c, k):
    return F.one_hot(c, num_classes=k).float()

@torch.no_grad()
def save_samples(vfm_wrapper, vq_model, epoch, step, device, n_samples=4):
    imgs = vfm_wrapper.generate(n_samples=n_samples)
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    os.makedirs("samples", exist_ok=True)
    grid = utils.make_grid(imgs, nrow=n_samples)
    path = f"samples/epoch_{epoch}_step_{step}.png"
    utils.save_image(grid, path)
    wandb.log({"samples": wandb.Image(path)}, step=step)

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    config = dict(
        vocab_size=16384,
        num_classes=1000,
        block_size=64,
        batch_size=128,
        n_epochs=200,
        sample_every_steps=500,
        lr=1e-4,
        grad_clip=1.0,
        resume_checkpoint=None,
    )

    wandb.init(project="catflow-cifar10", config=config)

    vocab_size = config["vocab_size"]
    num_classes = config["num_classes"]
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    sample_every_steps = config["sample_every_steps"]
    lr = config["lr"]
    grad_clip = config["grad_clip"]
    resume_checkpoint = config["resume_checkpoint"]

    vq_model = VQ_8().to(device)
    vq_checkpoint_path = hf_hub_download(repo_id="GAD-cell/VQ_VAE_8", filename="vq_ds8_c2i.pt", cache_dir="src/checkpoints")
    vq_model.load_state_dict(torch.load(vq_checkpoint_path, map_location=device)["model"])
    vq_model.eval()

    model = GPT_B(
        vocab_size=vocab_size,
        num_classes=num_classes,
        block_size=block_size,    
    ).to(device)
    model.train()

    vfm_wrapper = LlamaCatFlow(model, vq_model, obs_dim=(16,))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = load_imagenet64(batch_size)
    
    start_epoch = 0
    global_step = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = start_epoch * len(dataloader)
        print(f"Reprise depuis l'epoque {start_epoch}")

    noise_std = 1.0 / np.sqrt(vocab_size)

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}")
        
        for images, cond_idx in pbar:
            images = images.to(device)
            cond_idx = cond_idx.to(device)
            
            with torch.no_grad():
                quant, _, (_, _, idx) = vq_model.encode(images)
                x1_indices = idx.view(images.shape[0], -1)
            
            x0 = (torch.randn_like(to_one_hot(x1_indices, k=vq_model.config.codebook_size))* noise_std).to(device)
            t = torch.rand(images.shape[0], device=device)
            
            loss = vfm_wrapper.criterion(t, x0, x1_indices)
            
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            
            epoch_loss += loss.item()
            global_step += 1

            wandb.log({
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
            }, step=global_step)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % config["sample_every_steps"] == 0:
                model.eval()
                with torch.no_grad():
                    imgs = vfm_wrapper.generate(n_samples=4)
                    imgs = (imgs.clamp(-1, 1) + 1) / 2
                    grid = utils.make_grid(imgs, nrow=4)
                    wandb.log({"samples": wandb.Image(grid)}, step=global_step)
                model.train()

        
        avg_loss = epoch_loss / len(dataloader)
        wandb.log({
            "epoch/avg_loss": avg_loss,
            "epoch": epoch,
        }, step=global_step)
        print(f"Epoch {epoch} | avg_loss {avg_loss:.4f}")


        os.makedirs("checkpoints", exist_ok=True)
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "avg_loss": avg_loss,
            }, f"checkpoints/epoch_{epoch}.pt")

    wandb.finish()

if __name__ == "__main__":
    main()