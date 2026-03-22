import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.llama_models import GPT_B
from models.vq_model import VQ_Cifar_L
from models.vfm_wrapper import LlamaCatFlow


EPS=1e-8

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


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vq_model = VQ_Cifar_L().to(device)
    vq_model.load_state_dict(torch.load("checkpoints/vq_cifar_epoch_20.pt", map_location=device)["model_state_dict"])
    vq_model.eval()

    model = GPT_B(vocab_size=512, num_classes=10, block_size=256).to(device)
    path = "checkpoints/llamagen_epoch_250.pt"
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    """print(path)
    state_dict = checkpoint["ema_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_key = new_key.replace("module.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)"""
    model.eval()
    
    try:
        model = torch.compile(model)
    except:
        pass

    vfm_wrapper = LlamaCatFlow(model, vq_model, obs_dim=(256,))
    codebook = vfm_wrapper.get_codebook()

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cond_idx_vis = torch.arange(10, device=device).repeat(4)
    
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        vis_imgs = vfm_wrapper.generate(n_samples=40,n_steps=2,method="euler", cond_idx=cond_idx_vis)
        vis_imgs = (vis_imgs.clamp(-1, 1) + 1) / 2
        vis_imgs = vis_imgs.float().cpu()

    fig, axes = plt.subplots(4, 10, figsize=(15, 6))
    for i in range(4):
        for j in range(10):
            idx = i * 10 + j
            img_np = vis_imgs[idx].permute(1, 2, 0).numpy()
            axes[i, j].imshow(img_np)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(cifar_classes[j])
    plt.tight_layout()
    plt.savefig("samples_classes_150.png")
    plt.close()

    fid = FrechetInceptionDistance(feature=2048).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    num_fid_samples = 10000
    real_processed = 0

    for images, _ in tqdm(dataloader, desc="Real images FID"):
        if real_processed >= num_fid_samples:
            break
        images = images.to(device)
        images_uint8 = ((images.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
        fid.update(images_uint8, real=True)
        real_processed += images.size(0)

    fake_processed = 0
    batch_size_gen = 512

    with torch.no_grad():
        with tqdm(total=num_fid_samples, desc="Fake images FID") as pbar:
            while fake_processed < num_fid_samples:
                current_batch = min(batch_size_gen, num_fid_samples - fake_processed)
                cond_idx_gen = torch.randint(0, 10, (current_batch,), device=device)
                
                with torch.autocast(device_type="cuda"):
                    fake_imgs = vfm_wrapper.generate(n_samples=current_batch, n_steps=2, method="euler", cond_idx=cond_idx_gen)
                fake_imgs_uint8 = ((fake_imgs.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
                fid.update(fake_imgs_uint8, real=False)
                fake_processed += current_batch
                pbar.update(current_batch)

    fid_score = fid.compute()
    print(f"FID Score (150 epochs, {num_fid_samples} samples): {fid_score.item():.4f}")

if __name__ == "__main__":
    main()