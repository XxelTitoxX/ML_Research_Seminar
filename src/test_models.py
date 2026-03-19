import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.llama_models import GPT_B
from models.vq_model import VQ_Cifar_L
from models.vfm_wrapper import LlamaCatFlow

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vq_model = VQ_Cifar_L().to(device)
    vq_model.load_state_dict(torch.load("checkpoints/vq_cifar_epoch_20.pt", map_location=device)["model_state_dict"])
    vq_model.eval()

    model = GPT_B(vocab_size=512, num_classes=10, block_size=256).to(device)
    checkpoint = torch.load("checkpoints/distilled_llamagen_epoch_5.pt", map_location=device)
    state_dict = checkpoint["ema_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_key = new_key.replace("module.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    
    try:
        model = torch.compile(model)
    except:
        pass

    vfm_wrapper = LlamaCatFlow(model, vq_model, obs_dim=(256,))

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cond_idx_vis = torch.arange(10, device=device).repeat(4)
    
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        vis_imgs = vfm_wrapper.generate(n_samples=40,n_steps=8,method="euler", cond_idx=cond_idx_vis)
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
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

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
                    fake_imgs = vfm_wrapper.generate(n_samples=current_batch, n_steps=8, method="euler", cond_idx=cond_idx_gen)
                    
                fake_imgs_uint8 = ((fake_imgs.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
                fid.update(fake_imgs_uint8, real=False)
                fake_processed += current_batch
                pbar.update(current_batch)

    fid_score = fid.compute()
    print(f"FID Score (150 epochs, {num_fid_samples} samples): {fid_score.item():.4f}")

if __name__ == "__main__":
    main()