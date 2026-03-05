import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import sys
import os
from pathlib import Path
import lpips

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.vq_model import VQ_Cifar, VQ_Cifar_L

def evaluate_rfid(model, dataloader, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    model.eval()
    
    with torch.no_grad():
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            real_imgs_uint8 = ((real_imgs.clamp(-1, 1) + 1) / 2)
            fid.update(real_imgs_uint8, real=True)
            
            reconstructed, _ = model(real_imgs)
            
            if isinstance(reconstructed, tuple):
                reconstructed = reconstructed[0]
                
            reconstructed_uint8 = (reconstructed.clamp(-1, 1) + 1) / 2
            fid.update(reconstructed_uint8, real=False)
            
    score = fid.compute().item()
    model.train()
    return score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = VQ_Cifar_L().to(device)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    os.makedirs("checkpoints", exist_ok=True)

    model.train()
    for epoch in range(50):
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            reconstructed, (vq_loss, commit_loss, entropy_loss, usage) = model(images)
            
            recon_loss = F.mse_loss(reconstructed, images)
            p_loss = perceptual_loss_fn(reconstructed, images).mean()
            
            loss = recon_loss + 0.4 * p_loss + vq_loss + 1.2*commit_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}] | Total: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | LPIPS: {p_loss.item():.4f} | Usage: {usage:.2%}")
        
        print(f"\n rFID at epoch {epoch}...")
        rfid_score = evaluate_rfid(model, test_loader, device)
        print(f"--- epoch {epoch} | rFID: {rfid_score:.4f} ---")
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'rfid': rfid_score
            }, f"checkpoints/vq_cifar_epoch_{epoch}.pt")

    torch.save(model.state_dict(), "checkpoints/vq_cifar_final.pt")

if __name__ == "__main__":
    main()