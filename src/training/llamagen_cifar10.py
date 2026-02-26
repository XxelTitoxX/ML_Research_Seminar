import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.llama_models import GPT_B
from models.vq_model import VQ_8
from models.vfm_wrapper import LlamaCatFlow

def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def to_one_hot(c, k):
    return F.one_hot(c, num_classes=k).float()

def catflow_loss(model, x1_indices, cond_idx, k, device):
    b, seq_len = x1_indices.shape
    x1 = to_one_hot(x1_indices, k).to(device)
    
    x0 = torch.randn_like(x1)
    t = torch.rand((b, 1, 1), device=device)
    x_t = t * x1 + (1.0 - t) * x0
    
    logits, _ = model(x_t, t.squeeze(-1), cond_idx)
    if cond_idx:
        logits = logits[:, 1:, :]
    
    loss = F.cross_entropy(logits.reshape(-1, k), x1_indices.view(-1).to(device))
    return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = 16384
    num_classes = 10 
    block_size = 16 
    batch_size = 64
    n_steps = 10000
    lr = 1e-4

    vq_model = VQ_8().to(device)
    vq_checkpoint_path = "src/checkpoints/vq_ds8_c2i.pt"
    vq_model.load_state_dict(torch.load(vq_checkpoint_path, map_location=device)["model"])
    vq_model.eval()

    model = GPT_B(
            vocab_size=vocab_size,
            num_classes=num_classes,
            block_size=block_size,    
            ).to(device)
    
    llama_checkpoint_path = hf_hub_download(repo_id="FoundationVision/LlamaGen", filename="c2i_B_256.pt", cache_dir="src/checkpoints")
    model.load_state_dict(torch.load(llama_checkpoint_path, map_location=device), strict=False)
    model.train()

    vfm_wrapper = LlamaCatFlow(model, vq_model, obs_dim=(16,))
    '''
    print("test wrapper")
    out = vfm_wrapper.generate(n_samples=10,
                         top_p=0.9,
                         top_k=0.9)
    '''
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = load_cifar10(batch_size)
    
    pbar = tqdm(range(1, n_steps + 1), desc="Training CatFlow")
    data_iter = iter(dataloader)

    for step in pbar:
        try:
            images, cond_idx = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, cond_idx = next(data_iter)
            
        images = images.to(device)
        cond_idx = cond_idx.to(device)
        
        with torch.no_grad():
            quant, _, (_, _, idx) = vq_model.encode(images)
            x1_indices = idx.view(batch_size, -1)

        loss = catflow_loss(model, x1_indices, cond_idx=None, k=vocab_size, device=device) #remove none to use class conditioning
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

if __name__ == "__main__":
    main()