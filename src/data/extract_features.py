"""Extract image features using a pretrained backbone and save to disk.

Usage:
  python -m src.data.extract_features \
    --data_root ./data/raw/my_dataset \
    --output_dir ./data/processed/features \
    --model dino_v2 \
    --batch_size 64
"""

import argparse
import os
from pathlib import Path
from typing import Tuple
from tqdm import tqdm


def _require_torchvision():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency. Install torch and torchvision to run feature extraction."
        ) from exc


def _device_from_arg(device: str):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_model(model_name: str, device: str):
    import torch
    import torchvision

    model_name = model_name.lower()
    if model_name in {"dino", "dino_v2", "dinov2"}:
        # Uses torch.hub to download Meta's DINOv2 weights on first use.
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval()
        model.to(device)
        return model, (224, 224), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if model_name in {"inception", "inception_v3", "inceptionv3"}:
        weights = torchvision.models.Inception_V3_Weights.DEFAULT
        model = torchvision.models.inception_v3(weights=weights, aux_logits=False)
        # Replace classifier head to return 2048-dim pool features.
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(device)
        return model, (299, 299), list(weights.meta["mean"]), list(weights.meta["std"])

    raise ValueError(f"Unsupported model: {model_name}")


def _build_dataset(data_root: Path, image_size: Tuple[int, int], mean, std):
    import torchvision
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=str(data_root), transform=transform)
    return dataset


def _extract_features(model, dataloader, device: str):
    import torch

    feats = []
    paths = []
    model.eval()
    with torch.inference_mode():
        for batch, _ in dataloader:
            batch = batch.to(device)
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats.append(out.detach().cpu())
            # ImageFolder exposes samples with (path, class_idx)
            start = len(paths)
            end = start + batch.size(0)
            batch_paths = [dataloader.dataset.samples[i][0] for i in range(start, end)]
            paths.extend(batch_paths)
    features = torch.cat(feats, dim=0)
    return features, paths


def extract_cifar10_dinov2_features(
    data_root: Path,
    output_path: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "auto",
    split: str = "train",
):
    """Extract DINOv2 features for CIFAR-10 and save to a single .pt file.

    Saved format:
      {
        "features": Tensor [N, D],
        "targets": Tensor [N],
        "dataset": "cifar10",
        "split": "train" | "test",
        "model": "dinov2_vits14",
      }
    """
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms

    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got: {split}")

    run_device = _device_from_arg(device)
    model, image_size, mean, std = _load_model("dinov2", run_device)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=(split == "train"),
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    model.eval()
    feats = []
    pbar = tqdm(dataloader, desc=f"Extracting DINOv2 features for CIFAR-10 {split}")
    with torch.inference_mode():
        for batch, _ in pbar:
            batch = batch.to(run_device)
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats.append(out.detach().cpu())

    features = torch.cat(feats, dim=0)
    targets = torch.tensor(dataset.targets, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "targets": targets,
            "dataset": "cifar10",
            "split": split,
            "model": "dinov2_vits14",
        },
        output_path,
    )
    print(f"Saved CIFAR-10 DINOv2 features to {output_path}")


def main():
    _require_torchvision()

    parser = argparse.ArgumentParser(description="Extract features from an image dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to ImageFolder root")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed/features",
        help="Directory to write features",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dino_v2",
        choices=["dino_v2", "dinov2", "dino", "inception_v3", "inceptionv3", "inception"],
        help="Backbone to use",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--cifar10_dinov2",
        action="store_true",
        help="Run dedicated CIFAR-10 DINOv2 extraction and ignore --model ImageFolder flow.",
    )
    parser.add_argument(
        "--cifar10_split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="CIFAR-10 split when --cifar10_dinov2 is enabled.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Exact output file path. For CIFAR-10 use e.g. data/processed/cifar10_dinov2.pt",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cifar10_dinov2:
        output_path = (
            Path(args.output_file)
            if args.output_file
            else output_dir / f"cifar10_dinov2_{args.cifar10_split}.pt"
        )
        extract_cifar10_dinov2_features(
            data_root=data_root,
            output_path=output_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            split=args.cifar10_split,
        )
        return

    device = _device_from_arg(args.device)
    model, image_size, mean, std = _load_model(args.model, device)

    dataset = _build_dataset(data_root, image_size, mean, std)
    import torch
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    features, paths = _extract_features(model, dataloader, device)

    # Save features only
    dataset_name = data_root.name.replace(" ", "_")
    model_name = args.model.replace(" ", "_")
    features_path = output_dir / f"features_{dataset_name}_{model_name}.pt"

    torch.save({"features": features, "paths": paths}, features_path)

    print(f"Saved features to {features_path}")


if __name__ == "__main__":
    main()
