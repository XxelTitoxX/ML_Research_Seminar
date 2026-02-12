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

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
