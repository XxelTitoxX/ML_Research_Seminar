from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.utils import (
    argmax_readout,
    decode_indices_with_vqvae,
    generate_in_codebook_space,
    load_catflow_from_checkpoint,
    load_codebook_catflow_from_checkpoint,
    load_vqvae_from_checkpoint,
    pick_device,
    plot_first_positions_distributions,
    sample_probability_sequences,
    save_image_grid,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 images from a trained CatFlow transformer + VQ-VAE."
    )
    parser.add_argument("--flow_checkpoint", type=Path, default=ROOT / "checkpoints" / "step_25000.pt")
    parser.add_argument("--vq_checkpoint", type=Path, default=ROOT / "checkpoints" / "vq_cifar_epoch_20.pt")
    parser.add_argument("--output_path", type=Path, default=ROOT / "src" / "experiments" / "mosaic.png")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--in_codebook_space", action="store_true", help="Sample and save in codebook space without VQ-VAE decoding.")
    parser.add_argument("--nrow", type=int, default=0, help="Grid columns. If 0, uses sqrt(n_samples).")
    parser.add_argument("--ode_method", type=str, default="euler")
    parser.add_argument("--sigma_min", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--no_show", action="store_true", help="Disable matplotlib display.")
    parser.add_argument(
        "--n_prob_positions",
        type=int,
        default=3,
        help="Number of first sequence positions to plot per sample.",
    )
    parser.add_argument("--max_print_tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("`n_samples` must be > 0")

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"[setup] Device: {device}")

    print(f"[load] VQ-VAE checkpoint: {args.vq_checkpoint}")
    vq_model = load_vqvae_from_checkpoint(args.vq_checkpoint, device=device)

    print(f"[load] CatFlow transformer checkpoint: {args.flow_checkpoint}")
    flow, model_cfg, obs_dim = load_catflow_from_checkpoint(
        flow_checkpoint_path=args.flow_checkpoint,
        vq_model=vq_model,
        device=device,
        sigma_min=args.sigma_min,
    )
    print(f"[load] obs_dim={obs_dim}, grid=({model_cfg.grid_h}, {model_cfg.grid_w}), codebook_dim={model_cfg.codebook_dim}")

    if args.in_codebook_space:
        print("[sample] Generating samples in codebook space without VQ-VAE decoding...")
        decoded_img = generate_in_codebook_space(
            flow=flow,
            vq_model=vq_model,
            n_samples=args.n_samples,
            method=args.ode_method,
        )
        nrow = args.nrow if args.nrow > 0 else max(1, int(math.sqrt(args.n_samples)))
        saved = save_image_grid(
            images=decoded_img,
            output_path=args.output_path,
            nrow=nrow,
            show=not args.no_show,
            title=f"CatFlow CIFAR-10 samples in codebook space (n={args.n_samples})",
        )
        print(f"[done] Saved mosaic image to: {saved}")
        return
    probs = sample_probability_sequences(flow, n_samples=args.n_samples, method=args.ode_method, n_steps=500)
    plot_first_positions_distributions(
        probs,
        n_positions=args.n_prob_positions,
        show=not args.no_show,
    )

    indices = argmax_readout(probs)
    # print_index_sequences(indices, max_tokens=max(32, args.max_print_tokens))

    recon = decode_indices_with_vqvae(
        vq_model=vq_model,
        indices=indices,
        codebook_dim=model_cfg.codebook_dim,
        grid_h=model_cfg.grid_h,
        grid_w=model_cfg.grid_w,
    )

    nrow = args.nrow if args.nrow > 0 else max(1, int(math.sqrt(args.n_samples)))
    saved = save_image_grid(
        images=recon,
        output_path=args.output_path,
        nrow=nrow,
        show=not args.no_show,
        title=f"CatFlow CIFAR-10 samples (n={args.n_samples})",
    )
    print(f"[done] Saved mosaic image to: {saved}")


if __name__ == "__main__":
    main()
