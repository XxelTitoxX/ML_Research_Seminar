import torch
import wandb
import math
import copy
import datasets
from diffusers import AutoencoderKL
from transformers import AutoModel
import argparse


def vanilla_scfm(
    teacher_net,
    student_net,
    ema_net,
    vae,
    dataloader,
    nb_steps,
    batch_size,
    nb_teacher_steps,
    ratio_teacher_samples,
    device,
    optimizer,
    cycling,
    restart_interval,
    run,
):
    loss_fn = torch.nn.MSELoss(reduction="sum")
    nb_teacher_samples = int(ratio_teacher_samples * batch_size)
    nb_student_samples = batch_size - nb_teacher_samples
    timesteps = torch.linspace(1, 0, nb_teacher_steps + 1, device=device)

    for step in range(nb_steps):
        optimizer.zero_grad()
        x_0 = next(dataloader).to(device)
        if vae:
            x_0 = vae.encode(x_0).latent_dist.sample()
            x_0 = x_0 * vae.config.scaling_factor
        x_1 = torch.randn_like(x_0)

        s = torch.empty(batch_size, device=device).uniform_(2.5, 4.5)

        # Part A: Teacher (Skip = 1)
        j_teacher = torch.randint(
            0, nb_teacher_steps - 1, (nb_teacher_samples,), device=device
        )
        t1_idx_T = j_teacher
        t2_idx_T = j_teacher + 1
        t3_idx_T = j_teacher + 2

        # Part B: Student (Random Stride)
        max_power = int(math.log2(nb_teacher_steps)) - 1
        powers = torch.randint(1, max_power + 1, (nb_student_samples,), device=device)
        strides = 2**powers

        # 2. Sample j such that j + 2*stride <= n
        #    max_j = n - 2*stride
        #    We generate a random float [0, 1] and map it to the valid range for each stride
        max_j = nb_teacher_steps - 2 * strides
        random_floats = torch.rand(nb_student_samples, device=device)
        j_student = (random_floats * (max_j + 1)).long()  # +1 because floor

        t1_idx_S = j_student
        t2_idx_S = j_student + strides
        t3_idx_S = j_student + 2 * strides

        t1_idx = torch.cat([t1_idx_T, t1_idx_S])
        t2_idx = torch.cat([t2_idx_T, t2_idx_S])
        t3_idx = torch.cat([t3_idx_T, t3_idx_S])

        def get_shifted_time(indices, shift_vals):
            raw_t = timesteps[indices]
            return (shift_vals * raw_t) / (1 + (shift_vals - 1) * raw_t)

        t_1 = get_shifted_time(t1_idx, s).view(-1, 1, 1, 1)
        t_2 = get_shifted_time(t2_idx, s).view(-1, 1, 1, 1)
        t_3 = get_shifted_time(t3_idx, s).view(-1, 1, 1, 1)

        x_t1 = (1 - t_1) * x_0 + t_1 * x_1
        x_t2 = (1 - t_2) * x_0 + t_2 * x_1

        d_1, d_2 = t_1 - t_2, t_2 - t_3
        w = d_1 / (d_1 + d_2 + 1e-8)

        student_vel = student_net(x_t1, t_1.squeeze())

        with torch.no_grad():
            target_vel = torch.zeros_like(student_vel)

            teacher_vel_t1 = teacher_net(
                x_t1[:nb_teacher_samples], t_1[:nb_teacher_samples].squeeze()
            )
            teacher_vel_t2 = teacher_net(
                x_t2[:nb_teacher_samples], t_2[:nb_teacher_samples].squeeze()
            )
            target_vel[:nb_teacher_samples] = (
                w[:nb_teacher_samples] * teacher_vel_t1
                + (1 - w[:nb_teacher_samples]) * teacher_vel_t2
            )

            ema_vel_t1 = ema_net(
                x_t1[nb_teacher_samples:], t_1[nb_teacher_samples:].squeeze()
            )
            ema_vel_t2 = ema_net(
                x_t2[nb_teacher_samples:], t_2[nb_teacher_samples:].squeeze()
            )
            target_vel[nb_teacher_samples:] = (
                w[nb_teacher_samples:] * ema_vel_t1
                + (1 - w[nb_teacher_samples:]) * ema_vel_t2
            )

        distillation_loss = (
            loss_fn(student_vel[:nb_teacher_samples], target_vel[:nb_teacher_samples])
            / batch_size
        )
        self_consistency_loss = (
            loss_fn(student_vel[nb_teacher_samples:], target_vel[nb_teacher_samples:])
            / batch_size
        )
        scfm_loss = distillation_loss + self_consistency_loss

        scfm_loss.backward()
        optimizer.step()

        if cycling and step != 0 and step % restart_interval == 0:
            ema_dict = ema_net.state_dict()
            student_dict = student_net.state_dict()
            for key in ema_dict.keys():
                ema_dict[key] = copy.deepcopy(student_dict[key])
        else:
            ema_net.update_parameters(student_net)

        if run:
            run.log(
                {
                    "Self-consistency loss": self_consistency_loss.item(),
                    "Distillation loss": distillation_loss.item(),
                    "SCFM loss": scfm_loss.item(),
                }
            )
    if run:
        run.finish()


def dual_ema_scfm(
    teacher_net,
    student_net,
    slow_ema_net,
    fast_ema_net,
    vae,
    dataloader,
    nb_steps,
    batch_size,
    nb_teacher_steps,
    ratio_teacher_samples,
    device,
    optimizer,
    run,
):
    loss_fn = torch.nn.MSELoss(reduction="sum")
    nb_teacher_samples = int(ratio_teacher_samples * batch_size)
    nb_student_samples = batch_size - nb_teacher_samples
    timesteps = torch.linspace(1, 0, nb_teacher_steps + 1, device=device)

    for step in range(nb_steps):
        optimizer.zero_grad()
        x_0 = next(dataloader).to(device)
        if vae:
            x_0 = vae.encode(x_0).latent_dist.sample()
            x_0 = x_0 * vae.config.scaling_factor
        x_1 = torch.randn_like(x_0)

        s = torch.empty(batch_size, device=device).uniform_(2.5, 4.5)

        # Part A: Teacher (Skip = 1)
        j_teacher = torch.randint(
            0, nb_teacher_steps - 1, (nb_teacher_samples,), device=device
        )
        t1_idx_T = j_teacher
        t2_idx_T = j_teacher + 1
        t3_idx_T = j_teacher + 2

        # Part B: Student (Random Stride)
        max_power = int(math.log2(nb_teacher_steps)) - 1
        powers = torch.randint(1, max_power + 1, (nb_student_samples,), device=device)
        strides = 2**powers

        # 2. Sample j such that j + 2*stride <= n
        #    max_j = n - 2*stride
        #    We generate a random float [0, 1] and map it to the valid range for each stride
        max_j = nb_teacher_steps - 2 * strides
        random_floats = torch.rand(nb_student_samples, device=device)
        j_student = (random_floats * (max_j + 1)).long()  # +1 because floor

        t1_idx_S = j_student
        t2_idx_S = j_student + strides
        t3_idx_S = j_student + 2 * strides

        t1_idx = torch.cat([t1_idx_T, t1_idx_S])
        t2_idx = torch.cat([t2_idx_T, t2_idx_S])
        t3_idx = torch.cat([t3_idx_T, t3_idx_S])

        def get_shifted_time(indices, shift_vals):
            raw_t = timesteps[indices]
            return (shift_vals * raw_t) / (1 + (shift_vals - 1) * raw_t)

        t_1 = get_shifted_time(t1_idx, s).view(-1, 1, 1, 1)
        t_2 = get_shifted_time(t2_idx, s).view(-1, 1, 1, 1)
        t_3 = get_shifted_time(t3_idx, s).view(-1, 1, 1, 1)

        x_t1 = (1 - t_1) * x_0 + t_1 * x_1
        x_t2 = (1 - t_2) * x_0 + t_2 * x_1

        d_1, d_2 = t_1 - t_2, t_2 - t_3
        w = d_1 / (d_1 + d_2 + 1e-8)

        student_vel = student_net(x_t1, t_1.squeeze())

        with torch.no_grad():
            target_vel = torch.zeros_like(student_vel)

            teacher_vel_t1 = teacher_net(
                x_t1[:nb_teacher_samples], t_1[:nb_teacher_samples].squeeze()
            )
            slow_ema_vel_t2 = slow_ema_net(
                x_t2[:nb_teacher_samples], t_2[:nb_teacher_samples].squeeze()
            )
            target_vel[:nb_teacher_samples] = (
                w[:nb_teacher_samples] * teacher_vel_t1
                + (1 - w[:nb_teacher_samples]) * slow_ema_vel_t2
            )

            fast_ema_vel_t1 = fast_ema_net(
                x_t1[nb_teacher_samples:], t_1[nb_teacher_samples:].squeeze()
            )
            slow_ema_vel_t2 = slow_ema_net(
                x_t2[nb_teacher_samples:], t_2[nb_teacher_samples:].squeeze()
            )
            target_vel[nb_teacher_samples:] = (
                w[nb_teacher_samples:] * fast_ema_vel_t1
                + (1 - w[nb_teacher_samples:]) * slow_ema_vel_t2
            )

        distillation_loss = (
            loss_fn(student_vel[:nb_teacher_samples], target_vel[:nb_teacher_samples])
            / batch_size
        )
        self_consistency_loss = (
            loss_fn(student_vel[nb_teacher_samples:], target_vel[nb_teacher_samples:])
            / batch_size
        )
        scfm_loss = distillation_loss + self_consistency_loss

        scfm_loss.backward()
        optimizer.step()

        slow_ema_net.update()
        fast_ema_net.update()

        if run:
            run.log(
                {
                    "Self-consistency loss": self_consistency_loss.item(),
                    "Distillation loss": distillation_loss.item(),
                    "SCFM loss": scfm_loss.item(),
                }
            )
    if run:
        run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla/Dual SCFM Training Script")

    # --- Model & Data Paths ---
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to training data or name of the dataset on HuggingFace.",
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        required=True,
        help="Path to pre-trained teacher or name on HuggingFace (e.g., Flux).",
    )
    parser.add_argument(
        "--student_path",
        type=str,
        required=True,
        help="Path to student model or name on HuggingFace.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        help="Path to VAE model or name on HuggingFace.",
    )

    # --- Training Hyperparameters ---
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Total training batch size."
    )
    parser.add_argument(
        "--nb_steps",
        type=int,
        default=100000,
        help="Total number of training iterations.",
    )
    parser.add_argument(
        "--learning_rate", "--lr", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", "--wd", type=float, default=2e-5, help="Weight decay."
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM.",
    )

    # --- SCFM Algorithm Parameters ---
    parser.add_argument(
        "--method",
        choices=["vanilla", "dual"],
        default="vanilla",
        help="Distillation method.",
    )
    parser.add_argument(
        "--nb_teacher_steps",
        type=int,
        default=32,
        help="Number of teacher discrete steps (n).",
    )
    parser.add_argument(
        "--ratio_teacher_samples",
        type=float,
        default=0.4,
        help="Ratio of batch used for teacher guidance (k/N).",
    )
    parser.add_argument(
        "--s_range",
        type=float,
        nargs=2,
        default=[2.5, 4.5],
        help="Range for sampling the shift parameter s (provide two values).",
    )

    # --- EMA & Dual Settings ---
    parser.add_argument(
        "--fast_mu",
        type=float,
        default=0.99,
        help="EMA decay rate for the 'fast' model (Dual mode).",
    )
    parser.add_argument(
        "--slow_mu",
        type=float,
        default=0.999,
        help="EMA decay rate for the 'slow' model (Vanilla/Dual mode).",
    )
    parser.add_argument(
        "--cycling",
        action="store_true",
        help="Enable cyclic restarting of the EMA model.",
    )
    parser.add_argument(
        "--restart_interval",
        type=int,
        default=1000,
        help="Steps between EMA restarts if cycling is enabled.",
    )

    # --- LoRA Configuration ---
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="Rank (r) for LoRA adapters."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="Alpha scaling for LoRA."
    )

    # --- Miscellaneous ---
    # Since default is True, we add a flag to disable it
    parser.add_argument(
        "--no_wandb",
        action="store_false",
        dest="use_wandb",
        help="Disable WandB monitoring.",
    )
    parser.set_defaults(use_wandb=True)

    parser.add_argument(
        "--project_name",
        type=str,
        help="WandB project name.",
    )

    return parser.parse_args()


def main(args):
    """
    SCFM Training Script: Distilling Flow Matching models into few-step samplers.
    """

    run = None
    if args.use_wandb:
        run = wandb.init(project=args.project_name, config=args)
        print("WandB is enabled")

    print(f"Starting {args.method} SCFM training...")
    print(f"Resolution Shift Range: {args.s_range[0]} to {args.s_range[1]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    dataset = datasets.load_dataset(args.dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )
    print(f"Loaded image dataset stored at {args.dataset_path}")

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16

    teacher_net = AutoModel.from_pretrained(
        args.teacher_path, weights_only=False, torch_dtype=dtype, device=device
    )
    student_net = AutoModel.from_pretrained(
        args.student_path, weights_only=False, torch_dtype=dtype, device=device
    )
    slow_ema_net = torch.optim.swa_utils.AveragedModel(
        student_net,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.slow_mu),
        device=device,
    )
    fast_ema_net = None
    if args.method == "dual":
        fast_ema_net = torch.optim.swa_utils.AveragedModel(
            student_net,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.fast_mu),
            device=device,
        )
    vae = None
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=dtype)

    teacher_net.eval()
    student_net.train()
    slow_ema_net.eval()
    if fast_ema_net:
        fast_ema_net.eval()

    optimizer = torch.optim.AdamW(
        student_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    if args.method == "vanilla":
        vanilla_scfm(
            teacher_net,
            student_net,
            slow_ema_net,
            vae,
            dataloader,
            args.nb_steps,
            args.batch_size,
            args.nb_teacher_steps,
            args.ratio_teacher_samples,
            device,
            optimizer,
            args.cycling,
            args.restart_interval,
            run,
        )

    if args.method == "dual":
        dual_ema_scfm(
            teacher_net,
            student_net,
            slow_ema_net,
            fast_ema_net,
            vae,
            dataloader,
            args.nb_steps,
            args.batch_size,
            args.nb_teacher_steps,
            args.ratio_teacher_samples,
            device,
            optimizer,
            run,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
