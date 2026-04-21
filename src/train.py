from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import wandb
from tqdm.auto import tqdm
from src.dataset import create_dataloaders
from src.metrics import move_labels_to_device, run_validation
from src.model import create_model, create_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Deformable DETR on the homework dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("nycu-hw2-data"))
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--train-annotation-file", type=Path, default=None)
    parser.add_argument("--valid-image-dir", type=Path, default=None)
    parser.add_argument("--valid-annotation-file", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--scheduler", type=str, default="warmup_cosine", choices=["warmup_cosine", "cosine", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--metric-backend", type=str, default="pycocotools", choices=["pycocotools", "custom"])
    parser.add_argument("--amp", type=str, default="bf16" if torch.cuda.is_available() else "none", choices=["none", "bf16", "fp16"])
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--augment", action="store_true", help="Enable basic color jitter augmentation for training.")
    parser.add_argument("--aug-color", action="store_true", help="Enable color jitter augmentation.")
    parser.add_argument("--aug-geom", action="store_true", help="Enable geometric affine augmentation.")
    parser.add_argument("--color-prob", type=float, default=1.0)
    parser.add_argument("--geom-prob", type=float, default=0.0)
    parser.add_argument("--color-brightness", type=float, default=0.3)
    parser.add_argument("--color-contrast", type=float, default=0.3)
    parser.add_argument("--color-saturation", type=float, default=0.3)
    parser.add_argument("--color-hue", type=float, default=0.03)
    parser.add_argument("--geom-degrees", type=float, default=3.0)
    parser.add_argument("--geom-translate", type=float, default=0.02)
    parser.add_argument("--geom-scale-min", type=float, default=0.95)
    parser.add_argument("--geom-scale-max", type=float, default=1.05)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/detr"))
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint .pt file to resume training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", type=str, default="dlcv-hw2-detr-tranformer-from-sketch")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def init_wandb(args: argparse.Namespace, paths: dict[str, Path], output_dir: Path):
    if args.wandb_mode == "disabled":
        return None

    config = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "warmup_start_factor": args.warmup_start_factor,
        "batch_size": args.batch_size,
        "metric_backend": args.metric_backend,
        "amp": args.amp,
        "compile": args.compile,
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_workers": args.num_workers,
        "augment": args.augment,
        "aug_color": args.aug_color,
        "aug_geom": args.aug_geom,
        "color_prob": args.color_prob,
        "geom_prob": args.geom_prob,
        "color_brightness": args.color_brightness,
        "color_contrast": args.color_contrast,
        "color_saturation": args.color_saturation,
        "color_hue": args.color_hue,
        "geom_degrees": args.geom_degrees,
        "geom_translate": args.geom_translate,
        "geom_scale_min": args.geom_scale_min,
        "geom_scale_max": args.geom_scale_max,
        "device": args.device,
        "train_image_dir": str(paths["train_image_dir"]),
        "train_annotation_file": str(paths["train_annotation_file"]),
        "valid_image_dir": str(paths["valid_image_dir"]),
        "valid_annotation_file": str(paths["valid_annotation_file"]),
        "output_dir": str(output_dir),
        "resume": str(args.resume) if args.resume is not None else None,
    }

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=config,
    )


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
    amp_autocast_dtype: torch.dtype | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0

    phase = "train" if is_training else "eval"
    epoch_prefix = f"Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "Epoch"
    progress = tqdm(dataloader, desc=f"{epoch_prefix} [{phase}]", leave=False)

    for step, batch in enumerate(progress, start=1):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = move_labels_to_device(batch["labels"], device)
        amp_enabled = amp_autocast_dtype is not None and device.type == "cuda"

        with torch.set_grad_enabled(is_training):
            with torch.autocast(device_type="cuda", dtype=amp_autocast_dtype, enabled=amp_enabled):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss detected at step {step}: {loss.item()}")

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / step:.4f}")

    return total_loss / max(len(dataloader), 1)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    train_image_dir = args.train_image_dir or args.data_root / "train"
    train_annotation_file = args.train_annotation_file or args.data_root / "train.json"
    valid_image_dir = args.valid_image_dir or args.data_root / "valid"
    valid_annotation_file = args.valid_annotation_file or args.data_root / "valid.json"

    return {
        "train_image_dir": train_image_dir,
        "train_annotation_file": train_annotation_file,
        "valid_image_dir": valid_image_dir,
        "valid_annotation_file": valid_annotation_file,
    }


def train(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    if args.resume is not None:
        run_dir = args.resume.parent
    else:
        run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    if args.augment and not args.aug_color and not args.aug_geom:
        args.aug_color = True

    wandb_run = init_wandb(args, paths, run_dir)

    processor = create_processor(args.model_name)
    model = create_model(paths["train_annotation_file"], args.model_name)

    train_loader, valid_loader = create_dataloaders(
        train_image_dir=paths["train_image_dir"],
        train_annotation_file=paths["train_annotation_file"],
        valid_image_dir=paths["valid_image_dir"],
        valid_annotation_file=paths["valid_annotation_file"],
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        aug_color=args.aug_color,
        aug_geom=args.aug_geom,
        color_prob=args.color_prob,
        geom_prob=args.geom_prob,
        color_brightness=args.color_brightness,
        color_contrast=args.color_contrast,
        color_saturation=args.color_saturation,
        color_hue=args.color_hue,
        geom_degrees=args.geom_degrees,
        geom_translate=args.geom_translate,
        geom_scale_min=args.geom_scale_min,
        geom_scale_max=args.geom_scale_max,
    )

    device = torch.device(args.device)
    model.to(device)
    amp_autocast_dtype: torch.dtype | None = None
    if args.amp != "none":
        if device.type != "cuda":
            print(f"AMP requested ({args.amp}) but device={device.type}; AMP disabled.")
        elif args.amp == "bf16":
            amp_autocast_dtype = torch.bfloat16
        else:
            amp_autocast_dtype = torch.float16

    backbone_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": args.lr},
            {"params": backbone_parameters, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "warmup_cosine":
        warmup_epochs = max(0, min(args.warmup_epochs, args.epochs - 1))
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_valid_loss = float("inf")
    best_map_50_95 = float("-inf")
    start_epoch = 1

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")

        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_loss = float(checkpoint.get("best_valid_loss", checkpoint.get("valid_loss", float("inf"))))
        best_map_50_95 = float(checkpoint.get("best_map_50_95", checkpoint.get("valid_map_50_95", float("-inf"))))

        print(
            f"Resumed from {args.resume} " f"(start_epoch={start_epoch}, best_valid_loss={best_valid_loss:.4f}, best_map_50_95={best_map_50_95:.4f})"
        )

    train_model = model
    if args.compile:
        if hasattr(torch, "compile"):
            train_model = torch.compile(model)
            print("Enabled torch.compile")
        else:
            print("torch.compile is not available in this environment; continue without compile.")

    scaler = torch.cuda.amp.GradScaler(enabled=amp_autocast_dtype == torch.float16)
    if amp_autocast_dtype is not None:
        print(f"Enabled AMP autocast ({args.amp})")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            train_model,
            train_loader,
            device,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
            amp_autocast_dtype=amp_autocast_dtype,
            scaler=scaler,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        valid_loss, valid_metrics = run_validation(
            train_model,
            valid_loader,
            processor,
            device,
            metric_backend=args.metric_backend,
            amp_autocast_dtype=amp_autocast_dtype,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        if scheduler is not None:
            scheduler.step()

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"mAP@0.5={valid_metrics['map_50']:.4f} "
            f"mAP@0.5:0.95={valid_metrics['map_50_95']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    "valid/map_50": valid_metrics["map_50"],
                    "valid/map_50_95": valid_metrics["map_50_95"],
                    "best/valid_loss": min(best_valid_loss, valid_loss),
                    "best/map_50_95": max(best_map_50_95, valid_metrics["map_50_95"]),
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_backbone": optimizer.param_groups[1]["lr"],
                }
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_map_50": valid_metrics["map_50"],
            "valid_map_50_95": valid_metrics["map_50_95"],
            "best_valid_loss": min(best_valid_loss, valid_loss),
            "best_map_50_95": max(best_map_50_95, valid_metrics["map_50_95"]),
            "model_name": args.model_name,
        }

        torch.save(checkpoint, run_dir / "last.pt")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(checkpoint, run_dir / "best_loss.pt")

        if valid_metrics["map_50_95"] > best_map_50_95:
            best_map_50_95 = valid_metrics["map_50_95"]
            torch.save(checkpoint, run_dir / "best_map.pt")

    model.save_pretrained(run_dir / "hf_model")
    processor.save_pretrained(run_dir / "hf_model")

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
