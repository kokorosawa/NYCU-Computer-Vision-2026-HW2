from __future__ import annotations

import argparse
from pathlib import Path

import torch
import wandb
from tqdm.auto import tqdm
from src.dataset import create_dataloaders
from src.metrics import move_labels_to_device, run_validation
from src.model import create_model, create_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DETR on the homework dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("nycu-hw2-data"))
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--train-annotation-file", type=Path, default=None)
    parser.add_argument("--valid-image-dir", type=Path, default=None)
    parser.add_argument("--valid-annotation-file", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/detr"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", type=str, default="dlcv-hw2-detr")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def init_wandb(args: argparse.Namespace, paths: dict[str, Path]):
    if args.wandb_mode == "disabled":
        return None

    config = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_workers": args.num_workers,
        "device": args.device,
        "train_image_dir": str(paths["train_image_dir"]),
        "train_annotation_file": str(paths["train_annotation_file"]),
        "valid_image_dir": str(paths["valid_image_dir"]),
        "valid_annotation_file": str(paths["valid_annotation_file"]),
        "output_dir": str(args.output_dir),
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

        with torch.set_grad_enabled(is_training):
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss detected at step {step}: {loss.item()}")

            if is_training:
                optimizer.zero_grad(set_to_none=True)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, paths)

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
    )

    device = torch.device(args.device)
    model.to(device)
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

    best_valid_loss = float("inf")
    best_map_50_95 = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        valid_loss, valid_metrics = run_validation(
            model,
            valid_loader,
            processor,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

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
                }
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_map_50": valid_metrics["map_50"],
            "valid_map_50_95": valid_metrics["map_50_95"],
            "model_name": args.model_name,
        }

        torch.save(checkpoint, args.output_dir / "last.pt")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(checkpoint, args.output_dir / "best_loss.pt")

        if valid_metrics["map_50_95"] > best_map_50_95:
            best_map_50_95 = valid_metrics["map_50_95"]
            torch.save(checkpoint, args.output_dir / "best_map.pt")

    model.save_pretrained(args.output_dir / "hf_model")
    processor.save_pretrained(args.output_dir / "hf_model")

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
