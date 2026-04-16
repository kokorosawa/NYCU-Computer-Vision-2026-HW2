from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.dataset import create_dataloaders
from src.metrics import run_validation
from src.model import create_model, create_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DETR on validation set.")
    parser.add_argument("--data-root", type=Path, default=Path("nycu-hw2-data"))
    parser.add_argument("--valid-image-dir", type=Path, default=None)
    parser.add_argument("--valid-annotation-file", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/detr/hf_model"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a .pt checkpoint (optional).")
    parser.add_argument(
        "--label-annotation-file",
        type=Path,
        default=Path("nycu-hw2-data/train.json"),
        help="COCO json used to build label mapping when loading from checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model_and_processor(
    model_dir: Path,
    checkpoint: Path | None,
    label_annotation_file: Path,
    device: torch.device,
) -> tuple[DetrImageProcessor, DetrForObjectDetection]:
    if model_dir.is_dir():
        processor = DetrImageProcessor.from_pretrained(model_dir)
        model = DetrForObjectDetection.from_pretrained(model_dir)
    else:
        if checkpoint is None:
            raise FileNotFoundError(
                f"{model_dir} not found. Provide --checkpoint or ensure the hf_model directory exists."
            )
        processor = create_processor()
        model = create_model(label_annotation_file)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=False)

    model.to(device)
    model.eval()
    return processor, model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    valid_image_dir = args.valid_image_dir or args.data_root / "valid"
    valid_annotation_file = args.valid_annotation_file or args.data_root / "valid.json"

    processor, model = load_model_and_processor(
        args.model_dir,
        args.checkpoint,
        args.label_annotation_file,
        device,
    )

    _, valid_loader = create_dataloaders(
        train_image_dir=valid_image_dir,
        train_annotation_file=valid_annotation_file,
        valid_image_dir=valid_image_dir,
        valid_annotation_file=valid_annotation_file,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )

    valid_loss, metrics = run_validation(model, valid_loader, processor, device, threshold=args.threshold)
    print(f"valid_loss={valid_loss:.4f} mAP@0.5={metrics['map_50']:.4f} mAP@0.5:0.95={metrics['map_50_95']:.4f}")


if __name__ == "__main__":
    main()
