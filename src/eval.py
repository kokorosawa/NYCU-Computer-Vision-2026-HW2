from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.dataset import build_category_id_mappings, create_dataloaders
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
    parser.add_argument("--metric-backend", type=str, default="pycocotools", choices=["pycocotools", "custom"])
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


def run_validation_pycoco(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    processor: DetrImageProcessor,
    device: torch.device,
    threshold: float,
    contiguous_to_raw: dict[int, int],
) -> dict[str, float]:
    coco_gt = dataloader.dataset.coco
    detections: list[dict[str, float | int | list[float]]] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            target_sizes = torch.stack([label["orig_size"] for label in labels])
            results = processor.post_process_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
            )

            for label, result in zip(labels, results):
                image_id = int(label["image_id"].item())
                for score, class_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                    x_min, y_min, x_max, y_max = [float(value.item()) for value in box]
                    contiguous_class_id = int(class_id.item())
                    detections.append(
                        {
                            "image_id": image_id,
                            "category_id": contiguous_to_raw[contiguous_class_id],
                            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                            "score": float(score.item()),
                        }
                    )

    if not detections:
        return {"map_50": 0.0, "map_50_95": 0.0}

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "map_50_95": float(coco_eval.stats[0]),
        "map_50": float(coco_eval.stats[1]),
    }


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
        aug_color=False,
        aug_geom=False,
        color_prob=0.0,
        geom_prob=0.0,
        color_brightness=0.0,
        color_contrast=0.0,
        color_saturation=0.0,
        color_hue=0.0,
        geom_degrees=0.0,
        geom_translate=0.0,
        geom_scale_min=1.0,
        geom_scale_max=1.0,
    )
    _, contiguous_to_raw = build_category_id_mappings(valid_annotation_file)

    if args.metric_backend == "pycocotools":
        metrics = run_validation_pycoco(
            model,
            valid_loader,
            processor,
            device,
            threshold=args.threshold,
            contiguous_to_raw=contiguous_to_raw,
        )
        print(f"[pycocotools] mAP@0.5={metrics['map_50']:.4f} mAP@0.5:0.95={metrics['map_50_95']:.4f}")
    else:
        valid_loss, metrics = run_validation(model, valid_loader, processor, device, threshold=args.threshold)
        print(f"[custom] valid_loss={valid_loss:.4f} mAP@0.5={metrics['map_50']:.4f} mAP@0.5:0.95={metrics['map_50_95']:.4f}")


if __name__ == "__main__":
    main()
