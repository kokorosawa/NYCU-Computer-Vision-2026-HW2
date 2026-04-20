from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor

from src.dataset import build_category_id_mappings
from src.model import create_model, create_processor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Deformable DETR inference on one image or a directory.")
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/detr/hf_model"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a .pt checkpoint (optional).")
    parser.add_argument(
        "--label-annotation-file",
        type=Path,
        default=Path("nycu-hw2-data/train.json"),
        help="COCO json used to build label mapping when loading from checkpoint.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to an image or a directory of images.")
    parser.add_argument("--annotation-file", type=Path, default=None, help="COCO json used to map file_name to image_id.")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    return sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_model_and_processor(
    model_dir: Path,
    checkpoint: Path | None,
    label_annotation_file: Path,
    device: torch.device,
) -> tuple[DeformableDetrImageProcessor, DeformableDetrForObjectDetection]:
    if model_dir.is_dir():
        processor = DeformableDetrImageProcessor.from_pretrained(model_dir)
        model = DeformableDetrForObjectDetection.from_pretrained(model_dir)
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


def load_image_id_mapping(annotation_file: Path) -> dict[str, int]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    return {image["file_name"]: image["id"] for image in data["images"]}

def infer_image_id_from_filename(image_path: Path) -> int | None:
    stem = image_path.stem
    if stem.isdigit():
        return int(stem)
    return None


def predict_batch(
    image_paths: list[Path],
    processor: DeformableDetrImageProcessor,
    model: DeformableDetrForObjectDetection,
    device: torch.device,
    threshold: float,
    image_ids: list[int | None],
    contiguous_to_raw: dict[int, int],
) -> list[dict[str, object]]:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1] for image in images], device=device)
    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )

    predictions: list[dict[str, object]] = []
    for image_id, result in zip(image_ids, results):
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            class_id = contiguous_to_raw[int(label.item())]
            x_min, y_min, x_max, y_max = [float(value.item()) for value in box]

            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": class_id,
                    "score": float(score.item()),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                }
            )

    return predictions


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    processor, model = load_model_and_processor(
        args.model_dir,
        args.checkpoint,
        args.label_annotation_file,
        device,
    )
    _, contiguous_to_raw = build_category_id_mappings(args.label_annotation_file)
    image_paths = list_images(args.input)
    image_id_mapping = load_image_id_mapping(args.annotation_file) if args.annotation_file is not None else None

    results = []
    batch_size = max(1, args.batch_size)
    for start in tqdm(range(0, len(image_paths), batch_size), desc="inference"):
        batch_paths = image_paths[start : start + batch_size]
        batch_ids = []
        for image_path in batch_paths:
            if image_id_mapping is not None:
                batch_ids.append(image_id_mapping.get(image_path.name))
            else:
                batch_ids.append(infer_image_id_from_filename(image_path))

        results.extend(
            predict_batch(
                image_paths=batch_paths,
                processor=processor,
                model=model,
                device=device,
                threshold=args.threshold,
                image_ids=batch_ids,
                contiguous_to_raw=contiguous_to_raw,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
