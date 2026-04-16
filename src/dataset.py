from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as T
from transformers import DetrImageProcessor


def load_categories(annotation_file: str | Path) -> list[dict[str, Any]]:
    annotation_path = Path(annotation_file)
    with annotation_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return sorted(data["categories"], key=lambda category: category["id"])


def build_label_mappings(annotation_file: str | Path) -> tuple[dict[int, str], dict[str, int]]:
    categories = load_categories(annotation_file)
    id2label = {index: category["name"] for index, category in enumerate(categories)}
    label2id = {label: index for index, label in id2label.items()}
    return id2label, label2id


class DetrCocoDataset(CocoDetection):
    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        processor: DetrImageProcessor,
        augment: bool = False,
    ) -> None:
        super().__init__(root=str(image_dir), annFile=str(annotation_file))
        self.processor = processor
        self.augment = augment
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        image, annotations = super().__getitem__(index)
        image_id = self.ids[index]

        if self.augment:
            image = self.color_jitter(image)

        target = {"image_id": image_id, "annotations": annotations}
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        # The annotation file uses category ids 1..10, but DETR expects 0-based labels.
        labels["class_labels"] = labels["class_labels"] - 1
        return pixel_values, labels


def build_collate_fn(processor: DetrImageProcessor):
    def collate_fn(batch: list[tuple[Any, dict[str, Any]]]) -> dict[str, Any]:
        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        max_height = max(image.shape[1] for image in pixel_values)
        max_width = max(image.shape[2] for image in pixel_values)

        padded_images = []
        pixel_masks = []

        for image in pixel_values:
            _, height, width = image.shape
            padded_images.append(F.pad(image, (0, max_width - width, 0, max_height - height)))

            pixel_mask = torch.zeros((max_height, max_width), dtype=torch.bool)
            pixel_mask[:height, :width] = True
            pixel_masks.append(pixel_mask)

        return {
            "pixel_values": torch.stack(padded_images),
            "pixel_mask": torch.stack(pixel_masks),
            "labels": labels,
        }

    return collate_fn


def create_dataloaders(
    train_image_dir: str | Path,
    train_annotation_file: str | Path,
    valid_image_dir: str | Path,
    valid_annotation_file: str | Path,
    processor: DetrImageProcessor,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> tuple[DataLoader, DataLoader]:
    collate_fn = build_collate_fn(processor)

    train_dataset = DetrCocoDataset(train_image_dir, train_annotation_file, processor, augment=augment)
    valid_dataset = DetrCocoDataset(valid_image_dir, valid_annotation_file, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader
