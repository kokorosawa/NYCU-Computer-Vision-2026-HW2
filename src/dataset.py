from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes
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
        aug_color: bool = True,
        aug_geom: bool = False,
        color_prob: float = 1.0,
        geom_prob: float = 0.0,
        color_brightness: float = 0.3,
        color_contrast: float = 0.3,
        color_saturation: float = 0.3,
        color_hue: float = 0.03,
        geom_degrees: float = 3.0,
        geom_translate: float = 0.02,
        geom_scale_min: float = 0.95,
        geom_scale_max: float = 1.05,
    ) -> None:
        super().__init__(root=str(image_dir), annFile=str(annotation_file))
        self.processor = processor
        self.augment = augment
        self.aug_color = aug_color
        self.aug_geom = aug_geom
        self.color_prob = color_prob
        self.geom_prob = geom_prob
        self.color_jitter = T.ColorJitter(
            brightness=color_brightness,
            contrast=color_contrast,
            saturation=color_saturation,
            hue=color_hue,
        )
        self.to_image = T.ToImage()
        self.to_dtype = T.ToDtype(torch.float32, scale=True)
        self.random_affine = T.RandomAffine(
            degrees=geom_degrees,
            translate=(geom_translate, geom_translate),
            scale=(geom_scale_min, geom_scale_max),
        )

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        image, annotations = super().__getitem__(index)
        image_id = self.ids[index]

        if self.augment and self.aug_color and random.random() < self.color_prob:
            image = self.color_jitter(image)

        if self.augment and self.aug_geom and random.random() < self.geom_prob:
            original_annotations = annotations
            image = self.to_image(image)
            image = self.to_dtype(image)
            original_image = image.clone()

            height, width = image.shape[-2], image.shape[-1]
            boxes = []
            categories = []
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                categories.append(ann["category_id"])

            if boxes:
                boxes_tensor = BoundingBoxes(
                    torch.tensor(boxes, dtype=torch.float32),
                    format="XYXY",
                    canvas_size=(height, width),
                )
                image, boxes_tensor = self.random_affine(image, boxes_tensor)
                boxes_xyxy = boxes_tensor.as_subclass(torch.Tensor)
                boxes_xyxy[:, 0::2].clamp_(0, image.shape[-1])
                boxes_xyxy[:, 1::2].clamp_(0, image.shape[-2])
                keep = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
                boxes_xyxy = boxes_xyxy[keep]
                categories = [cat for cat, flag in zip(categories, keep.tolist()) if flag]

                if len(categories) == 0:
                    image = original_image
                    annotations = original_annotations
                else:
                    annotations = []
                    for (x0, y0, x1, y1), cat in zip(boxes_xyxy.tolist(), categories):
                        annotations.append(
                            {
                                "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "category_id": cat,
                                "iscrowd": 0,
                                "area": (x1 - x0) * (y1 - y0),
                            }
                        )
            else:
                image = original_image
                annotations = original_annotations

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
    aug_color: bool,
    aug_geom: bool,
    color_prob: float,
    geom_prob: float,
    color_brightness: float,
    color_contrast: float,
    color_saturation: float,
    color_hue: float,
    geom_degrees: float,
    geom_translate: float,
    geom_scale_min: float,
    geom_scale_max: float,
) -> tuple[DataLoader, DataLoader]:
    collate_fn = build_collate_fn(processor)

    train_dataset = DetrCocoDataset(
        train_image_dir,
        train_annotation_file,
        processor,
        augment=augment,
        aug_color=aug_color,
        aug_geom=aug_geom,
        color_prob=color_prob,
        geom_prob=geom_prob,
        color_brightness=color_brightness,
        color_contrast=color_contrast,
        color_saturation=color_saturation,
        color_hue=color_hue,
        geom_degrees=geom_degrees,
        geom_translate=geom_translate,
        geom_scale_min=geom_scale_min,
        geom_scale_max=geom_scale_max,
    )
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
