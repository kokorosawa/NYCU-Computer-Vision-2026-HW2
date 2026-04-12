from __future__ import annotations

from pathlib import Path

from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

from src.dataset import build_label_mappings


DEFAULT_MODEL_NAME = "facebook/detr-resnet-50"


def create_processor(model_name: str = DEFAULT_MODEL_NAME) -> DetrImageProcessor:
    return DetrImageProcessor.from_pretrained(model_name)


def create_model(
    annotation_file: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> DetrForObjectDetection:
    id2label, label2id = build_label_mappings(annotation_file)

    config = DetrConfig.from_pretrained(model_name)
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id

    return DetrForObjectDetection.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
