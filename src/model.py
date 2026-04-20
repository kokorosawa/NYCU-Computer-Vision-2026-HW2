from __future__ import annotations

from pathlib import Path

from torchvision.models import ResNet50_Weights, resnet50
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

    model = DetrForObjectDetection(config)

    imagenet_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    missing_keys, unexpected_keys = model.model.backbone.model.load_state_dict(
        imagenet_resnet.state_dict(),
        strict=False,
    )
    unexpected_set = set(unexpected_keys)
    if missing_keys or unexpected_set != {"fc.weight", "fc.bias"}:
        raise RuntimeError(
            "Failed to load pretrained ResNet-50 backbone cleanly. "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    return model
