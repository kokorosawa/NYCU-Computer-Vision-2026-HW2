from __future__ import annotations

from typing import Any

import torch
from pycocotools.cocoeval import COCOeval
from tqdm.auto import tqdm
from torchvision.ops import box_iou


def move_labels_to_device(
    labels: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in label.items()} for label in labels]


def calculate_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0

    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    ap = 0.0
    for index in range(1, len(recalls)):
        if recalls[index] != recalls[index - 1]:
            ap += (recalls[index] - recalls[index - 1]) * precisions[index]

    return ap


def build_ground_truth_index(dataset) -> tuple[dict[int, dict[int, list[list[float]]]], dict[int, int]]:
    ground_truths: dict[int, dict[int, list[list[float]]]] = {}
    gt_counts: dict[int, int] = {}
    raw_to_contiguous = getattr(dataset, "raw_to_contiguous", None)

    for image_id in dataset.ids:
        annotations = dataset.coco.imgToAnns[image_id]
        for annotation in annotations:
            raw_category_id = int(annotation["category_id"])
            if raw_to_contiguous is not None:
                class_id = raw_to_contiguous[raw_category_id]
            else:
                class_id = raw_category_id - 1
            x, y, width, height = annotation["bbox"]
            box = [x, y, x + width, y + height]

            ground_truths.setdefault(class_id, {}).setdefault(image_id, []).append(box)
            gt_counts[class_id] = gt_counts.get(class_id, 0) + 1

    return ground_truths, gt_counts


def evaluate_map(
    predictions: list[dict[str, float | int | list[float]]],
    dataset,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * step for step in range(10)]

    ground_truths, gt_counts = build_ground_truth_index(dataset)
    class_ids = sorted(ground_truths.keys())
    ap_by_threshold: dict[float, list[float]] = {threshold: [] for threshold in iou_thresholds}

    for threshold in iou_thresholds:
        for class_id in class_ids:
            class_predictions = [prediction for prediction in predictions if prediction["label_id"] == class_id]
            class_predictions.sort(key=lambda prediction: float(prediction["score"]), reverse=True)

            matched = {
                image_id: [False] * len(boxes)
                for image_id, boxes in ground_truths.get(class_id, {}).items()
            }

            true_positives: list[float] = []
            false_positives: list[float] = []

            for prediction in class_predictions:
                image_id = int(prediction["image_id"])
                predicted_box = torch.tensor([prediction["bbox_xyxy"]], dtype=torch.float32)
                gt_boxes = ground_truths.get(class_id, {}).get(image_id, [])

                if not gt_boxes:
                    true_positives.append(0.0)
                    false_positives.append(1.0)
                    continue

                gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
                ious = box_iou(predicted_box, gt_tensor)[0]
                best_iou, best_index = torch.max(ious, dim=0)

                if best_iou.item() >= threshold and not matched[image_id][best_index.item()]:
                    matched[image_id][best_index.item()] = True
                    true_positives.append(1.0)
                    false_positives.append(0.0)
                else:
                    true_positives.append(0.0)
                    false_positives.append(1.0)

            if gt_counts.get(class_id, 0) == 0:
                continue

            cumulative_tp = 0.0
            cumulative_fp = 0.0
            recalls: list[float] = []
            precisions: list[float] = []

            for tp, fp in zip(true_positives, false_positives):
                cumulative_tp += tp
                cumulative_fp += fp

                recalls.append(cumulative_tp / gt_counts[class_id])
                precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1e-8))

            ap_by_threshold[threshold].append(calculate_ap(recalls, precisions))

    map_50 = sum(ap_by_threshold[0.5]) / max(len(ap_by_threshold[0.5]), 1)
    all_ap_values = [ap for ap_list in ap_by_threshold.values() for ap in ap_list]
    map_50_95 = sum(all_ap_values) / max(len(all_ap_values), 1)

    return {
        "map_50": map_50,
        "map_50_95": map_50_95,
    }


def evaluate_map_pycocotools(
    predictions: list[dict[str, float | int | list[float]]],
    dataset,
) -> dict[str, float]:
    if not predictions:
        return {"map_50": 0.0, "map_50_95": 0.0}

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "map_50_95": float(coco_eval.stats[0]),
        "map_50": float(coco_eval.stats[1]),
    }


def run_validation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    processor: Any,
    device: torch.device,
    threshold: float = 0.0,
    metric_backend: str = "pycocotools",
    amp_autocast_dtype: torch.dtype | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    predictions_custom: list[dict[str, float | int | list[float]]] = []
    predictions_pycoco: list[dict[str, float | int | list[float]]] = []
    contiguous_to_raw = getattr(dataloader.dataset, "contiguous_to_raw", None)
    epoch_prefix = f"Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "Epoch"
    progress = tqdm(dataloader, desc=f"{epoch_prefix} [valid]", leave=False)

    with torch.no_grad():
        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = move_labels_to_device(batch["labels"], device)
            amp_enabled = amp_autocast_dtype is not None and device.type == "cuda"

            with torch.autocast(device_type="cuda", dtype=amp_autocast_dtype, enabled=amp_enabled):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            total_loss += outputs.loss.item()
            progress.set_postfix(loss=f"{outputs.loss.item():.4f}", avg=f"{total_loss / step:.4f}")

            target_sizes = torch.stack([label["orig_size"] for label in labels])
            processed_results = processor.post_process_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
            )

            for label, result in zip(labels, processed_results):
                image_id = int(label["image_id"].item())

                for score, class_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                    contiguous_class_id = int(class_id.item())
                    box_xyxy = [float(value.item()) for value in box]
                    if metric_backend == "pycocotools":
                        x_min, y_min, x_max, y_max = box_xyxy
                        if contiguous_to_raw is not None:
                            raw_category_id = contiguous_to_raw[contiguous_class_id]
                        else:
                            raw_category_id = contiguous_class_id + 1
                        predictions_pycoco.append(
                            {
                                "image_id": image_id,
                                "category_id": raw_category_id,
                                "score": float(score.item()),
                                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                            }
                        )
                    else:
                        predictions_custom.append(
                            {
                                "image_id": image_id,
                                "label_id": contiguous_class_id,
                                "score": float(score.item()),
                                "bbox_xyxy": box_xyxy,
                            }
                        )

    valid_loss = total_loss / max(len(dataloader), 1)
    if metric_backend == "pycocotools":
        metrics = evaluate_map_pycocotools(predictions_pycoco, dataloader.dataset)
    else:
        metrics = evaluate_map(predictions_custom, dataloader.dataset)
    return valid_loss, metrics
