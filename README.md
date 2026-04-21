# DLCV HW2 - Deformable DETR

## Introduction

This repository contains our implementation for NYCU DLCV HW2 object detection.
The core model is **Deformable DETR** (`SenseTime/deformable-detr`) with a custom training and evaluation pipeline.

Main scripts are in `src/`:
- `src/train.py`: model training
- `src/eval.py`: validation / mAP evaluation
- `src/inference.py`: test-time inference and JSON prediction export

## Environment Setup

### 1. Requirements

- Python >= 3.11
- CUDA GPU is recommended for training/inference speed

### 2. Install dependencies (uv)

```bash
uv sync
```

If you prefer `pip`, install packages listed in `pyproject.toml`.

### 3. Dataset

Default dataset path is:

```text
nycu-hw2-data/
  train/
  valid/
  train.json
  valid.json
```

You can also override paths with command-line arguments in `train.py` and `eval.py`.

## Usage

### 1. Train

```bash
uv run python src/train.py \
  --data-root nycu-hw2-data \
  --epochs 20 \
  --batch-size 4 \
  --output-dir checkpoints/detr
```

Useful options:
- `--augment --aug-color --aug-geom`: enable data augmentation
- `--amp bf16` or `--amp fp16`: mixed precision on CUDA
- `--resume <checkpoint.pt>`: resume training

### 2. Evaluate

```bash
uv run python src/eval.py \
  --data-root nycu-hw2-data \
  --model-dir checkpoints/detr/hf_model \
  --metric-backend pycocotools \
  --batch-size 8
```

If `hf_model` is not available, use:

```bash
uv run python src/eval.py \
  --data-root nycu-hw2-data \
  --model-dir checkpoints/detr/hf_model \
  --checkpoint checkpoints/detr/best_map.pt \
  --label-annotation-file nycu-hw2-data/train.json
```

### 3. Inference

```bash
uv run python src/inference.py \
  --input nycu-hw2-data/valid \
  --annotation-file nycu-hw2-data/valid.json \
  --model-dir checkpoints/detr/hf_model \
  --output pred.json \
  --threshold 0.5
```

## Performance Snapshot

The following snapshot is from the **latest successful Deformable DETR run with complete metrics**:
`checkpoints/deformable_detr/20260420_140057/best_map.pt`.

- Epoch: `13`
- Validation Loss: `1.0145`
- mAP@0.5: `0.5785`
- mAP@0.5:0.95: `0.2993`

You can recompute metrics anytime by running:

```bash
uv run python src/eval.py --data-root nycu-hw2-data --model-dir checkpoints/detr/hf_model
```
