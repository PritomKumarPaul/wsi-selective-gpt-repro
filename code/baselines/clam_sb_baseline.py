#!/usr/bin/env python3
"""End-to-end, slide-label-only CLAM-SB baseline for CAMELYON16.

Pipeline: mask-blind CLAM-style tissue segmentation -> uncapped CONCH-Z feature bags
-> CLAM-SB training on slide labels -> validation-selected checkpoint and
threshold -> one final test evaluation. No tumor masks/XML, MLP, GPT, or
patch-level supervision is used.

The existing mask-guided regression feature cache is deliberately NOT reused:
it contains mask-derived patch selection and is not a fair input for a weakly
supervised slide-label-only baseline. A fresh mask-blind cache is written one
slide at a time and is resumable. Cropped images are never saved.

Architecture follows the public CLAM-SB design: gated attention pooling with
slide-level cross entropy plus the CLAM top/bottom-attention instance-cluster
loss. Reference implementation: https://github.com/mahmoodlab/CLAM
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel


DEFAULT_DATA = Path.home() / 'data' / 'camelyon16_full'
DEFAULT_SPLIT_MANIFEST = Path.home() / 'wsi_patch_fraction_cache_v1' / 'manifest.json'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'results' / 'clam_sb_fullslide'
PATCH = 256
PATCH_STEP = 256
SEGMENTATION_DOWNSAMPLE = 64.0
SEGMENTATION_SATURATION_THRESHOLD = 8
SEGMENTATION_MEDIAN_KERNEL = 7
SEGMENTATION_CLOSE_KERNEL = 4
MIN_TISSUE_CONTOUR_PATCHES = 100
MIN_HOLE_PATCHES = 16
MAX_HOLES_PER_CONTOUR = 8
FEATURE_DIM = 768
FEATURE_STORE_DTYPE = 'float32'
TRAIN_SEED = 42
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.25
INSTANCE_K = 8
BAG_LOSS_WEIGHT = 0.70


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clam_tissue_segmentation(slide: openslide.OpenSlide) -> tuple[np.ndarray, float, float, list[np.ndarray]]:
    """CLAM-style HSV segmentation near 64x downsample with contour area filtering."""
    downsamples = np.asarray(slide.level_downsamples, dtype=np.float64)
    seg_level = int(np.argmin(np.abs(downsamples - SEGMENTATION_DOWNSAMPLE)))
    seg_w, seg_h = slide.level_dimensions[seg_level]
    seg_ds_x = slide.dimensions[0] / seg_w
    seg_ds_y = slide.dimensions[1] / seg_h
    rgb = np.asarray(slide.read_region((0, 0), seg_level, (seg_w, seg_h)).convert('RGB'))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = cv2.medianBlur(hsv[:, :, 1], SEGMENTATION_MEDIAN_KERNEL)
    _, binary = cv2.threshold(saturation, SEGMENTATION_SATURATION_THRESHOLD, 255,
                              cv2.THRESH_BINARY)
    close = np.ones((SEGMENTATION_CLOSE_KERNEL, SEGMENTATION_CLOSE_KERNEL), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close)

    # CLAM's a_t/a_h contour thresholds are expressed in equivalent 512x512
    # level-0 patches. Preserve foreground contours and their larger holes.
    ref_patch_area = (512.0 * 512.0) / (seg_ds_x * seg_ds_y)
    min_tissue_area = MIN_TISSUE_CONTOUR_PATCHES * ref_patch_area
    min_hole_area = MIN_HOLE_PATCHES * ref_patch_area
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    tissue = np.zeros_like(binary)
    foreground = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for idx, contour in enumerate(contours):
            if hierarchy[idx][3] != -1:
                continue
            child_ids = []
            child = hierarchy[idx][2]
            while child != -1:
                child_ids.append(child)
                child = hierarchy[child][0]
            contour_area = cv2.contourArea(contour)
            hole_areas = [cv2.contourArea(contours[j]) for j in child_ids]
            if contour_area - sum(hole_areas) < min_tissue_area:
                continue
            cv2.drawContours(tissue, contours, idx, 255, thickness=cv2.FILLED)
            kept_holes = sorted(child_ids, key=lambda j: cv2.contourArea(contours[j]),
                                reverse=True)[:MAX_HOLES_PER_CONTOUR]
            for hole_idx in kept_holes:
                if cv2.contourArea(contours[hole_idx]) > min_hole_area:
                    cv2.drawContours(tissue, contours, hole_idx, 0, thickness=cv2.FILLED)
            foreground.append(contour)
    return tissue, seg_ds_x, seg_ds_y, foreground


def sample_clam_patches(slide: openslide.OpenSlide) -> list[dict[str, Any]]:
    """All standard 256px, non-overlapping patch positions inside segmented tissue.

    There is no maximum patch count and no per-patch tissue-fraction threshold.
    Candidate positions use the CLAM ``four_pt`` idea: retain a tile if any of
    four interior sample points lies within a filtered tissue contour.
    """
    width, height = slide.dimensions
    mask, ds_x, ds_y, contours = clam_tissue_segmentation(slide)
    seg_h, seg_w = mask.shape
    sample_offsets = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
    seen: set[tuple[int, int]] = set()
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        x0 = max(0, int(math.floor(bx * ds_x)))
        y0 = max(0, int(math.floor(by * ds_y)))
        x1 = min(width, int(math.ceil((bx + bw) * ds_x)))
        y1 = min(height, int(math.ceil((by + bh) * ds_y)))
        xs = np.arange(x0, x1, PATCH_STEP, dtype=np.int64)
        ys = np.arange(y0, y1, PATCH_STEP, dtype=np.int64)
        if not len(xs) or not len(ys):
            continue
        grid_x, grid_y = np.meshgrid(xs, ys)
        inside = np.zeros(grid_x.shape, dtype=bool)
        for ox, oy in sample_offsets:
            sx = np.clip(np.rint((grid_x + ox * PATCH) / ds_x).astype(np.int64), 0, seg_w - 1)
            sy = np.clip(np.rint((grid_y + oy * PATCH) / ds_y).astype(np.int64), 0, seg_h - 1)
            inside |= tissue_sample(mask, sx, sy)
        for x, y in zip(grid_x[inside], grid_y[inside]):
            coord = (int(x), int(y))
            if coord not in seen:
                seen.add(coord)
                candidates.append({'x': coord[0], 'y': coord[1]})
    candidates.sort(key=lambda p: (p['y'], p['x']))
    return candidates


def tissue_sample(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mask[y, x] > 0


def make_rows(data: Path, split_manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the established slide split only; obtain class labels from reference.csv."""
    split_manifest = json.loads(split_manifest_path.read_text())
    reference = data / 'evaluation' / 'reference.csv'
    reference_rows = {r['image']: dict(r) for r in csv.DictReader(reference.open(newline=''))}
    train_val = []
    for item in split_manifest['slides']:
        if item['split'] not in {'train', 'validation'}:
            continue
        ref = reference_rows.get(item['image'])
        if ref is None:
            raise ValueError(f"Training/validation slide missing from reference.csv: {item['image']}")
        if ref['type'] not in {'normal', 'tumor'}:
            raise ValueError(f"Expected binary type for {item['image']}: {ref['type']}")
        if ref['type'] != item['type']:
            raise ValueError(f"Split manifest/reference label mismatch for {item['image']}")
        train_val.append({'image': item['image'], 'split': item['split'],
                          'label': 1 if ref['type'] == 'tumor' else 0,
                          'class_name': ref['type']})
    test = []
    for ref in reference_rows.values():
        if Path(ref['image']).stem.startswith('test_'):
            if ref['type'] not in {'normal', 'tumor'}:
                raise ValueError(f"Expected binary type for {ref['image']}: {ref['type']}")
            test.append({'image': ref['image'], 'split': 'test',
                         'label': 1 if ref['type'] == 'tumor' else 0,
                         'class_name': ref['type']})
    test.sort(key=lambda r: r['image'])
    counts = {s: Counter(r['class_name'] for r in train_val if r['split'] == s)
              for s in ('train', 'validation')}
    test_counts = Counter(r['class_name'] for r in test)
    if len([r for r in train_val if r['split'] == 'train']) != 200 or len(
            [r for r in train_val if r['split'] == 'validation']) != 50:
        raise ValueError('Expected the established 200 train / 50 validation split.')
    if len(test) != 129 or test_counts != {'normal': 80, 'tumor': 49}:
        raise ValueError(f'Expected 129 test slides (80 normal, 49 tumor); got {len(test)} {dict(test_counts)}')
    rows = train_val + test
    missing = [str(data / 'images' / r['image']) for r in rows
               if not (data / 'images' / r['image']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} slide images missing; first: {missing[0]}')
    return rows, {
        'train_slides': len([r for r in rows if r['split'] == 'train']),
        'train_classes': dict(counts['train']),
        'validation_slides': len([r for r in rows if r['split'] == 'validation']),
        'validation_classes': dict(counts['validation']),
        'test_slides': len(test), 'test_classes': dict(test_counts),
    }


def make_config(data: Path, model_dir: Path, split_manifest: Path,
                reference: Path) -> dict[str, Any]:
    return {
        'protocol': 'mask-blind CLAM-SB; slide-level binary labels only',
        'split_manifest': str(split_manifest),
        'split_manifest_sha256': sha256(split_manifest),
        'data_root': str(data), 'reference_sha256': sha256(reference),
        'encoder': {
            'model_dir': str(model_dir), 'model_config_sha256': sha256(model_dir / 'config.json'),
            'weights': 'local frozen CONCH-Z; FP32 inference; no quantization',
            'preprocessing': 'local encoder.eval_transform on each 256x256 level-0 RGB crop',
            'output_dim': FEATURE_DIM, 'cache_dtype': FEATURE_STORE_DTYPE,
        },
        'candidate_sampler': {
            'method': 'CLAM-style HSV tissue contours near 64x downsample; four-point patch inclusion',
            'segmentation': {
                'target_downsample': SEGMENTATION_DOWNSAMPLE,
                'saturation_threshold': SEGMENTATION_SATURATION_THRESHOLD,
                'median_blur_kernel': SEGMENTATION_MEDIAN_KERNEL,
                'morphological_close_kernel': SEGMENTATION_CLOSE_KERNEL,
                'minimum_foreground_area_equivalent_512px_patches': MIN_TISSUE_CONTOUR_PATCHES,
                'minimum_hole_area_equivalent_512px_patches': MIN_HOLE_PATCHES,
                'maximum_holes_per_contour': MAX_HOLES_PER_CONTOUR,
            },
            'grid': f'{PATCH}x{PATCH} level-0 patches; stride {PATCH_STEP}; non-overlapping',
            'edge_rule': 'CLAM-style padding permitted for edge patches whose interior samples hit tissue',
            'blank_patch_filter': 'exclude mean HSV saturation <15 or mean RGB <50, matching CLAM defaults',
            'tissue_fraction_threshold': None,
            'candidate_cap': None,
            'mask_policy': 'tumor masks and XML annotations are never opened',
        },
        'training': {
            'architecture': 'CLAM-SB gated attention; 768->512 embedding; attention dim 256',
            'loss': '0.70 slide cross-entropy + 0.30 top/bottom-k instance cross-entropy',
            'instance_k': INSTANCE_K, 'dropout': DROPOUT,
            'class_balance': 'inverse-frequency weighted slide sampling with replacement',
            'optimizer': 'Adam', 'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY, 'max_epochs': MAX_EPOCHS,
            'early_stopping_patience': PATIENCE, 'checkpoint_selection': 'validation AUROC, then validation loss',
            'seed': TRAIN_SEED,
        },
        'threshold': 'validation-only threshold maximizing balanced accuracy; ties closest to 0.5',
    }


def safe_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cache_path(output: Path, row: dict[str, Any]) -> Path:
    return output / 'features' / f"{row['split']}__{Path(row['image']).stem}.npz"


def read_feature_cache(path: Path, config_sha: str, row: dict[str, Any]) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata'].item()))
        features = z['features']
        xy = z['xy']
    expected = {'config_sha256': config_sha, 'image': row['image'], 'split': row['split'],
                'label': row['label'], 'class_name': row['class_name']}
    if metadata != expected:
        raise ValueError(f'Incompatible feature cache {path}; choose a fresh --output folder.')
    if (features.ndim != 2 or features.shape[1] != FEATURE_DIM or not len(features)
            or xy.shape != (len(features), 2) or not np.isfinite(features).all()):
        raise ValueError(f'Corrupt or malformed feature cache: {path}')
    return {'features': features, 'xy': xy}


def disk_guard(output: Path, min_free_gb: float) -> None:
    free = shutil.disk_usage(output).free
    reserve = min_free_gb * 1_000_000_000
    if free < reserve:
        raise RuntimeError(f'Free space fell below --min-free-gb={min_free_gb}; stopping safely.')


def encode_slide(slide_path: Path, row: dict[str, Any], encoder: Any,
                 device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    with openslide.OpenSlide(str(slide_path)) as slide:
        pool = sample_clam_patches(slide)
        if not pool:
            raise ValueError(f"No patches were produced by CLAM-style tissue segmentation for {row['image']}.")
        features = []
        kept_xy = []
        for start in range(0, len(pool), batch_size):
            batch = pool[start:start + batch_size]
            tensors = []
            batch_xy = []
            for p in batch:
                crop = slide.read_region((p['x'], p['y']), 0, (PATCH, PATCH)).convert('RGB')
                rgb = np.asarray(crop)
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                # CLAM's default optional blank-patch screening: remove nearly
                # white and nearly black tiles after contour-based tissue sampling.
                if float(hsv[:, :, 1].mean()) < 15.0 or float(rgb.mean()) < 50.0:
                    continue
                tensors.append(encoder.eval_transform(crop))
                batch_xy.append((p['x'], p['y']))
            if not tensors:
                continue
            images = torch.stack(tensors).to(device)
            with torch.inference_mode():
                encoded = encoder.encode_image(images).float()
            if encoded.ndim != 2 or encoded.shape[1] != FEATURE_DIM:
                raise ValueError(f"Expected {FEATURE_DIM}-D CONCH-Z embeddings; got {tuple(encoded.shape)}")
            features.append(encoded.cpu().numpy())
            kept_xy.extend(batch_xy)
            done = min(start + len(batch), len(pool))
            logging.info('%s encoded batch through candidate %d/%d; retained %d patches',
                         Path(row['image']).stem, done, len(pool), len(kept_xy))
        if not features:
            raise ValueError(f"All tissue candidates were blank-filtered for {row['image']}.")
        return {
            'features': np.concatenate(features).astype(np.float32),
            'xy': np.asarray(kept_xy, dtype=np.int32),
        }


class CLAMSingleBranch(nn.Module):
    """Gated single-branch CLAM with binary bag and instance classifiers."""
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 512,
                 attention_dim: int = 256, dropout: float = DROPOUT):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                       nn.Dropout(dropout))
        self.attention_tanh = nn.Sequential(nn.Linear(hidden_dim, attention_dim),
                                            nn.Tanh(), nn.Dropout(dropout))
        self.attention_sigmoid = nn.Sequential(nn.Linear(hidden_dim, attention_dim),
                                               nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_score = nn.Linear(attention_dim, 1)
        self.slide_classifier = nn.Linear(hidden_dim, 2)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 2), nn.Linear(hidden_dim, 2)])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.embedding(features)
        gated = self.attention_tanh(hidden) * self.attention_sigmoid(hidden)
        raw_attention = self.attention_score(gated).squeeze(-1)
        attention = F.softmax(raw_attention, dim=0)
        pooled = torch.sum(attention[:, None] * hidden, dim=0, keepdim=True)
        return self.slide_classifier(pooled), attention, hidden

    def instance_loss(self, attention: torch.Tensor, hidden: torch.Tensor,
                      slide_label: int, k: int) -> torch.Tensor:
        """CLAM in-class top-k positive and bottom-k negative pseudo-instance loss."""
        order = torch.argsort(attention)
        bottom = order[:k]
        top = order[-k:]
        selected = torch.cat([hidden[top], hidden[bottom]], dim=0)
        targets = torch.cat([
            torch.ones(k, dtype=torch.long, device=hidden.device),
            torch.zeros(k, dtype=torch.long, device=hidden.device),
        ])
        logits = self.instance_classifiers[slide_label](selected)
        return F.cross_entropy(logits, targets)


def slide_forward(model: CLAMSingleBranch, features: np.ndarray, label: int | None,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    logits, attention, hidden = model(x)
    inst_loss = None
    if label is not None:
        k = min(INSTANCE_K, max(1, len(features) // 2))
        inst_loss = model.instance_loss(attention, hidden, label, k)
    return logits, attention, inst_loss


def load_bags(rows: list[dict[str, Any]], output: Path, config_sha: str) -> list[dict[str, Any]]:
    bags = []
    for row in rows:
        path = cache_path(output, row)
        cache = read_feature_cache(path, config_sha, row)
        if cache is None:
            raise FileNotFoundError(f'Missing feature cache {path}; complete feature preparation first.')
        # Validate each cache but keep only its path in memory. Full WSI bags
        # can be large, so training/evaluation loads one slide at a time.
        bags.append({**row, 'feature_path': path, 'config_sha256': config_sha})
    return bags


def get_bag_arrays(bag: dict[str, Any]) -> dict[str, np.ndarray]:
    cache = read_feature_cache(bag['feature_path'], bag['config_sha256'], bag)
    if cache is None:
        raise FileNotFoundError(bag['feature_path'])
    return cache


def evaluate(model: CLAMSingleBranch, bags: list[dict[str, Any]],
             device: torch.device) -> tuple[list[dict[str, Any]], float]:
    model.eval()
    records = []
    losses = []
    with torch.inference_mode():
        for bag in bags:
            arrays = get_bag_arrays(bag)
            logits, attention, _ = slide_forward(model, arrays['features'], None, device)
            label_tensor = torch.tensor([bag['label']], device=device)
            losses.append(float(F.cross_entropy(logits, label_tensor).item()))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            order = torch.argsort(attention, descending=True)[:8].cpu().numpy()
            top_coords = [{'attention_rank': i + 1,
                           'x_level0': int(arrays['xy'][idx, 0]),
                           'y_level0': int(arrays['xy'][idx, 1]),
                           'attention': float(attention[idx].item())}
                          for i, idx in enumerate(order)]
            records.append({
                'slide': Path(bag['image']).stem, 'image': bag['image'],
                'split': bag['split'], 'truth': 'TUMOR' if bag['label'] else 'NORMAL',
                'tumor_probability': float(probabilities[1]),
                'normal_probability': float(probabilities[0]),
                'candidate_count': int(len(arrays['features'])),
                'top8_attention_coordinates': top_coords,
            })
            del arrays
    return records, float(np.mean(losses)) if losses else math.nan


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    unique = np.unique(scores)
    candidates = np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2, [1.0]))
    scored = []
    for threshold in np.unique(candidates):
        pred = scores >= threshold
        positive = labels == 1
        negative = labels == 0
        sensitivity = float(np.mean(pred[positive]))
        specificity = float(np.mean(~pred[negative]))
        balanced = (sensitivity + specificity) / 2
        scored.append((balanced, -abs(float(threshold) - 0.5), float(threshold),
                       sensitivity, specificity))
    best = max(scored)
    return best[2], {'balanced_accuracy': best[0], 'sensitivity': best[3],
                     'specificity': best[4]}


def metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y = np.asarray([1 if r['truth'] == 'TUMOR' else 0 for r in records], dtype=np.int64)
    p = np.asarray([r['tumor_probability'] for r in records], dtype=np.float64)
    pred = p >= threshold
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tp = int(np.sum((y == 1) & pred))
    return {
        'slides': len(records), 'auroc': float(roc_auc_score(y, p)),
        'accuracy': float((tn + tp) / len(y)),
        'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
        'specificity': float(tn / (tn + fp)) if tn + fp else None,
        'precision_tumor': float(tp / (tp + fp)) if tp + fp else None,
        'threshold': float(threshold),
        'confusion_matrix_truth_rows_prediction_columns_NORMAL_TUMOR': [[tn, fp], [fn, tp]],
    }


def save_predictions(path: Path, records: list[dict[str, Any]], threshold: float) -> None:
    rows = []
    for r in records:
        row = dict(r)
        row['prediction'] = 'TUMOR' if r['tumor_probability'] >= threshold else 'NORMAL'
        row['correct'] = row['prediction'] == row['truth']
        rows.append(row)
    atomic_csv(path, rows, ['slide', 'image', 'split', 'truth', 'prediction', 'correct',
                            'tumor_probability', 'normal_probability', 'candidate_count',
                            'top8_attention_coordinates'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--split-manifest', type=Path, default=DEFAULT_SPLIT_MANIFEST,
                        help='Read for the established train/validation slide split only; its patch cache is not reused.')
    parser.add_argument('--model', type=Path, default=None,
                        help='Local CONCH-Z folder; defaults to model_dir in split manifest.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--min-free-gb', type=float, default=10.0,
                        help='Stop before the filesystem drops below this reserve; there is no candidate/cache-size cap.')
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    args.split_manifest = args.split_manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error('Output must remain inside the organized WSI project folder.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
        parser.error('CUDA requested but unavailable; choose --device cpu to run on CPU.')
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable.')

    split_manifest = json.loads(args.split_manifest.read_text())
    model_dir = (args.model.expanduser().resolve() if args.model
                 else Path(split_manifest['config']['model_dir']))
    if not (model_dir / 'config.json').is_file():
        raise FileNotFoundError(f'Local CONCH-Z config not found: {model_dir / "config.json"}')
    reference = args.data / 'evaluation' / 'reference.csv'
    if not reference.is_file():
        raise FileNotFoundError(reference)
    rows, split_summary = make_rows(args.data, args.split_manifest)
    config = make_config(args.data, model_dir, args.split_manifest, reference)
    config_path = args.output / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f'Output cache uses a different configuration: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        atomic_json(config_path, config)
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    log_handlers = [logging.StreamHandler(sys.stdout),
                    logging.FileHandler(args.output / 'run.log')]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=log_handlers, force=True)
    atomic_json(args.output / 'split_summary.json', split_summary)
    logging.info('Splits: %s', json.dumps(split_summary, sort_keys=True))
    logging.info('Sampler: CLAM-style contour segmentation, 256px patches, no candidate cap or patch tissue-fraction cutoff; masks not read.')
    logging.info('Feature cache: frozen FP32 CONCH-Z embeddings stored as FP32; no crops saved.')

    device = torch.device(args.device)
    missing = [r for r in rows if read_feature_cache(cache_path(args.output, r), config_sha, r) is None]
    if missing:
        from transformers import AutoModel
        logging.info('Loading local CONCH-Z model from %s on %s', model_dir, device)
        encoder = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                            local_files_only=True).eval().float().to(device)
        if not hasattr(encoder, 'eval_transform') or not hasattr(encoder, 'encode_image'):
            raise AttributeError('Loaded model lacks CONCH-Z eval_transform/encode_image API.')
        for index, row in enumerate(missing, 1):
            start = time.perf_counter()
            disk_guard(args.output, args.min_free_gb)
            item = encode_slide(args.data / 'images' / row['image'], row, encoder,
                                device, args.batch_size)
            metadata = {'config_sha256': config_sha, 'image': row['image'],
                        'split': row['split'], 'label': row['label'],
                        'class_name': row['class_name']}
            safe_save_npz(cache_path(args.output, row),
                          features=item['features'], xy=item['xy'],
                          metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
            logging.info('cached %s/%s %s patches=%d elapsed=%.1fs', index,
                         len(missing), Path(row['image']).stem,
                         len(item['features']), time.perf_counter() - start)
        del encoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        logging.info('All %d slide feature bags already cached; skipping CONCH-Z extraction.', len(rows))

    bags = load_bags(rows, args.output, config_sha)
    train_bags = [b for b in bags if b['split'] == 'train']
    val_bags = [b for b in bags if b['split'] == 'validation']
    test_bags = [b for b in bags if b['split'] == 'test']
    if len(train_bags) != 200 or len(val_bags) != 50 or len(test_bags) != 129:
        raise ValueError('Unexpected split sizes after feature-cache load.')

    seed_everything(TRAIN_SEED)
    model = CLAMSingleBranch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    labels = np.asarray([b['label'] for b in train_bags], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    slide_weights = np.asarray([1.0 / class_counts[y] for y in labels], dtype=np.float64)
    slide_weights /= slide_weights.sum()
    best_auc, best_val_loss, best_epoch, stale = -math.inf, math.inf, 0, 0
    history = []
    best_path = args.output / 'clam_sb_best.pt'
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(TRAIN_SEED + epoch)
        epoch_indices = rng.choice(len(train_bags), size=len(train_bags),
                                   replace=True, p=slide_weights)
        train_losses = []
        for bag_idx in epoch_indices:
            bag = train_bags[int(bag_idx)]
            arrays = get_bag_arrays(bag)
            optimizer.zero_grad(set_to_none=True)
            logits, _, inst_loss = slide_forward(model, arrays['features'], bag['label'], device)
            label_tensor = torch.tensor([bag['label']], dtype=torch.long, device=device)
            bag_loss = F.cross_entropy(logits, label_tensor)
            loss = BAG_LOSS_WEIGHT * bag_loss + (1.0 - BAG_LOSS_WEIGHT) * inst_loss
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            del arrays

        val_records, val_loss = evaluate(model, val_bags, device)
        val_labels = np.asarray([1 if r['truth'] == 'TUMOR' else 0 for r in val_records])
        val_scores = np.asarray([r['tumor_probability'] for r in val_records])
        val_auc = float(roc_auc_score(val_labels, val_scores))
        row = {'epoch': epoch, 'train_loss': float(np.mean(train_losses)),
               'validation_loss': val_loss, 'validation_auroc': val_auc}
        history.append(row)
        atomic_csv(args.output / 'training_history.csv', history,
                   ['epoch', 'train_loss', 'validation_loss', 'validation_auroc'])
        improved = (val_auc > best_auc + 1e-8
                    or (abs(val_auc - best_auc) <= 1e-8 and val_loss < best_val_loss))
        if improved:
            best_auc, best_val_loss, best_epoch, stale = val_auc, val_loss, epoch, 0
            tmp = best_path.with_suffix('.pt.tmp')
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                        'validation_auroc': val_auc, 'validation_loss': val_loss,
                        'config_sha256': config_sha}, tmp)
            os.replace(tmp, best_path)
        else:
            stale += 1
        logging.info('epoch %02d/%02d train_loss=%.5f val_loss=%.5f val_AUROC=%.4f best=%02d stale=%d',
                     epoch, MAX_EPOCHS, row['train_loss'], val_loss, val_auc,
                     best_epoch, stale)
        if stale >= PATIENCE:
            logging.info('Early stopping after %d epochs without validation improvement.', PATIENCE)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    val_records, final_val_loss = evaluate(model, val_bags, device)
    val_labels = np.asarray([1 if r['truth'] == 'TUMOR' else 0 for r in val_records])
    val_scores = np.asarray([r['tumor_probability'] for r in val_records])
    threshold, threshold_stats = choose_threshold(val_labels, val_scores)
    val_metrics = metrics(val_records, threshold)
    val_metrics['loss'] = final_val_loss
    val_metrics['threshold_selection'] = threshold_stats
    save_predictions(args.output / 'validation_predictions.csv', val_records, threshold)

    # This is the single test evaluation. Threshold and checkpoint are already fixed from validation.
    test_records, test_loss = evaluate(model, test_bags, device)
    test_metrics = metrics(test_records, threshold)
    test_metrics['loss'] = test_loss
    save_predictions(args.output / 'test_predictions.csv', test_records, threshold)
    summary = {
        'protocol': config,
        'split_summary': split_summary,
        'training': {'best_epoch': int(checkpoint['epoch']),
                     'epochs_run': len(history), 'best_validation_auroc': float(checkpoint['validation_auroc']),
                     'best_validation_loss': float(checkpoint['validation_loss']),
                     'checkpoint': str(best_path)},
        'validation': val_metrics,
        'test': test_metrics,
        'caution': 'The official test set has prior exploratory use elsewhere in this project; '
                   'treat this CLAM test result as exploratory, not an untouched confirmatory result.',
    }
    atomic_json(args.output / 'summary.json', summary)
    logging.info('Validation: %s', json.dumps(val_metrics, sort_keys=True))
    logging.info('Test: %s', json.dumps(test_metrics, sort_keys=True))
    logging.info('CLAM-SB run complete; no GPT API calls were made.')


if __name__ == '__main__':
    main()
