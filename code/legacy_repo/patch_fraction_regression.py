#!/usr/bin/env python3
"""Predict native tumor-mask coverage from frozen CONCH-Z patch features.

plan: preview the fixed training/validation split and storage upper bound.
prepare: sample crops, encode locally, checkpoint one compact NPZ per slide.
fit: fit a coverage regressor and evaluate held-out slides using cached features.

No downloads or cloud calls. Images and masks remain in the existing dataset.
Only features, coordinates, fractions, metadata and model coefficients are saved.
"""
import argparse
import csv
import json
import shutil
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import openslide
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

from camelyon_conchz_pilot import encode_candidates, mask_measure, spatial_shortlist
from train_retrieval_study import DEFAULT_DATA, DEFAULT_MODEL, MODEL_REVISION, sha256, train_rows
from wsi_core import tissue_mask

VERSION = 1
PATCH = 1024
K_VALUES = (1, 2, 4, 8, 16, 32)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def used_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def storage_guard(args, reserve):
    args.output.mkdir(parents=True, exist_ok=True)
    if used_bytes(args.output) + reserve > args.max_cache_gb * 1e9:
        raise RuntimeError('Output size cap reached. No dataset files were changed.')
    if shutil.disk_usage(args.output).free - reserve < args.min_free_gb * 1e9:
        raise RuntimeError('Free-space reserve reached; restart after freeing space.')


def build_manifest(args):
    rows = train_rows(args.data, None, 4, args.seed, all_clean=True)
    # Split WSIs before drawing any crops; nearby crops cannot cross this split.
    strata = [f'{r["class"]}:{r["center"]}' for r in rows]
    train, val = train_test_split(rows, test_size=args.val_fraction,
                                 random_state=args.seed, stratify=strata)
    split = {r['image']: 'train' for r in train}
    split.update({r['image']: 'validation' for r in val})
    config = {
        'version': VERSION, 'data': str(args.data), 'model_dir': str(args.model),
        'model_revision': MODEL_REVISION, 'model_config_sha256': sha256(args.model / 'config.json'),
        'reference_sha256': sha256(args.data / 'evaluation/reference.csv'),
        'annotation_readme_sha256': sha256(args.data / 'README.md'),
        'seed': args.seed, 'val_fraction': args.val_fraction, 'patch_size': PATCH,
        'train_positive_target': args.positives, 'train_negative_target': args.negatives,
        'validation_candidates': args.val_candidates, 'tissue_minimum': 0.5,
        'target': 'count(native tumor-mask pixels == 2) / 1024**2',
        'features': 'FP32 frozen CONCH-Z encode_image; store float16, fit float32',
        'sampling': 'two grids offset (0,0) and (512,512); XML-centered shifted positives',
    }
    return {'config': config, 'slides': [dict(r, split=split[r['image']]) for r in rows]}


def ensure_manifest(args, manifest, write=False):
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Settings differ from this cache. Use a new --output directory.')
    if write and not path.exists():
        atomic_json(path, manifest)


def slide_cache(args, row):
    return args.output / 'features' / (Path(row['image']).stem + '.npz')


def read_cache(args, row, manifest):
    path = slide_cache(args, row)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'].item()))
        arrays = {k: z[k] for k in ('features', 'xy', 'fraction')}
    if meta != {'config': manifest['config'], 'slide': row}:
        raise ValueError(f'Incompatible cache: {path}; choose a new --output')
    x, xy, y = arrays['features'], arrays['xy'], arrays['fraction']
    if (x.ndim != 2 or x.shape[1] != 768 or xy.shape != (len(x), 2)
            or y.shape != (len(x),) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any((y < 0) | (y > 1))):
        raise ValueError(f'Invalid cache: {path}; remove only this file and rerun prepare')
    return arrays


def tissue_grids(slide):
    """Overlapping 1024 crops, stride 1024 in each of two half-offset grids."""
    width, height = slide.dimensions
    thumb = np.asarray(slide.get_thumbnail((2048, 2048)).convert('RGB'))
    tissue = tissue_mask(thumb)
    th, tw = tissue.shape
    result, seen = [], set()
    for offset in (0, PATCH // 2):
        for y in range(offset, height - PATCH + 1, PATCH):
            for x in range(offset, width - PATCH + 1, PATCH):
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                x1, y1 = round(x * tw / width), round(y * th / height)
                x2 = max(x1 + 1, round((x + PATCH) * tw / width))
                y2 = max(y1 + 1, round((y + PATCH) * th / height))
                fraction = float(tissue[y1:y2, x1:x2].mean())
                if fraction >= 0.5:
                    result.append({'x': x, 'y': y, 'tissue_fraction': fraction})
    if not result:
        raise ValueError('No grid crops pass the thumbnail tissue filter')
    return result


def tumor_centers(xml_path):
    polygons = []
    for ann in ET.parse(xml_path).getroot().findall('.//Annotation'):
        if ann.get('PartOfGroup', '').lower() != 'tumor':
            continue
        points = np.array([(float(p.attrib['X']), float(p.attrib['Y']))
                           for p in ann.findall('.//Coordinate')])
        if len(points) >= 3:
            polygons.append(points)
    if not polygons:
        raise ValueError(f'No tumor contours in {xml_path}')
    return polygons


def bin_index(fraction):
    # Only nonzero fractions enter this function. Equal quotas encourage variety.
    return int(np.searchsorted([0.01, 0.10, 0.50], fraction, side='right'))


def sample_training(args, row, slide, mask, grid, rng):
    positive, negative, seen = [], [], set()
    if row['type'] == 'tumor':
        polygons = tumor_centers(args.data / 'annotations' / (Path(row['image']).stem + '.xml'))
        buckets = [[] for _ in range(4)]
        width, height = slide.dimensions
        # Off-center crops vary coverage and context. These are unique coordinates,
        # but adjacent crops may overlap; they remain in the same WSI split.
        for attempt in range(max(256, args.positives * 16)):
            polygon = polygons[attempt % len(polygons)]
            if attempt < len(polygons):
                point = polygon.mean(axis=0)
            elif rng.random() < 0.5:
                point = polygon[rng.integers(len(polygon))]
            else:
                point = rng.uniform(polygon.min(axis=0), polygon.max(axis=0))
                if cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, point)), False) < 0:
                    point = polygon[rng.integers(len(polygon))]
            shift = np.zeros(2) if attempt < len(polygons) else rng.integers(-PATCH // 2, PATCH // 2 + 1, 2)
            x = int(np.clip(round(point[0]) - PATCH // 2 + shift[0], 0, width - PATCH))
            y = int(np.clip(round(point[1]) - PATCH // 2 + shift[1], 0, height - PATCH))
            if (x, y) in seen:
                continue
            seen.add((x, y))
            p = {'x': x, 'y': y}
            _, fraction = mask_measure(mask, p, PATCH)
            if fraction > 0:
                p['fraction'] = fraction
                buckets[bin_index(fraction)].append(p)
            if all(len(b) >= int(np.ceil(args.positives / 4)) for b in buckets):
                break
            # Do not repeatedly read native masks trying to manufacture high
            # coverage for a lesion too small to fill a 1024-pixel crop.
            if sum(map(len, buckets)) >= args.positives * 3:
                break
        # Round robin available bins; small lesions need not fill impossible high bins.
        while len(positive) < args.positives and any(buckets):
            for bucket in buckets:
                if bucket and len(positive) < args.positives:
                    positive.append(bucket.pop(0))
        if not positive:
            raise ValueError(f'No positive crops found for {row["image"]}')
    for idx in rng.permutation(len(grid)):
        p = grid[idx].copy()
        if (p['x'], p['y']) in seen:
            continue
        _, fraction = mask_measure(mask, p, PATCH)
        if fraction == 0:
            p['fraction'] = 0.0
            negative.append(p)
            if len(negative) >= args.negatives:
                break
    if not negative:
        raise ValueError(f'No zero-overlap tissue crops found for {row["image"]}')
    return positive + negative


def prepare(args, manifest):
    storage_guard(args, 0)
    ensure_manifest(args, manifest, write=True)
    pending = [r for r in manifest['slides'] if read_cache(args, r, manifest) is None]
    print(f'{len(manifest["slides"]) - len(pending)} cached; {len(pending)} remaining', flush=True)
    if not pending:
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; specify a usable --device')
    model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True,
                                     local_files_only=True).eval().float().to(args.device)
    for i, row in enumerate(pending, 1):
        # Reserve enough for a full per-slide checkpoint and temporary output.
        maximum = (args.val_candidates if row['split'] == 'validation'
                   else args.positives + args.negatives)
        storage_guard(args, max(8 * 1024**2, maximum * 1700 + 1024**2))
        stem = Path(row['image']).stem
        rng = np.random.default_rng(args.seed + int(stem.rsplit('_', 1)[1])
                                    + (10000 if stem.startswith('normal') else 0))
        print(f'{i}/{len(pending)} {stem} [{row["split"]}]', flush=True)
        with openslide.OpenSlide(str(args.data / 'images' / row['image'])) as slide:
            grid = tissue_grids(slide)
            # Validation coordinates are fixed WITHOUT accessing the tumor mask.
            selected = (spatial_shortlist(grid, args.val_candidates, *slide.dimensions)
                        if row['split'] == 'validation' else None)
            with openslide.OpenSlide(str(args.data / 'masks' / f'{stem}_mask.tif')) as mask:
                if slide.dimensions != mask.dimensions:
                    raise ValueError(f'Image/mask alignment mismatch: {stem}')
                if selected is None:
                    selected = sample_training(args, row, slide, mask, grid, rng)
                else:
                    for p in selected:
                        _, p['fraction'] = mask_measure(mask, p, PATCH)
            vectors = encode_candidates(slide, model, selected, PATCH,
                                        torch.device(args.device), args.batch_size)
        path = slide_cache(args, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.npz.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, features=vectors.numpy().astype(np.float16),
                                xy=np.array([(p['x'], p['y']) for p in selected], dtype=np.int64),
                                fraction=np.array([p['fraction'] for p in selected], dtype=np.float32),
                                metadata=np.array(json.dumps({'config': manifest['config'], 'slide': row})))
        temp.replace(path)
        counts = Counter('zero' if p['fraction'] == 0 else str(bin_index(p['fraction'])) for p in selected)
        print(f'  {len(selected)} crops, coverage bins {dict(counts)}; cache {used_bytes(args.output)/1e6:.1f} MB', flush=True)


def score_metrics(y, scores):
    labels = y > 0
    return {'auroc': float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            'average_precision': float(average_precision_score(labels, scores)) if labels.any() else None}


def fit(args, manifest):
    ensure_manifest(args, manifest)
    storage_guard(args, 16 * 1024**2)
    records = []
    for row in manifest['slides']:
        cache = read_cache(args, row, manifest)
        if cache is None:
            raise FileNotFoundError(f'Run prepare first: {slide_cache(args, row)}')
        records.append((row, cache))
    xs, ys, ws = [], [], []
    for row, c in records:
        if row['split'] != 'train':
            continue
        x, y = c['features'].astype(np.float32), c['fraction']
        # Equal slide contribution, and equal positive/zero contribution within
        # positive slides. This avoids a sea of zero targets dominating the loss.
        w = np.full(len(y), 1 / len(y), dtype=np.float64)
        if (y > 0).any() and (y == 0).any():
            w[y > 0] = 0.5 / (y > 0).sum()
            w[y == 0] = 0.5 / (y == 0).sum()
        xs.append(x); ys.append(y); ws.append(w)
    x, y, weights = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(x)
    regressor = Ridge(alpha=args.alpha).fit(scaler.transform(x), y, sample_weight=weights)
    np.savez_compressed(args.output / 'regressor.npz', mean=scaler.mean_, scale=scaler.scale_,
                        coef=regressor.coef_, intercept=np.array(regressor.intercept_))
    evaluate_regressor(args, manifest, records, regressor, scaler, y, weights)


def evaluate_regressor(args, manifest, records, regressor, scaler, y, weights, info=None):
    """Shared held-out evaluation for Ridge and neural regression heads."""
    baseline = float(np.average(y, weights=weights))
    per_slide, patch_rows, all_y, all_s = [], [], [], []
    for row, c in records:
        if row['split'] != 'validation':
            continue
        target = c['fraction']
        # Rank by raw regression output. Clip only for bounded coverage reporting;
        # clipping before ranking would create ties among low scores.
        scores = regressor.predict(scaler.transform(c['features'].astype(np.float32)))
        predicted = np.clip(scores, 0, 1)
        order = np.argsort(-scores, kind='stable')
        metrics = score_metrics(target, scores)
        corr = None
        if np.std(target) > 0 and np.std(scores) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                value = float(spearmanr(target, scores).statistic)
            corr = value if np.isfinite(value) else None
        stem = Path(row['image']).stem
        entry = {'slide': stem, 'class': row['class'], 'candidates': len(target),
                 'tumor_candidates': int((target > 0).sum()),
                 'mae': float(np.abs(predicted-target).mean()),
                 'constant_baseline_mae': float(np.abs(baseline-target).mean()),
                 'spearman': corr, **metrics,
                 'top_k': {str(k): {'tumor_hit': bool((target[order[:k]] > 0).any()),
                                    'strong_hit': bool((target[order[:k]] >= 0.001).any()),
                                    'precision': float((target[order[:k]] > 0).mean()),
                                    'mean_true_coverage': float(target[order[:k]].mean())}
                           for k in K_VALUES}}
        per_slide.append(entry)
        for rank, idx in enumerate(order, 1):
            patch_rows.append({'slide': stem, 'class': row['class'], 'rank': rank,
                               'x': int(c['xy'][idx, 0]), 'y': int(c['xy'][idx, 1]),
                               'predicted_fraction': float(predicted[idx]),
                               'raw_score': float(scores[idx]), 'true_fraction': float(target[idx])})
        all_y.append(target); all_s.append(scores)
    yy, ss = np.concatenate(all_y), np.concatenate(all_s)
    positives = [r for r in per_slide if r['class'] != 'negative']
    covered = [r for r in positives if r['tumor_candidates'] > 0]
    summary = {
        'objective': 'weighted squared error to native annotated tumor fraction; frozen CONCH-Z + Ridge',
        'alpha': getattr(args, 'alpha', None), 'train_patches': len(y), 'train_nonzero_patches': int((y > 0).sum()),
        'train_slides': sum(r['split'] == 'train' for r in manifest['slides']),
        'validation_slides': len(per_slide), 'validation_positive_slides': len(positives),
        'candidate_hit_positive_slides': len(covered),
        'validation_pooled_patch_metrics': score_metrics(yy, ss),
        'validation_patch_mae': float(np.abs(np.clip(ss, 0, 1) - yy).mean()),
        'constant_baseline_mae': float(np.abs(baseline - yy).mean()),
        'mean_within_slide_auroc': float(np.mean([r['auroc'] for r in covered if r['auroc'] is not None]))
            if any(r['auroc'] is not None for r in covered) else None,
        'top_k': {str(k): {'hit_positive_slides': sum(r['top_k'][str(k)]['tumor_hit'] for r in positives),
                           'strong_hit_positive_slides': sum(r['top_k'][str(k)]['strong_hit'] for r in positives),
                           'total_positive_slides': len(positives),
                           'candidate_covered_positive_slides': len(covered)} for k in K_VALUES},
        'limitations': [
            'Coverage is annotated metastasis area, not clinical severity or every abnormality.',
            'Tiny true tumors have low regression targets; top-K recall is reported separately.',
            'Mask-guided training distribution differs from mask-blind validation distribution.',
            'Candidate subsampling and thumbnail filtering may miss tumor; no exhaustive slide scan.',
            'This development validation split is from the previously inspected training cohort.',
            'Output is an uncalibrated coverage estimate; no slide diagnosis threshold is fitted.',
        ],
    }
    if info is not None:
        summary.pop('alpha', None)
        summary.update(info)
    atomic_json(args.output / 'summary.json', summary)
    atomic_json(args.output / 'validation_slides.json', per_slide)
    with (args.output / 'validation_patches.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(patch_rows[0]))
        writer.writeheader(); writer.writerows(patch_rows)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan', 'prepare', 'fit'))
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--output', type=Path, default=Path.home() / 'wsi_patch_fraction_cache_v1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--positives', type=int, default=64)
    p.add_argument('--negatives', type=int, default=64)
    p.add_argument('--val-candidates', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--alpha', type=float, default=10.0)
    p.add_argument('--max-cache-gb', type=float, default=2.0)
    p.add_argument('--min-free-gb', type=float, default=2.0)
    args = p.parse_args()
    if (min(args.positives, args.negatives, args.val_candidates, args.batch_size) < 1
            or not 0.1 <= args.val_fraction <= 0.4 or args.alpha <= 0
            or args.max_cache_gb <= 0 or args.min_free_gb < 0):
        p.error('Require positive counts/alpha/cap, nonnegative free reserve and val-fraction 0.1–0.4')
    for key in ('data', 'model', 'output'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.output.is_relative_to(Path.home().resolve()):
        p.error('--output must be under the user home directory so caches stay off the dataset volume')
    manifest = build_manifest(args)
    ensure_manifest(args, manifest)
    counts = Counter((r['split'], r['class']) for r in manifest['slides'])
    maximum = sum(args.val_candidates if r['split'] == 'validation' else
                  args.negatives + (args.positives if r['type'] == 'tumor' else 0)
                  for r in manifest['slides'])
    print('WSI split:', dict(counts), flush=True)
    print(f'At most {maximum:,} cached patches; raw feature storage '
          f'{maximum * 768 * 2 / 1e6:.1f} MB plus coordinates/metadata; no image crops saved.', flush=True)
    if args.stage == 'plan':
        print('Plan only: no inference or output writes. Cache:', args.output)
    elif args.stage == 'prepare':
        prepare(args, manifest)
    else:
        fit(args, manifest)


if __name__ == '__main__':
    main()
