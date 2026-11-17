#!/usr/bin/env python3
"""Validation-only low-tissue-cutoff CONCH-Z + frozen FP32 MLP retrieval.

No training and no GPT/API calls. `plan` checks the existing validation split
and paths; `run` encodes every two-grid crop with >=5% thumbnail tissue once,
then compares 5% and 10% cutoffs using the trained MLP. Tumor masks are read
only after coordinates and model scores are fixed, to compute retrieval metrics.
"""
import argparse
import csv
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import openslide
import torch
from transformers import AutoModel

from patch_fraction_regression import atomic_json
from test_patch_regression_heads import SavedScaler
from train_patch_neural_head import make_head
from train_retrieval_study import sha256
from wsi_core import tissue_mask

PATCH = 1024
THRESHOLDS = (0.05, 0.10)
K_VALUES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_TRAINING = Path.home() / 'wsi_patch_fraction_cache_v1'
DEFAULT_OUTPUT = Path.home() / 'wsi_validation_low_tissue_retrieval_v1'


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def used_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def storage_guard(args, reserve=0):
    args.output.mkdir(parents=True, exist_ok=True)
    if used_bytes(args.output) + reserve > args.max_cache_gb * 1e9:
        raise RuntimeError('Output size limit reached; no dataset files were changed.')
    if shutil.disk_usage(args.output).free - reserve < args.min_free_gb * 1e9:
        raise RuntimeError('Configured free-space reserve would be violated.')


def build_setup(args):
    source_path = args.training / 'manifest.json'
    source = json.loads(source_path.read_text())
    checkpoint_path = args.training / 'neural_mlp_seed42' / 'model.pt'
    run_config_path = checkpoint_path.parent / 'run_config.json'
    run_config = json.loads(run_config_path.read_text())
    if run_config.get('head') != 'mlp' or run_config.get('cache_manifest') != source:
        raise ValueError('Saved MLP does not match the existing training feature manifest.')
    rows = [r for r in source['slides'] if r['split'] == 'validation']
    if len(rows) != 50 or {r['image'] for r in rows} != set(run_config['outer_validation_slides']):
        raise ValueError('Expected exactly the existing 50 development validation WSIs.')
    if any(Path(r['image']).stem.startswith('test_') for r in rows):
        raise ValueError('Validation manifest unexpectedly contains official test slides.')
    args.data = Path(source['config']['data'])
    args.encoder = Path(source['config']['model_dir'])
    if sha256(args.data / 'evaluation/reference.csv') != source['config']['reference_sha256']:
        raise ValueError('Dataset reference labels changed since MLP training.')
    if sha256(args.encoder / 'config.json') != source['config']['model_config_sha256']:
        raise ValueError('CONCH-Z config changed since MLP training.')
    for row in rows:
        stem = Path(row['image']).stem
        for path in (args.data / 'images' / row['image'],
                     args.data / 'masks' / f'{stem}_mask.tif'):
            if not path.is_file():
                raise FileNotFoundError(path)
    checkpoint_hash = sha256(checkpoint_path)
    config = {
        'protocol_version': 1,
        'split': 'existing 50-WSI development validation split; not official test',
        'source_manifest_sha256': sha256(source_path),
        'training_checkpoint_sha256': checkpoint_hash,
        'encoder_config_sha256': sha256(args.encoder / 'config.json'),
        'reference_sha256': source['config']['reference_sha256'],
        'candidate_definition': 'all full 1024x1024 patches on two fixed grids passing >=5% thumbnail tissue; no candidate cap',
        'comparison_tissue_cutoffs': list(THRESHOLDS),
        'grid_offsets_level0': [[0, 0], [512, 512]],
        'grid_stride_level0': PATCH,
        'edge_rule': 'only full patches entirely inside slide; no padding or shifted edge crops',
        'thumbnail_max_side': 2048,
        'tissue_mask': {
            'color_space': 'OpenCV uint8 HSV', 'saturation_gt': 35, 'value_lt': 250,
            'opening_kernel': '3x3 ellipse', 'closing_kernel': '5x5 ellipse',
            'fraction': 'mean tissue-mask pixels in mapped thumbnail rectangle',
        },
        'encoder': 'local CONCH-Z, unquantized FP32; frozen; eval_transform; encode_image 768-D projected L2-normalized embedding',
        'mlp': 'saved trained MLP, original FP32 weights, eval mode; no fitting or quantization',
        'feature_scaler': 'saved training-only StandardScaler statistics; exact SavedScaler transform',
        'ranking': 'descending raw MLP predicted tumor coverage; stable tie order; no NMS or spatial reranking',
        'mask_policy': 'masks are opened after candidate features/scores are fixed; evaluation only; tumor fraction is mask label 2 pixels / 1024^2',
        'top_k_values': list(K_VALUES),
        'patch_preprocessing': 'CONCH-Z model.eval_transform from local model; no stain normalization',
        'no_gpt_call': True,
    }
    return source, rows, checkpoint_path, config


def make_grid(slide, min_tissue):
    """All valid positions on the two existing grids above a permissive cutoff."""
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
                if fraction >= min_tissue:
                    result.append({'x': x, 'y': y, 'tissue_fraction': fraction})
    return result


def feature_path(args, row):
    return args.output / 'features' / f'{Path(row["image"]).stem}.npz'


def load_feature(path, config, row):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'].item()))
        if meta != {'config': config, 'slide': row}:
            raise ValueError(f'Cache settings mismatch: {path}')
        arrays = {k: z[k] for k in ('features', 'xy', 'tissue_fraction', 'tumor_fraction', 'scores')}
    n = len(arrays['features'])
    if (arrays['features'].shape != (n, 768) or arrays['xy'].shape != (n, 2)
            or any(arrays[k].shape != (n,) for k in ('tissue_fraction', 'tumor_fraction', 'scores'))
            or not n or not all(np.isfinite(v).all() for v in arrays.values())):
        raise ValueError(f'Invalid feature cache: {path}')
    return arrays


def infer_scores(head, scaler, features, device, batch_size):
    x = scaler.transform(features.astype(np.float32))
    scores = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            scores.append(head(xb).flatten().float().cpu().numpy())
    return np.concatenate(scores)


def encode_one(args, row, config, encoder, head, scaler):
    stem = Path(row['image']).stem
    started = time.perf_counter()
    with openslide.OpenSlide(str(args.data / 'images' / row['image'])) as slide:
        grid = make_grid(slide, min(THRESHOLDS))
        if not grid:
            raise ValueError(f'{stem}: no grid positions pass the {min(THRESHOLDS):.0%} cutoff')
        embeddings = []
        for start in range(0, len(grid), args.batch_size):
            batch = grid[start:start + args.batch_size]
            tensors = [encoder.eval_transform(
                slide.read_region((p['x'], p['y']), 0, (PATCH, PATCH)).convert('RGB'))
                for p in batch]
            images = torch.stack(tensors).to(args.device)
            with torch.inference_mode():
                embeddings.append(encoder.encode_image(images).float().cpu().numpy())
            print(f'  {stem}: encoded {min(start + len(batch), len(grid))}/{len(grid)}', flush=True)
        features = np.concatenate(embeddings).astype(np.float32)
        scores = infer_scores(head, scaler, features, args.device, args.batch_size)
        xy = np.asarray([(p['x'], p['y']) for p in grid], dtype=np.int64)
        tissue_fraction = np.asarray([p['tissue_fraction'] for p in grid], dtype=np.float32)
        # Evaluation labels are read only after model scores and coordinates are fixed.
        with openslide.OpenSlide(str(args.data / 'masks' / f'{stem}_mask.tif')) as mask:
            if mask.dimensions != slide.dimensions:
                raise ValueError(f'{stem}: slide/mask dimensions differ')
            tumor_fraction = []
            for p in grid:
                labels = np.asarray(mask.read_region((p['x'], p['y']), 0,
                                                       (PATCH, PATCH)).convert('RGB'))[:, :, 0]
                tumor_fraction.append(np.count_nonzero(labels == 2) / labels.size)
        tumor_fraction = np.asarray(tumor_fraction, dtype=np.float32)
    result = {'features': features.astype(np.float16), 'xy': xy,
              'tissue_fraction': tissue_fraction, 'tumor_fraction': tumor_fraction,
              'scores': scores.astype(np.float32),
              'metadata': np.array(json.dumps({'config': config, 'slide': row}))}
    path = feature_path(args, row)
    storage_guard(args, max(8 * 1024**2, features.nbytes * 2 + 2 * 1024**2))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(path, **result)
    return result, time.perf_counter() - started


def evaluate(args, rows, config):
    slide_rows, patch_rows, summary = [], [], {'thresholds': {}, 'top_k': list(K_VALUES)}
    for threshold in THRESHOLDS:
        slide_metrics = []
        for row in rows:
            stem = Path(row['image']).stem
            data = load_feature(feature_path(args, row), config, row)
            eligible = np.flatnonzero(data['tissue_fraction'] >= threshold)
            if not len(eligible):
                raise ValueError(f'{stem}: no candidates for tissue cutoff {threshold}')
            order = eligible[np.argsort(-data['scores'][eligible], kind='stable')]
            positive = row['type'] == 'tumor'
            candidate_hit = bool(np.any(data['tumor_fraction'][eligible] > 0))
            rec = {'slide': stem, 'class': row['class'], 'positive_slide': positive,
                   'tissue_cutoff': threshold, 'eligible_candidates': int(len(eligible)),
                   'candidate_tumor_hits': int(np.count_nonzero(data['tumor_fraction'][eligible] > 0)),
                   'candidate_coverage_hit': candidate_hit,
                   'max_candidate_tumor_fraction': float(data['tumor_fraction'][eligible].max()),
                   'mean_candidate_tumor_fraction': float(data['tumor_fraction'][eligible].mean())}
            for k in K_VALUES:
                selected = order[:min(k, len(order))]
                fractions = data['tumor_fraction'][selected]
                hits = fractions > 0
                strong = fractions >= 0.001
                rec[f'top{k}_any_hit'] = bool(hits.any())
                rec[f'top{k}_strong_hit'] = bool(strong.any())
                rec[f'top{k}_positive_patches'] = int(hits.sum())
                rec[f'top{k}_positive_patch_fraction'] = float(hits.mean())
                rec[f'top{k}_max_tumor_fraction'] = float(fractions.max())
                if k == 64:
                    for rank, index in enumerate(selected, 1):
                        patch_rows.append({
                            'slide': stem, 'class': row['class'], 'positive_slide': positive,
                            'tissue_cutoff': threshold, 'rank': rank,
                            'x_level0': int(data['xy'][index, 0]), 'y_level0': int(data['xy'][index, 1]),
                            'mlp_score_raw': float(data['scores'][index]),
                            'tissue_fraction': float(data['tissue_fraction'][index]),
                            'tumor_fraction_mask_evaluation_only': float(data['tumor_fraction'][index]),
                        })
            slide_metrics.append(rec)
            slide_rows.append(rec)
        positives = [r for r in slide_metrics if r['positive_slide']]
        threshold_result = {
            'validation_slides': len(slide_metrics),
            'positive_slides': len(positives),
            'negative_slides': len(slide_metrics) - len(positives),
            'positive_candidate_coverage_hits': sum(r['candidate_coverage_hit'] for r in positives),
            'positive_candidate_coverage_rate': float(np.mean([r['candidate_coverage_hit'] for r in positives])),
            'mean_candidate_count': float(np.mean([r['eligible_candidates'] for r in slide_metrics])),
            'median_candidate_count': float(np.median([r['eligible_candidates'] for r in slide_metrics])),
            'mean_positive_candidate_count': float(np.mean([r['eligible_candidates'] for r in positives])),
            'subtypes': {}, 'top_k': {},
        }
        for subtype in ('micro', 'macro'):
            group = [r for r in positives if r['class'] == subtype]
            threshold_result['subtypes'][subtype] = {
                'positive_slides': len(group),
                'candidate_coverage_hits': sum(r['candidate_coverage_hit'] for r in group),
                **{f'top{k}_any_hits': sum(r[f'top{k}_any_hit'] for r in group) for k in K_VALUES},
            }
        for k in K_VALUES:
            hit_count = sum(r[f'top{k}_any_hit'] for r in positives)
            strong_count = sum(r[f'top{k}_strong_hit'] for r in positives)
            threshold_result['top_k'][str(k)] = {
                'positive_slides_with_any_overlap': hit_count,
                'positive_slide_recall_any': hit_count / len(positives),
                'positive_slides_with_strong_overlap_ge_0.1_percent': strong_count,
                'positive_slide_recall_strong': strong_count / len(positives),
                'positive_patches_among_top_k_on_positive_slides': sum(r[f'top{k}_positive_patches'] for r in positives),
                'all_selected_patches_on_positive_slides': sum(min(k, r['eligible_candidates']) for r in positives),
                'patch_precision_any_overlap': (
                    sum(r[f'top{k}_positive_patches'] for r in positives)
                    / sum(min(k, r['eligible_candidates']) for r in positives)),
                'micro_slides_hit': sum(r[f'top{k}_any_hit'] for r in positives if r['class'] == 'micro'),
                'macro_slides_hit': sum(r[f'top{k}_any_hit'] for r in positives if r['class'] == 'macro'),
            }
        summary['thresholds'][str(threshold)] = threshold_result
    atomic_json(args.output / 'summary.json', summary)
    if slide_rows:
        with (args.output / 'per_slide_metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(slide_rows[0]))
            writer.writeheader(); writer.writerows(slide_rows)
    if patch_rows:
        with (args.output / 'ranked_candidates_top64.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(patch_rows[0]))
            writer.writeheader(); writer.writerows(patch_rows)
    for t, result in summary['thresholds'].items():
        print(f'Cutoff {float(t):.0%}: mean candidates={result["mean_candidate_count"]:.1f}; '
              f'candidate coverage={result["positive_candidate_coverage_hits"]}/{result["positive_slides"]}', flush=True)
        for k, metrics in result['top_k'].items():
            print(f'  top{k}: any-hit={metrics["positive_slides_with_any_overlap"]}/{result["positive_slides"]}; '
                  f'strong-hit={metrics["positive_slides_with_strong_overlap_ge_0.1_percent"]}/{result["positive_slides"]}; '
                  f'patch precision={metrics["patch_precision_any_overlap"]:.3f}', flush=True)
    print('Saved validation embeddings, per-slide metrics and ranked patch coordinates to', args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan', 'run', 'status'))
    p.add_argument('--training', type=Path, default=DEFAULT_TRAINING)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-cache-gb', type=float, default=2.0)
    p.add_argument('--min-free-gb', type=float, default=2.0)
    args = p.parse_args()
    args.training = args.training.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if (not args.output.is_relative_to(Path.home())
            or args.output == args.training or args.output.is_relative_to(args.training)
            or args.training.is_relative_to(args.output)):
        p.error('Output must be separate from the training cache and under the user home directory.')
    if args.batch_size < 1 or args.threads < 1 or args.max_cache_gb <= 0 or args.min_free_gb < 0:
        p.error('Invalid batch size, thread count or storage limit.')
    source, rows, checkpoint_path, config = build_setup(args)
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing output uses different inputs/settings; choose a new --output.')
    if args.stage == 'plan':
        print('Validation-only CONCH-Z + MLP retrieval; no inference, output writes or GPT calls.')
        print(f'Validation slides: {len(rows)} ({dict(Counter(r["class"] for r in rows))})')
        print(f'Cutoffs compared: {THRESHOLDS}; no candidate cap; top-k: {K_VALUES}')
        print(f'CONCH-Z: unquantized FP32; MLP: saved FP32 checkpoint {checkpoint_path}')
        print(f'Candidate embedding storage ceiling if ~3,000 patches/slide: '
              f'{len(rows) * 3000 * 768 * 2 / 1e9:.2f} GB plus metadata; hard cap {args.max_cache_gb} GB.')
        print(f'Dataset: {args.data}; output: {args.output}')
        return
    storage_guard(args)
    if args.stage == 'status':
        complete = sum(load_feature(feature_path(args, r), config, r) is not None for r in rows)
        print(f'Cached {complete}/{len(rows)} validation slides; {used_bytes(args.output)/1e6:.1f} MB used.')
        print('No inference or output writes.')
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu or a working CUDA device.')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    head = make_head('mlp', checkpoint['hidden'], checkpoint['dropout'])
    head.load_state_dict(checkpoint['state_dict'], strict=True)
    head.eval().float().to(args.device)
    if any(param.dtype != torch.float32 for param in head.parameters()):
        raise RuntimeError('The MLP must remain FP32.')
    scaler = SavedScaler(checkpoint['scaler_mean'].numpy(), checkpoint['scaler_scale'].numpy())
    print('Loading frozen local CONCH-Z in FP32; no quantization and no training.', flush=True)
    encoder = AutoModel.from_pretrained(str(args.encoder), trust_remote_code=True,
                                        local_files_only=True).eval().float().to(args.device)
    pending = [r for r in rows if load_feature(feature_path(args, r), config, r) is None]
    print(f'{len(rows) - len(pending)} complete; {len(pending)} slides remaining.', flush=True)
    for index, row in enumerate(pending, 1):
        print(f'{index}/{len(pending)} {Path(row["image"]).stem}', flush=True)
        encode_one(args, row, config, encoder, head, scaler)
    del encoder, head
    if any(load_feature(feature_path(args, r), config, r) is None for r in rows):
        raise RuntimeError('Some validation feature files are missing; safe to resume with run.')
    evaluate(args, rows, config)


if __name__ == '__main__':
    main()
