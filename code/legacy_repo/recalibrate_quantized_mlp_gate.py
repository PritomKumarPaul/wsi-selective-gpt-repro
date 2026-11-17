#!/usr/bin/env python3
"""Recompute the current MLP no-call gate on development validation using W8A8 CONCH-Z.

Runs the existing validation slides through the exact intended final sampler:
one 1024px grid, >=50% thumbnail tissue, cap 500, deterministic spatial spread.
Only CONCH-Z visual Linear layers are W8A8 quantized; the saved MLP/scaler stay
FP32. No masks or GPT calls are used. One JSON score record per slide supports
resuming. The resulting gate is written separately from all test outputs.
"""
import argparse
import csv
import json
import math
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
from validation_low_tissue_mlp_retrieval import make_grid
from cuda_int8_linear import check_cuda_int8, convert_cuda_linears


PATCH = 1024
CAP = 500
TISSUE_CUTOFF = 0.50
DEFAULT_TRAINING = Path.home() / 'wsi_patch_fraction_cache_v1'
DEFAULT_OUTPUT = Path.home() / 'mlp_validation_gate_quantized_v1'
QUANTIZATION = 'CUDA W8A8 torch._int_mm on encoder.visual nn.Linear only; all remaining CONCH ops FP32'


def one_grid_candidates(slide):
    """Mirror the first, (0,0) grid produced by validation_low_tissue_mlp_retrieval.make_grid."""
    all_grid = make_grid(slide, TISSUE_CUTOFF)
    return [p for p in all_grid if p['x'] % PATCH == 0 and p['y'] % PATCH == 0]


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def spatial_cap(candidates, width, height):
    if len(candidates) <= CAP:
        return candidates
    centers = np.column_stack(((np.asarray([p['x'] for p in candidates]) + PATCH / 2) / width,
                               (np.asarray([p['y'] for p in candidates]) + PATCH / 2) / height))
    tissue = np.asarray([p['tissue_fraction'] for p in candidates], dtype=np.float64)
    used = np.zeros(len(candidates), dtype=bool)
    min_distance = np.full(len(candidates), np.inf)
    current = int(np.argmax(tissue))
    chosen = []
    for _ in range(CAP):
        chosen.append(candidates[current])
        used[current] = True
        min_distance = np.minimum(min_distance, ((centers - centers[current]) ** 2).sum(axis=1))
        min_distance[used] = -1
        current = int(np.argmax(min_distance + 1e-12 * tissue))
    return chosen


def encode_score(slide, candidates, encoder, head, scaler, device, batch_size):
    feature_parts = []
    scores = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        images = torch.stack([
            encoder.eval_transform(slide.read_region((p['x'], p['y']), 0,
                                                     (PATCH, PATCH)).convert('RGB'))
            for p in batch]).to(device)
        with torch.inference_mode():
            features = encoder.encode_image(images).float().cpu().numpy()
        feature_parts.append(features.astype(np.float32, copy=False))
        standardized = scaler.transform(features.astype(np.float32))
        with torch.inference_mode():
            values = head(torch.from_numpy(standardized).to(device)).flatten().float().cpu().numpy()
        scores.extend(values.tolist())
        print(f'    encoded/scored {min(start + len(batch), len(candidates))}/{len(candidates)}',
              flush=True)
    result = np.asarray(scores, dtype=np.float32)
    if result.shape != (len(candidates),) or not np.isfinite(result).all():
        raise RuntimeError('Invalid/non-finite MLP scores.')
    return np.concatenate(feature_parts, axis=0), result


def threshold_sweep(records):
    values = sorted({r['max_raw_mlp_score'] for r in records})
    thresholds = ([math.nextafter(values[0], -math.inf)]
                  + [a + (b - a) / 2 for a, b in zip(values, values[1:])]
                  + [values[-1]])
    sweep = []
    for threshold in thresholds:
        skipped = [r for r in records if r['max_raw_mlp_score'] <= threshold]
        pos_skips = [r for r in skipped if r['truth'] == 'TUMOR']
        normal_skips = [r for r in skipped if r['truth'] == 'NORMAL']
        sweep.append({'threshold': float(threshold), 'skipped': len(skipped),
                      'normal_skipped': len(normal_skips), 'positive_skipped': len(pos_skips),
                      'skipped_normal_ids': [r['slide'] for r in normal_skips],
                      'skipped_positive_ids': [r['slide'] for r in pos_skips]})
    eligible = [r for r in sweep if r['positive_skipped'] == 0]
    selected = min(eligible, key=lambda r: (-r['normal_skipped'], r['threshold']))
    selected.update(gpt_calls=len(records) - selected['skipped'],
                    positive_call_recall=1.0)
    return selected, sweep


def fixed_threshold_result(records, threshold):
    skipped = [r for r in records if r['max_raw_mlp_score'] <= threshold]
    missed = [r for r in skipped if r['truth'] == 'TUMOR']
    normal = [r for r in skipped if r['truth'] == 'NORMAL']
    return {'threshold': threshold, 'skipped': len(skipped),
            'normal_skipped': len(normal), 'positive_skipped': len(missed),
            'gpt_calls': len(records) - len(skipped),
            'skipped_normal_ids': [r['slide'] for r in normal],
            'skipped_positive_ids': [r['slide'] for r in missed]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=DEFAULT_TRAINING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    args.training = args.training.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('Quantized CONCH-Z requires CUDA; specify an available --device.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    if not args.output.is_relative_to(Path.home().resolve()):
        parser.error('Output must be under the user home directory.')

    source_path = args.training / 'manifest.json'
    source = json.loads(source_path.read_text())
    checkpoint_path = args.training / 'neural_mlp_seed42' / 'model.pt'
    run_cfg = json.loads((checkpoint_path.parent / 'run_config.json').read_text())
    if run_cfg.get('head') != 'mlp' or run_cfg.get('cache_manifest') != source:
        raise ValueError('Saved current MLP does not match the training manifest.')
    rows = sorted([r for r in source['slides'] if r['split'] == 'validation'],
                  key=lambda r: r['image'])
    if len(rows) != 50 or sum(r['type'] == 'tumor' for r in rows) != 18:
        raise ValueError('Expected the saved 50-slide validation split (18 tumor-positive).')
    data = Path(source['config']['data'])
    encoder_path = Path(source['config']['model_dir'])
    if sha256(encoder_path / 'config.json') != source['config']['model_config_sha256']:
        raise ValueError('CONCH-Z model configuration differs from training.')

    config = {
        'split': 'development validation only; 50 slides',
        'training_manifest_sha256': sha256(source_path),
        'checkpoint_sha256': sha256(checkpoint_path),
        'encoder_config_sha256': sha256(encoder_path / 'config.json'),
        'quantization': QUANTIZATION,
        'mlp': 'saved current MLP, FP32; unchanged and not quantized',
        'sampler': {'patch': PATCH, 'grid': 'one (0,0) grid, stride 1024; complete crops only',
                    'tissue_cutoff': TISSUE_CUTOFF, 'candidate_cap': CAP,
                    'cap_rule': 'farthest-point spatial spread seeded by highest tissue fraction'},
        'score': 'maximum uncalibrated raw MLP tumor-coverage score among sampled candidates',
        'script_sha256': sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'config.json'
    if config_path.exists():
        saved_config = json.loads(config_path.read_text())
        # The script hash is provenance, not a feature-generation setting. Allow
        # a code-only repair (e.g. CSV export fix) to reuse completed slide caches.
        comparable_saved = {k: v for k, v in saved_config.items() if k != 'script_sha256'}
        comparable_current = {k: v for k, v in config.items() if k != 'script_sha256'}
        if comparable_saved != comparable_current:
            raise ValueError('Output contains a different run configuration; choose a new --output.')
        config = saved_config
    else:
        atomic_json(config_path, config)
    for row in rows:
        if not (data / 'images' / row['image']).is_file():
            raise FileNotFoundError(data / 'images' / row['image'])

    prior = {}
    for row in rows:
        path = args.output / 'slides' / f'{Path(row["image"]).stem}.json'
        feature_path = args.output / 'features' / f'{Path(row["image"]).stem}.npz'
        if path.is_file() and feature_path.is_file():
            record = json.loads(path.read_text())
            if record.get('config') != config or record.get('truth') != (
                    'TUMOR' if row['type'] == 'tumor' else 'NORMAL'):
                raise ValueError(f'Cached validation record mismatch: {path}')
            if record.get('feature_cache_sha256') != sha256(feature_path):
                raise ValueError(f'Cached feature file hash mismatch: {feature_path}')
            with np.load(feature_path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache['metadata'].item()))
                if metadata != {'config': config, 'slide': row}:
                    raise ValueError(f'Cached feature metadata mismatch: {feature_path}')
                n = len(cache['features'])
                if (cache['features'].shape != (n, 768)
                        or cache['xy'].shape != (n, 2)
                        or cache['tissue_fraction'].shape != (n,)
                        or cache['scores'].shape != (n,)
                        or not n or not np.isfinite(cache['features']).all()
                        or not np.isfinite(cache['scores']).all()):
                    raise ValueError(f'Invalid cached feature arrays: {feature_path}')
            prior[record['slide']] = record
    pending = [r for r in rows if Path(r['image']).stem not in prior]
    print(f'Quantized validation recalibration: {len(prior)}/50 cached, '
          f'{len(pending)} to process; MLP stays FP32.', flush=True)
    if pending:
        check_cuda_int8()
        model_started = time.perf_counter()
        encoder = AutoModel.from_pretrained(str(encoder_path), trust_remote_code=True,
                                            local_files_only=True).eval().float()
        converted = convert_cuda_linears(encoder.visual)
        if converted < 1:
            raise RuntimeError('No encoder.visual Linear layers were quantized.')
        encoder.to(args.device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        head = make_head('mlp', checkpoint['hidden'], checkpoint['dropout'])
        head.load_state_dict(checkpoint['state_dict'], strict=True)
        head.eval().float().to(args.device)
        scaler = SavedScaler(checkpoint['scaler_mean'].numpy(),
                             checkpoint['scaler_scale'].numpy())
        model_seconds = time.perf_counter() - model_started
        for index, row in enumerate(pending, 1):
            stem = Path(row['image']).stem
            slide_started = time.perf_counter()
            print(f'[{index}/{len(pending)}] {stem}', flush=True)
            with openslide.OpenSlide(str(data / 'images' / row['image'])) as slide:
                grid = one_grid_candidates(slide)
                # Match the validation calibrator's stable row-major order before farthest-point selection.
                grid.sort(key=lambda p: p['y'] * slide.dimensions[0] + p['x'])
                candidates = spatial_cap(grid, *slide.dimensions)
                if len(candidates) < 8:
                    raise ValueError(f'{stem}: fewer than eight eligible candidates.')
                features, scores = encode_score(slide, candidates, encoder, head, scaler,
                                                args.device, args.batch_size)
            feature_path = args.output / 'features' / f'{stem}.npz'
            atomic_npz(feature_path,
                       features=features.astype(np.float16),
                       xy=np.asarray([(p['x'], p['y']) for p in candidates], dtype=np.int64),
                       tissue_fraction=np.asarray([p['tissue_fraction'] for p in candidates],
                                                  dtype=np.float32),
                       scores=scores.astype(np.float32),
                       metadata=np.asarray(json.dumps({'config': config, 'slide': row})))
            record = {
                'config': config, 'slide': stem,
                'truth': 'TUMOR' if row['type'] == 'tumor' else 'NORMAL',
                'subtype': row['class'], 'candidate_pool_before_cap': len(grid),
                'candidates_after_cap': len(candidates),
                'max_raw_mlp_score': float(scores.max()),
                'feature_cache': str(feature_path.relative_to(args.output)),
                'feature_cache_sha256': sha256(feature_path),
                'seconds': time.perf_counter() - slide_started,
            }
            atomic_json(args.output / 'slides' / f'{stem}.json', record)
            prior[stem] = record
            print(f'  max score={record["max_raw_mlp_score"]:.6f}; '
                  f'{record["seconds"]:.1f}s', flush=True)
        del encoder, head
        torch.cuda.empty_cache()
    else:
        model_seconds = 0.0

    records = [prior[Path(r['image']).stem] for r in rows]
    selected, sweep = threshold_sweep(records)
    threshold = selected['threshold']
    gate = {
        'status': 'validation-calibrated experimental gate; apply unchanged to official-test run',
        'threshold': threshold,
        'decision_rule': 'skip GPT and predict NORMAL iff max raw MLP score <= threshold; else call GPT with top 8',
        'checkpoint_sha256': sha256(checkpoint_path),
        'encoder_config_sha256': sha256(encoder_path / 'config.json'),
        'validation_config_sha256': sha256(config_path),
        'validation_scores_sha256': sha256(args.output / 'scores.csv') if (args.output / 'scores.csv').is_file() else None,
        'quantization': QUANTIZATION,
        'mlp': 'FP32, original saved current MLP',
        'sampler': config['sampler'],
        'validation_only': True,
    }
    # Write the score table before hashing it into the gate metadata.
    fields = ['slide', 'truth', 'subtype', 'candidate_pool_before_cap',
              'candidates_after_cap', 'max_raw_mlp_score', 'feature_cache',
              'feature_cache_sha256', 'seconds']
    tmp = args.output / 'scores.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(args.output / 'scores.csv')
    gate['validation_scores_sha256'] = sha256(args.output / 'scores.csv')
    atomic_json(args.output / 'gate.json', gate)
    report = {
        'validation_slides': len(records),
        'positive_slides': sum(r['truth'] == 'TUMOR' for r in records),
        'normal_slides': sum(r['truth'] == 'NORMAL' for r in records),
        'selected_policy': selected,
        'fixed_threshold_0_2': fixed_threshold_result(records, 0.2),
        'lowest_positive': min((r for r in records if r['truth'] == 'TUMOR'),
                               key=lambda r: r['max_raw_mlp_score']),
        'highest_normal': max((r for r in records if r['truth'] == 'NORMAL'),
                              key=lambda r: r['max_raw_mlp_score']),
        'per_slide_inference_seconds_total': sum(r['seconds'] for r in records),
        'model_load_and_quantization_seconds_this_invocation': model_seconds,
        'limitations': [
            'The cutoff is selected and evaluated on the same 50 development-validation slides.',
            'There are only 18 validation-positive slides; zero observed skips is not a guarantee.',
            'Raw MLP scores are uncalibrated regression values, not probabilities.',
            'The threshold applies only to this FP32 MLP, quantized encoder and exact sampler.',
            'No masks, GPT calls, training, or official-test slides are used here.',
        ],
    }
    atomic_json(args.output / 'analysis.json', report)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Gate file for the test runner: {args.output / "gate.json"}', flush=True)


if __name__ == '__main__':
    main()
