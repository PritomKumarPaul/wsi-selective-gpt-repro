#!/usr/bin/env python3
"""Matched official CAMELYON16 test retrieval for saved Ridge/linear/MLP heads.

No training, threshold tuning, GPT calls or downloads. `prepare` encodes the
same mask-blind 256 tissue candidates used by the development protocol; masks
only provide evaluation targets after coordinates are fixed. `evaluate` uses
the shared feature cache and frozen saved heads. Run plan/status without GPUs.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from patch_fraction_regression import (
    atomic_json, ensure_manifest, evaluate_regressor, prepare, read_cache,
    storage_guard, used_bytes,
)
from train_patch_neural_head import Predictor, arrays, make_head
from train_retrieval_study import sha256


class SavedScaler:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        if self.mean.shape != (768,) or self.scale.shape != (768,) or np.any(self.scale <= 0):
            raise ValueError('Invalid saved feature scaler')

    def transform(self, x):
        # Match StandardScaler's original float32 in-place transform, retaining
        # its saved float64 statistics and rounding after each operation.
        transformed = np.asarray(x, dtype=np.float32).copy()
        transformed -= self.mean
        transformed /= self.scale
        return transformed


class RidgePredictor:
    def __init__(self, coef, intercept):
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = float(np.asarray(intercept).item())
        if self.coef.shape != (768,):
            raise ValueError('Invalid saved Ridge coefficient shape')

    def predict(self, x):
        return x @ self.coef + self.intercept


def checkpoints(args):
    return {'ridge': args.training / 'regressor.npz',
            'linear': args.training / 'neural_linear_seed42/model.pt',
            'mlp': args.training / 'neural_mlp_seed42/model.pt'}


def load_head(path, kind, device):
    if kind == 'ridge':
        with np.load(path, allow_pickle=False) as z:
            return RidgePredictor(z['coef'], z['intercept']), SavedScaler(z['mean'], z['scale'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['head'] != kind:
        raise ValueError(f'{path}: expected head {kind}')
    model = make_head(kind, checkpoint['hidden'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().to(device)
    scaler = SavedScaler(checkpoint['scaler_mean'].numpy(), checkpoint['scaler_scale'].numpy())
    return Predictor(model, device, 256), scaler


def build_manifest(args):
    source_path = args.training / 'manifest.json'
    source = json.loads(source_path.read_text())
    args.data = Path(source['config']['data'])
    args.model = Path(source['config']['model_dir'])
    args.seed = source['config']['seed']
    args.val_candidates = source['config']['validation_candidates']
    if source['config']['patch_size'] != 1024 or args.val_candidates != 256:
        raise ValueError('Expected the original 1024-pixel/256-candidate training protocol')
    if sha256(args.data / 'evaluation/reference.csv') != source['config']['reference_sha256']:
        raise ValueError('Reference file changed since training')
    if sha256(args.model / 'config.json') != source['config']['model_config_sha256']:
        raise ValueError('Encoder configuration changed since training')
    if any(not r['image'].startswith(('normal_', 'tumor_')) for r in source['slides']):
        raise ValueError('Training manifest contains unexpected slide identifiers')
    with (args.data / 'evaluation/reference.csv').open() as f:
        rows = [dict(r, split='validation') for r in csv.DictReader(f)
                if Path(r['image']).stem.startswith('test_')]
    rows.sort(key=lambda r: r['image'])
    if len(rows) != 129 or len({r['image'] for r in rows}) != 129:
        raise ValueError('Expected 129 distinct test slides in this public mirror')
    if Counter(r['type'] for r in rows) != {'normal': 80, 'tumor': 49}:
        raise ValueError('Unexpected test label distribution')
    for r in rows:
        for path in (args.data / 'images' / r['image'],
                     args.data / 'masks' / (Path(r['image']).stem + '_mask.tif')):
            if not path.is_file():
                raise FileNotFoundError(path)
    hashes = {}
    for kind, path in checkpoints(args).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[kind] = sha256(path)
        if kind != 'ridge':
            config = json.loads((path.parent / 'run_config.json').read_text())
            if config['cache_manifest'] != source:
                raise ValueError(f'{kind} head used a different training feature manifest')
    config = dict(source['config'], evaluation_partition='official_test',
                  training_manifest_sha256=sha256(source_path),
                  frozen_checkpoint_sha256=hashes)
    return source, {'config': config, 'slides': rows}


def evaluate(args, source, manifest):
    ensure_manifest(args, manifest)
    storage_guard(args, 32 * 1024**2)
    records = []
    for row in manifest['slides']:
        cache = read_cache(args, row, manifest)
        if cache is None:
            raise FileNotFoundError(f'Incomplete test cache for {row["image"]}; run prepare first')
        records.append((row, cache))
    # Only reconstruct training labels/weights to retain the same constant
    # baseline. No scaler fitting, optimizer steps or head refitting occurs.
    training_records = []
    for row in source['slides']:
        if row['split'] != 'train':
            continue
        c = read_cache(SimpleNamespace(output=args.training), row, source)
        if c is None:
            raise FileNotFoundError(f'Missing original training feature cache: {row["image"]}')
        training_records.append((row, c))
    _, targets, weights = arrays(training_records)
    comparison = {}
    for kind, path in checkpoints(args).items():
        print(f'Evaluate frozen {kind} head on {len(records)} test WSIs...', flush=True)
        model, scaler = load_head(path, kind, args.device)
        directory = args.output / kind
        directory.mkdir(parents=True, exist_ok=True)
        evaluation_args = SimpleNamespace(output=directory)
        # Reuse precisely the development evaluator, including clipping for MAE
        # but using raw scores for ordering and identical top-K hit definitions.
        evaluation_manifest = {'slides': source['slides'] + manifest['slides']}
        evaluate_regressor(evaluation_args, evaluation_manifest, records,
                           model, scaler, targets, weights, info={
                               'objective': f'fixed saved {kind} tumor-coverage regressor; official test retrieval',
                               'head': kind, 'checkpoint_sha256': sha256(path),
                               'encoder_precision': 'FP32; cached feature storage float16',
                               'parameters_fit_on_test': False,
                           })
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text())
        # The common evaluator calls held-out WSIs validation; label these test.
        summary = {('test_' + k[len('validation_'):] if k.startswith('validation_') else k): v
                   for k, v in summary.items()}
        summary['limitations'] = [s for s in summary['limitations']
                                  if 'development validation split' not in s]
        summary['limitations'].append(
            'Some official test slides were inspected in earlier exploratory zero-shot experiments; '
            'this is not a wholly untouched dataset. These heads were trained only on training WSIs.')
        positive = [r for r, _c in records if r['type'] == 'tumor']
        per_slide = json.loads((directory / 'validation_slides.json').read_text())
        summary['test_subtypes'] = dict(Counter(r['class'] for r, _c in records))
        summary['top4_by_subtype'] = {
            subtype: {'hit_slides': sum(r['top_k']['4']['tumor_hit'] for r in per_slide
                                        if r['class'] == subtype),
                      'positive_slides': sum(r['class'] == subtype for r in positive)}
            for subtype in ('micro', 'macro')}
        atomic_json(summary_path, summary)
        (directory / 'validation_slides.json').replace(directory / 'test_slides.json')
        (directory / 'validation_patches.csv').replace(directory / 'test_patches.csv')
        comparison[kind] = summary
        del model
    atomic_json(args.output / 'comparison.json', comparison)
    with (args.output / 'comparison.csv').open('w', newline='') as f:
        fields = ['head', 'test_positive_slides', 'candidate_hit_positive_slides',
                  'top1_hits', 'top4_hits', 'top8_hits', 'within_slide_auroc', 'pooled_auroc',
                  'average_precision', 'coverage_mae', 'micro_top4_hits', 'macro_top4_hits']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for kind, s in comparison.items():
            writer.writerow({
                'head': kind, 'test_positive_slides': s['test_positive_slides'],
                'candidate_hit_positive_slides': s['candidate_hit_positive_slides'],
                **{f'top{k}_hits': s['top_k'][str(k)]['hit_positive_slides'] for k in (1, 4, 8)},
                'within_slide_auroc': s['mean_within_slide_auroc'],
                'pooled_auroc': s['test_pooled_patch_metrics']['auroc'],
                'average_precision': s['test_pooled_patch_metrics']['average_precision'],
                'coverage_mae': s['test_patch_mae'],
                **{f'{t}_top4_hits': s['top4_by_subtype'][t]['hit_slides'] for t in ('micro', 'macro')},
            })
            print(f'{kind}: top4={s["top_k"]["4"]["hit_positive_slides"]}/{s["test_positive_slides"]}; '
                  f'candidate ceiling={s["candidate_hit_positive_slides"]}/{s["test_positive_slides"]}', flush=True)
    print('Comparison saved:', args.output / 'comparison.csv', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan', 'prepare', 'evaluate', 'status'))
    p.add_argument('--training', type=Path, default=Path.home() / 'wsi_patch_fraction_cache_v1')
    p.add_argument('--output', type=Path, default=Path.home() / 'wsi_patch_fraction_test_v1')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-cache-gb', type=float, default=2.0)
    p.add_argument('--min-free-gb', type=float, default=2.0)
    args = p.parse_args()
    args.training, args.output = args.training.expanduser().resolve(), args.output.expanduser().resolve()
    if (not args.output.is_relative_to(Path.home())
            or args.output == args.training or args.output.is_relative_to(args.training)
            or args.training.is_relative_to(args.output)):
        p.error('Use a separate test output under the user home directory, outside the training cache')
    if args.batch_size < 1 or args.max_cache_gb <= 0 or args.min_free_gb < 0:
        p.error('Invalid batch size or storage limits')
    source, manifest = build_manifest(args)
    ensure_manifest(args, manifest)
    print('Official mirror test:', dict(Counter(r['class'] for r in manifest['slides'])), flush=True)
    print(f'Same {args.val_candidates} mask-blind candidates per WSI for all 3 heads; '
          f'feature storage <= {len(manifest["slides"])*args.val_candidates*768*2/1e6:.1f} MB plus metadata.', flush=True)
    if args.stage in ('plan', 'status'):
        complete = sum(read_cache(args, r, manifest) is not None for r in manifest['slides'])
        print(f'Cached {complete}/{len(manifest["slides"])} slides; '
              f'{used_bytes(args.output)/1e6:.1f} MB. Output: {args.output}', flush=True)
        print('No training, image encoding or output writes.')
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    if args.stage == 'prepare':
        # Every test row has the common sampler's validation role: labels never
        # determine crop locations. Its mask-guided training branch is unused.
        prepare(args, manifest)
    else:
        evaluate(args, source, manifest)


if __name__ == '__main__':
    main()
