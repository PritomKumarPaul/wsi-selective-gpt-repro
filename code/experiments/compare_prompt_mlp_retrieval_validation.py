#!/usr/bin/env python3
"""Matched prompt-vs-MLP patch retrieval comparison on CAMELYON16 validation.

Uses cached CONCH-Z image features. It does not open/decode WSI pixels, encode
patches, train a model, call GPT, or use tumor masks while making selections.
The mask fractions already stored in the cache are read only after both ranking
orders are fixed, for retrieval evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CACHE = Path.home() / 'wsi_patch_fraction_cache_v1'
DEFAULT_FEATURE_CACHE = Path.home() / 'wsi_validation_low_tissue_retrieval_v1'
DEFAULT_PROMPTS = PROJECT / 'code/legacy_repo/prompts_camelyon16.json'
DEFAULT_OUTPUT = PROJECT / 'results/retrieval_prompt_vs_mlp_validation_v1'
PATCH = 1024
TISSUE_MIN = 0.50
CANDIDATE_CAP = 500
TOP_KS = (1, 4, 8, 16, 32)
STRONG_FRACTION = 0.001
EXPECTED_VALIDATION_SLIDES = 50
FEATURE_DIM = 768


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validation_rows(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    rows = [r for r in manifest['slides'] if r['split'] == 'validation']
    if len(rows) != EXPECTED_VALIDATION_SLIDES:
        raise ValueError(f'Expected {EXPECTED_VALIDATION_SLIDES} validation slides, got {len(rows)}.')
    if any(Path(r['image']).stem.startswith('test_') for r in rows):
        raise ValueError('Validation manifest unexpectedly contains a test slide.')
    return manifest, rows


def load_text_classifier(model_dir: Path, prompt_path: Path, device: torch.device):
    """Compute only the small prompt classifier, once; no image patches are encoded."""
    if not (model_dir / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Local CONCH-Z weights not found in {model_dir}; no download attempted.')
    prompt_spec = json.loads(prompt_path.read_text())
    print(f'Loading local CONCH-Z on {device} to encode the fixed text prompts only...', flush=True)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    ).eval().to(device)
    with torch.inference_mode():
        classifier = model.zero_shot_classifier(
            [prompt_spec['classes']['tumor'], prompt_spec['classes']['normal']],
            prompt_spec['templates'], device=device
        ).float().cpu()
    elapsed = time.perf_counter() - started
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if tuple(classifier.shape) != (FEATURE_DIM, 2):
        raise ValueError(f'Unexpected CONCH-Z classifier shape {tuple(classifier.shape)}')
    # Ensure the returned class prototypes are unit vectors for cosine scoring.
    classifier = torch.nn.functional.normalize(classifier, dim=0).numpy().astype(np.float32)
    print(f'Encoded text prompts in {elapsed:.2f}s; image-patch embeddings will be reused.', flush=True)
    return prompt_spec, classifier, elapsed


def capped_candidate_indices(xy: np.ndarray, tissue: np.ndarray,
                             width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the selected validation sampler: one origin grid, 50% tissue, cap 500."""
    one_grid = ((xy[:, 0] % PATCH == 0) & (xy[:, 1] % PATCH == 0)
                & (tissue >= TISSUE_MIN))
    pool_idx = np.flatnonzero(one_grid)
    if len(pool_idx) <= CANDIDATE_CAP:
        return pool_idx, pool_idx

    # This deterministic farthest-point rule matches the cached validation replay:
    # start at the highest-tissue crop; then maximize distance to selected centers.
    pool_xy = xy[pool_idx]
    centers = np.column_stack(((pool_xy[:, 0] + PATCH / 2) / width,
                               (pool_xy[:, 1] + PATCH / 2) / height))
    pool_tissue = tissue[pool_idx]
    used = np.zeros(len(pool_idx), dtype=bool)
    min_dist = np.full(len(pool_idx), np.inf)
    selected = np.empty(CANDIDATE_CAP, dtype=np.int64)
    current = int(np.argmax(pool_tissue))
    for n in range(CANDIDATE_CAP):
        selected[n] = current
        used[current] = True
        min_dist = np.minimum(min_dist, ((centers - centers[current]) ** 2).sum(axis=1))
        min_dist[used] = -1
        current = int(np.argmax(min_dist + 1e-12 * pool_tissue))
    return pool_idx, pool_idx[selected]


def metrics_for_method(rows: list[dict], method: str) -> dict:
    positive = [r for r in rows if r['positive_slide']]
    negative = [r for r in rows if not r['positive_slide']]
    output = {
        'positive_slides': len(positive),
        'negative_slides': len(negative),
        'positive_candidate_pool_coverage': int(sum(r['candidate_pool_has_tumor'] for r in positive)),
        'positive_sampled_set_coverage_after_cap': int(sum(r['sampled_candidates_have_tumor'] for r in positive)),
        'mean_candidates_after_cap': float(np.mean([r['candidate_count'] for r in rows])),
        'top_k': {},
    }
    for k in TOP_KS:
        # method_results[method] is already keyed by the requested K.
        pos = [r['methods'][method][str(k)] for r in positive]
        neg = [r['methods'][method][str(k)] for r in negative]
        crop_count = sum(v['tumor_overlap_crops'] for v in pos)
        strong_count = sum(v['strong_overlap_crops'] for v in pos)
        total_crops = sum(v['selected_crops'] for v in pos)
        output['top_k'][str(k)] = {
            'positive_slides_with_fewer_candidates_than_k': int(sum(r['candidate_count'] < k for r in positive)),
            'negative_slides_with_fewer_candidates_than_k': int(sum(r['candidate_count'] < k for r in negative)),
            'positive_slides_with_any_overlap': int(sum(v['any_overlap'] for v in pos)),
            'positive_slides_total': len(pos),
            'positive_slides_with_strong_overlap': int(sum(v['strong_overlap'] for v in pos)),
            'negative_slides_with_any_overlap': int(sum(v['any_overlap'] for v in neg)),
            'tumor_overlapping_crops_on_positive_slides': int(crop_count),
            'strong_overlapping_crops_on_positive_slides': int(strong_count),
            'selected_crops_on_positive_slides': int(total_crops),
            'crop_overlap_rate': float(crop_count / total_crops) if total_crops else 0.0,
        }
    return output


def evaluate(args) -> dict:
    manifest_path = args.training_cache / 'manifest.json'
    cache_config_path = args.feature_cache / 'config.json'
    if not manifest_path.is_file() or not cache_config_path.is_file():
        raise FileNotFoundError('Training manifest or validation feature-cache config is missing.')
    manifest, rows = validation_rows(manifest_path)
    cache_config = json.loads(cache_config_path.read_text())
    if cache_config.get('source_manifest_sha256') != sha256(manifest_path):
        raise ValueError('Validation feature cache was built from a different manifest.')
    if cache_config.get('encoder_config_sha256') != sha256(Path(manifest['config']['model_dir']) / 'config.json'):
        raise ValueError('Validation cache CONCH-Z config does not match the local model config.')
    if 'unquantized FP32' not in cache_config.get('encoder', ''):
        raise ValueError('Expected the existing unquantized CONCH-Z validation cache.')
    mlp_checkpoint = args.training_cache / 'neural_mlp_seed42' / 'model.pt'
    if not mlp_checkpoint.is_file():
        raise FileNotFoundError(f'Saved current MLP checkpoint missing: {mlp_checkpoint}')
    if cache_config.get('training_checkpoint_sha256') != sha256(mlp_checkpoint):
        raise ValueError('Cached MLP scores were produced by a different checkpoint.')

    model_path = Path(manifest['config']['model_dir'])
    model_config_hash = sha256(model_path / 'config.json')
    device = torch.device(args.device)
    prompt_spec, classifier, text_seconds = load_text_classifier(model_path, args.prompts, device)
    per_slide = []

    for n, row in enumerate(rows, 1):
        stem = Path(row['image']).stem
        cache_path = args.feature_cache / 'features' / f'{stem}.npz'
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata'].item()))
            if metadata.get('slide') != row or metadata.get('config') != cache_config:
                raise ValueError(f'{stem}: slide/cache metadata mismatch.')
            xy = z['xy'].astype(np.int64)
            tissue = z['tissue_fraction'].astype(np.float32)
            # Stored image embeddings are FP16 copies of normalized FP32 encoder
            # outputs; restore FP32 and renormalize for cosine similarity.
            features = z['features'].astype(np.float32)
            mlp_scores = z['scores'].astype(np.float32)
            if features.shape != (len(xy), FEATURE_DIM) or mlp_scores.shape != (len(xy),):
                raise ValueError(f'{stem}: invalid cached feature or score shape.')

            data_dir = Path(manifest['config']['data'])
            from openslide import OpenSlide
            with OpenSlide(str(data_dir / 'images' / row['image'])) as slide:
                width, height = slide.dimensions

            pool_idx, selected_idx = capped_candidate_indices(
                xy, tissue, width, height)

            image_vectors = features[selected_idx]
            image_vectors /= np.maximum(np.linalg.norm(image_vectors, axis=1, keepdims=True), 1e-12)
            similarities = image_vectors @ classifier
            prompt_margin = similarities[:, 0] - similarities[:, 1]
            selected_mlp_scores = mlp_scores[selected_idx]
            prompt_order = np.argsort(-prompt_margin, kind='stable')
            mlp_order = np.argsort(-selected_mlp_scores, kind='stable')

            # Retrieval decisions are now fixed. Only now read the cached mask labels.
            # These are evaluation-only mask statistics; neither set affects ranking.
            all_pool_tumor_fraction = z['tumor_fraction'].astype(np.float32)[pool_idx]
            tumor_fraction = z['tumor_fraction'].astype(np.float32)[selected_idx]

        candidate_pool_has_tumor = bool(np.any(all_pool_tumor_fraction > 0))
        sampled_candidates_have_tumor = bool(np.any(tumor_fraction > 0))
        method_results = {}
        for method, order in (('prompt', prompt_order), ('mlp', mlp_order)):
            top_k = {}
            for k in TOP_KS:
                local = order[:k]
                fractions = tumor_fraction[local]
                top_k[str(k)] = {
                    'requested_k': k,
                    'effective_k': int(len(local)),
                    'selected_crops': int(len(local)),
                    'tumor_overlap_crops': int(np.count_nonzero(fractions > 0)),
                    'strong_overlap_crops': int(np.count_nonzero(fractions >= STRONG_FRACTION)),
                    'any_overlap': bool(np.any(fractions > 0)),
                    'strong_overlap': bool(np.any(fractions >= STRONG_FRACTION)),
                }
            method_results[method] = top_k

        per_slide.append({
            'slide': stem,
            'positive_slide': row['type'] == 'tumor',
            'candidate_pool_before_cap': int(len(pool_idx)),
            'candidate_count': int(len(selected_idx)),
            'candidate_pool_has_tumor': candidate_pool_has_tumor,
            'sampled_candidates_have_tumor': sampled_candidates_have_tumor,
            'methods': method_results,
        })
        candidate_pool_count = int(len(pool_idx))
        print(f'{n}/{len(rows)} {stem}: pool={candidate_pool_count}; '
              f'kept={len(selected_idx)}; prompt top8 hits='
              f'{method_results["prompt"]["8"]["tumor_overlap_crops"]}; '
              f'MLP top8 hits={method_results["mlp"]["8"]["tumor_overlap_crops"]}', flush=True)

    summary = {
        'purpose': 'Matched validation comparison of prompt-based CONCH-Z and trained MLP patch retrieval; not slide diagnosis.',
        'split': 'CAMELYON16 development validation only; 50 slides (18 positive, 32 normal).',
        'sampler': {
            'grid': 'one origin-aligned 1024x1024 grid (offset 0,0), selected from existing two-grid cache',
            'minimum_tissue_fraction': TISSUE_MIN,
            'candidate_cap': CANDIDATE_CAP,
            'top_k_policy': 'Use up to K candidates; if a slide has fewer than K, rank all available candidates. Crop-overlap rates use the actual selected-crop denominator.',
            'cap_method': 'deterministic farthest-point spread seeded by highest tissue fraction',
            'mean_pre_cap_candidates': float(np.mean([r['candidate_pool_before_cap'] for r in per_slide])),
            'mean_candidates_kept': float(np.mean([r['candidate_count'] for r in per_slide])),
        },
        'feature_reuse': {
            'validation_cache': str(args.feature_cache),
            'validation_cache_config_sha256': sha256(cache_config_path),
            'training_manifest_sha256': sha256(manifest_path),
            'mlp_checkpoint_sha256': sha256(mlp_checkpoint),
            'prompt_spec_sha256': sha256(args.prompts),
            'image_features_recomputed': False,
            'cache_encoder': cache_config.get('encoder'),
            'embedding_cache_dtype': 'float16 (original normalized FP32 CONCH-Z outputs; cast to FP32 and renormalized for cosine)',
            'conch_model_config_sha256': model_config_hash,
            'mlp_scores_reused': True,
            'gpt_called': False,
            'mask_policy': 'cached tumor fractions are accessed only after both rankings are fixed; evaluation only',
        },
        'prompt_method': {
            'score': 'cosine(image, tumor class prototype) - cosine(image, normal class prototype)',
            'prompt_spec': prompt_spec,
            'text_encoder_device': str(device),
            'text_prompt_encoding_seconds': text_seconds,
        },
        'metrics': {
            'prompt': metrics_for_method(per_slide, 'prompt'),
            'mlp': metrics_for_method(per_slide, 'mlp'),
        },
        'per_slide': per_slide,
        'interpretation_limits': [
            'This is a retrieval comparison: mask overlap measures whether selected crops intersect annotated tumor, not slide-level diagnosis accuracy.',
            'Both rankers see exactly the same mask-blind candidate pool for each validation slide.',
            'The validation split has been used for prior development and threshold calibration; this is not an untouched benchmark.',
            'Tumor masks are annotation references and may be incomplete/noisy; overlap is not clinical severity.',
        ],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-cache', type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument('--feature-cache', type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument('--prompts', type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cpu', help='cpu by default; optionally cuda:N if approved/available')
    args = parser.parse_args()
    args.training_cache = args.training_cache.expanduser().resolve()
    args.feature_cache = args.feature_cache.expanduser().resolve()
    args.prompts = args.prompts.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(PROJECT):
        parser.error(f'Output must remain inside the project repo: {PROJECT}')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'Output directory already contains files; choose a new --output: {args.output}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but is not available.')
    if args.device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))

    print('Retrieval comparison only: prompt similarity vs saved CONCH-Z + MLP ranking.', flush=True)
    print('Will reuse image embeddings; no WSI image encoding, training, GPT, or mask-guided selection.', flush=True)
    summary = evaluate(args)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'summary.json', summary)
    with (args.output / 'per_slide.csv').open('w', newline='') as f:
        fields = [
            'slide', 'positive_slide', 'candidate_pool_before_cap', 'candidate_count',
            'candidate_pool_has_tumor', 'sampled_candidates_have_tumor',
        ]
        for method in ('prompt', 'mlp'):
            for k in TOP_KS:
                fields.extend([f'{method}_top{k}_overlap_crops', f'{method}_top{k}_strong_crops',
                               f'{method}_top{k}_any_slide_hit', f'{method}_top{k}_strong_slide_hit'])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary['per_slide']:
            out = {k: row[k] for k in fields if k in row}
            for method in ('prompt', 'mlp'):
                for k in TOP_KS:
                    m = row['methods'][method][str(k)]
                    out[f'{method}_top{k}_overlap_crops'] = m['tumor_overlap_crops']
                    out[f'{method}_top{k}_strong_crops'] = m['strong_overlap_crops']
                    out[f'{method}_top{k}_any_slide_hit'] = m['any_overlap']
                    out[f'{method}_top{k}_strong_slide_hit'] = m['strong_overlap']
            writer.writerow(out)
    print(json.dumps(summary['metrics'], indent=2), flush=True)
    print(f'Saved retrieval comparison to {args.output}', flush=True)


if __name__ == '__main__':
    main()
