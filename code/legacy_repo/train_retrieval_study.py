#!/usr/bin/env python3
"""Compare CONCH-Z patch samplers on clean CAMELYON16 training slides.

This script makes NO GPT calls and downloads nothing. The default uses the
original unquantized FP32 model. An INT8 weight-storage reference is optional
for a later comparison; it is not an accelerated INT8 kernel. Each slide is
checkpointed, so an interrupted run resumes without re-encoding it.

Tumor masks are opened only after both candidate lists and all rankings are
fixed. They are used to measure retrieval, never to select a patch.
"""

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import openslide
import torch
from transformers import AutoModel

from camelyon_conchz_pilot import (
    MODEL_REVISION, candidates_from_thumbnail, encode_candidates,
    mask_measure, spatial_shortlist,
)
from conchz_int8_fidelity import replace_linears, state_bytes


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path.home() / 'data' / 'camelyon16_full'
DEFAULT_MODEL = Path.home() / 'models' / 'conch-z'
DEFAULT_FP32_OUTPUT = HERE / 'outputs/train_retrieval_fp32'
DEFAULT_INT8_OUTPUT = HERE / 'outputs/train_retrieval_int8_reference'
PATCH = 1024
MIN_TISSUE = 0.5
TOP_K = 4
VERSION = 3
METHODS = ('spatial', 'hybrid', 'hybrid_diverse')
RANK_K = (1, 2, 4, 8, 16, 32)
STRONG_HIT_FRACTION = 0.001  # Secondary: at least 0.1% of the native patch.


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_features(path, features):
    """Save quantized-encoder outputs once for later slide-level training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, features=features.astype(np.float32))
    temporary.replace(path)


def incomplete_tumor_ids(data):
    """Use the mirror's explicit list of non-exhaustive tumor annotations."""
    readme = data / 'README.md'
    if not readme.is_file():
        raise FileNotFoundError(f'Need annotation completeness list: {readme}')
    names = set(re.findall(r'^\* (tumor_\d+)\b', readme.read_text(), re.M))
    if not names:
        raise ValueError(f'No non-exhaustive tumor list found in {readme}; '
                         'cannot safely select the clean cohort')
    return names


def train_rows(data, slides, per_group, seed, all_clean=False,
               positives_only=False):
    reference = data / 'evaluation/reference.csv'
    with reference.open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if Path(row['image']).stem.startswith(('normal_', 'tumor_'))]
    by_name = {Path(row['image']).stem: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError('Duplicate training slide in reference.csv')
    incomplete = incomplete_tumor_ids(data)
    unknown_exclusions = incomplete - set(by_name)
    if unknown_exclusions:
        raise ValueError(f'Incomplete-list slide(s) absent from reference: '
                         f'{sorted(unknown_exclusions)}')
    if slides:
        unknown = set(slides) - set(by_name)
        if unknown:
            raise ValueError(f'Unknown training slide(s): {sorted(unknown)}')
        forbidden = set(slides) & incomplete
        if forbidden:
            raise ValueError(f'Incomplete tumor masks excluded: {sorted(forbidden)}')
        selected = [by_name[name] for name in dict.fromkeys(slides)]
    elif all_clean:
        selected = [row for row in rows
                    if Path(row['image']).stem not in incomplete]
    else:
        rng = np.random.default_rng(seed)
        selected = []
        for subtype in ('negative', 'micro', 'macro'):
            group = sorted((r for r in rows if r['class'] == subtype
                            and Path(r['image']).stem not in incomplete),
                           key=lambda r: r['image'])
            if len(group) < per_group:
                raise ValueError(f'Only {len(group)} {subtype} training slides')
            selected.extend(group[i] for i in sorted(
                rng.choice(len(group), size=per_group, replace=False)))
    if positives_only:
        selected = [row for row in selected if row['type'] == 'tumor']
    if not selected:
        raise ValueError('No slides selected')
    selected.sort(key=lambda row: row['image'])
    for row in selected:
        stem = Path(row['image']).stem
        for path in (data / 'images' / row['image'],
                     data / 'masks' / f'{stem}_mask.tif'):
            if not path.is_file():
                raise FileNotFoundError(path)
    return selected


def config_for(args):
    return {
        'version': VERSION, 'model': 'MahmoodLab/CONCH-Z',
        'model_revision': MODEL_REVISION, 'model_dir': str(args.model),
        'precision': args.precision,
        'quantization': ('none; original FP32 image and text encoders'
                         if args.precision == 'fp32' else
                         'vision Linear INT8 per-output-channel stored weights; '
                         'FP32 dequantization and dense FP32 matmul; FP32 text'),
        'prompts_sha256': sha256(args.prompts),
        'reference_sha256': sha256(args.data / 'evaluation/reference.csv'),
        'annotation_readme_sha256': sha256(args.data / 'README.md'),
        'incomplete_tumor_ids': sorted(incomplete_tumor_ids(args.data)),
        'patch_size_level0': PATCH, 'thumbnail_max_size': 2048,
        'min_tissue_fraction': MIN_TISSUE,
        'candidate_budget_per_method': args.candidates,
        'hybrid': 'half farthest-point spatial, half thumbnail-texture plus spatial spread',
        'hybrid_texture_weight': 0.45,
        'hybrid_diversity_min_separation_patches': args.separation,
        'top_k': TOP_K, 'rank_k': list(RANK_K),
        'strong_hit_fraction': STRONG_HIT_FRACTION,
        'mask_positive_label': 2,
        'score': 'cosine(tumor prompts)-cosine(normal prompts)',
    }


def read_saved(path, config):
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get('config') != config:
        raise ValueError(f'Saved settings differ: {path}; choose another --output')
    if any(len(record.get('methods', {}).get(method, {}).get('top4', [])) != TOP_K
           for method in METHODS):
        raise ValueError(f'Incomplete saved slide: {path}')
    if any(len(record['methods'][name].get('ranked_candidates', [])) !=
           config['candidate_budget_per_method'] for name in ('spatial', 'hybrid')):
        raise ValueError(f'Incomplete ranked candidates: {path}')
    feature_file = record.get('feature_file')
    if not feature_file or not (path.parent.parent / feature_file).is_file():
        raise ValueError(f'Missing feature cache for saved slide: {path}')
    return record


def texture_values(eligible, thumb, dimensions):
    """Cheap thumbnail texture score, computed without annotations."""
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(gx, gy)
    width, height = dimensions
    thumb_h, thumb_w = gray.shape
    values = []
    for item in eligible:
        x1 = max(0, round(item['x'] * thumb_w / width))
        y1 = max(0, round(item['y'] * thumb_h / height))
        x2 = min(thumb_w, max(x1 + 1, round((item['x'] + PATCH) * thumb_w / width)))
        y2 = min(thumb_h, max(y1 + 1, round((item['y'] + PATCH) * thumb_h / height)))
        tile = gray[y1:y2, x1:x2]
        edge = edges[y1:y2, x1:x2]
        values.append(float(tile.std() + 0.5 * edge.mean()))
    values = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(values, [5, 95])
    return np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)


def candidate_lists(eligible, thumb, dimensions, budget):
    spatial = spatial_shortlist(eligible, budget, *dimensions)
    half = budget // 2
    # The first half is identical to the original farthest-point shortlist.
    chosen = [item.copy() for item in spatial[:half]]
    selected_xy = {(p['x'], p['y']) for p in chosen}
    texture = texture_values(eligible, thumb, dimensions)
    width, height = dimensions
    xy = np.asarray([[(p['x'] + PATCH / 2) / width,
                      (p['y'] + PATCH / 2) / height] for p in eligible])
    seed_idx = [i for i, p in enumerate(eligible)
                if (p['x'], p['y']) in selected_xy]
    min_distance = np.full(len(eligible), np.inf)
    for index in seed_idx:
        min_distance = np.minimum(min_distance,
                                  np.linalg.norm(xy - xy[index], axis=1))
    used = np.zeros(len(eligible), dtype=bool)
    used[seed_idx] = True
    for _ in range(budget - half):
        coverage = np.clip(min_distance / np.sqrt(2), 0, 1)
        score = 0.55 * coverage + 0.45 * texture
        score[used] = -np.inf
        index = int(np.argmax(score))  # eligible grid order breaks ties
        if not np.isfinite(score[index]):
            raise ValueError('Too few distinct eligible tissue positions')
        chosen.append(eligible[index].copy())
        used[index] = True
        min_distance = np.minimum(min_distance,
                                  np.linalg.norm(xy - xy[index], axis=1))
    # IDs are stable within each method; coordinates are the cross-method key.
    for j, patch in enumerate(chosen):
        patch['candidate_id'] = j
    return spatial, chosen


def rank_candidates(candidates, margins):
    return sorted(candidates, key=lambda p: (-margins[(p['x'], p['y'])],
                                             p['candidate_id']))


def ranked_top4(candidates, margins, separation=0):
    ordered = rank_candidates(candidates, margins)
    if separation <= 0:
        return ordered[:TOP_K]
    selected = []
    threshold = separation * PATCH
    for patch in ordered:
        if all(np.hypot(patch['x'] - old['x'], patch['y'] - old['y']) >= threshold
               for old in selected):
            selected.append(patch)
            if len(selected) == TOP_K:
                return selected
    # If the slide is too small, fill with highest-ranked remaining patches.
    selected_xy = {(p['x'], p['y']) for p in selected}
    selected.extend(p for p in ordered if (p['x'], p['y']) not in selected_xy)
    return selected[:TOP_K]


def load_model(args):
    if not (args.model / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Local model weights missing: {args.model}')
    print(f'Loading CONCH-Z ({args.precision}) and encoding text prompts...', flush=True)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True,
                                      local_files_only=True).eval()
    prompts = json.loads(args.prompts.read_text())
    device = torch.device(args.device)
    if args.precision == 'fp32':
        model.to(device).eval()
        with torch.inference_mode():
            classifier = model.zero_shot_classifier(
                [prompts['classes']['tumor'], prompts['classes']['normal']],
                prompts['templates'], device=device).float().cpu()
        info = {'precision': 'fp32',
                'vision_state_bytes': state_bytes(model.visual),
                'vision_linear_layers_converted': 0}
        print(f"FP32 model ready; vision state {info['vision_state_bytes'] / 1e9:.2f} GB; "
              f'{time.perf_counter() - started:.1f}s', flush=True)
        return model, classifier, info
    # The optional reference keeps INT8 weights but dequantizes for FP32 GEMM.
    with torch.inference_mode():
        classifier = model.zero_shot_classifier(
            [prompts['classes']['tumor'], prompts['classes']['normal']],
            prompts['templates'], device=torch.device('cpu')).float().cpu()
    fp32_bytes = state_bytes(model.visual)
    converted = replace_linears(model.visual)
    if converted == 0:
        raise RuntimeError('No vision Linear layers were quantized')
    int8_bytes = state_bytes(model.visual)
    model.to(device).eval()
    print(f'Quantized {converted} vision Linear layers; stored vision state '
          f'{fp32_bytes / 1e9:.2f} -> {int8_bytes / 1e9:.2f} GB; '
          f'load+text+conversion {time.perf_counter() - started:.1f}s', flush=True)
    info = {'precision': 'int8-reference',
            'vision_fp32_state_bytes': fp32_bytes,
            'vision_int8_stored_state_bytes': int8_bytes,
            'vision_linear_layers_converted': converted}
    return model, classifier, info


def run_one(args, row, config, model, classifier):
    stem = Path(row['image']).stem
    started = time.perf_counter()
    slide_path = args.data / 'images' / row['image']
    mask_path = args.data / 'masks' / f'{stem}_mask.tif'
    with openslide.OpenSlide(str(slide_path)) as slide:
        dimensions = list(slide.dimensions)
        eligible, grid_count, thumb, _tissue = candidates_from_thumbnail(
            slide, PATCH, MIN_TISSUE)
        if len(eligible) < args.candidates:
            raise ValueError(f'{stem}: only {len(eligible)} eligible positions; '
                             f'lower --candidates')
        spatial, hybrid = candidate_lists(eligible, thumb, slide.dimensions,
                                          args.candidates)
        union = list({(p['x'], p['y']): p for p in spatial + hybrid}.values())
        print(f'  {len(eligible)} eligible; {len(union)} unique patches to encode', flush=True)
        vectors = encode_candidates(slide, model, union, PATCH,
                                    torch.device(args.device), args.batch_size)
        sims = (vectors @ classifier).numpy()
        margins = {(p['x'], p['y']): float(sims[j, 0] - sims[j, 1])
                   for j, p in enumerate(union)}
        rank_spatial = rank_candidates(spatial, margins)
        rank_hybrid = rank_candidates(hybrid, margins)
        ranked = {
            'spatial': rank_spatial[:TOP_K],
            'hybrid': rank_hybrid[:TOP_K],
            'hybrid_diverse': ranked_top4(hybrid, margins, args.separation),
        }
        # Evaluation begins only after all candidate lists and ranks are fixed.
        with openslide.OpenSlide(str(mask_path)) as mask:
            if mask.dimensions != slide.dimensions:
                raise ValueError(f'{stem}: slide/mask level-0 dimensions differ')
            hits = {}
            for p in union:
                pixels, fraction = mask_measure(mask, p, PATCH)
                hits[(p['x'], p['y'])] = (pixels, fraction)
    def patch_info(p):
        key = (p['x'], p['y'])
        pixels, fraction = hits[key]
        return {'candidate_id': p['candidate_id'], 'x': p['x'], 'y': p['y'],
                'tissue_fraction': p['tissue_fraction'],
                'margin': round(margins[key], 6),
                'tumor_pixels': pixels, 'tumor_fraction': round(fraction, 6),
                'tumor_hit': pixels > 0}
    methods = {}
    for name, pool, top in (
        ('spatial', spatial, ranked['spatial']),
        ('hybrid', hybrid, ranked['hybrid']),
        ('hybrid_diverse', hybrid, ranked['hybrid_diverse']),
    ):
        methods[name] = {
            'candidate_mask_hits': sum(hits[(p['x'], p['y'])][0] > 0 for p in pool),
            'top4': [patch_info(p) for p in top],
            'top4_mask_hits': sum(hits[(p['x'], p['y'])][0] > 0 for p in top),
            'top4_strong_hits': sum(hits[(p['x'], p['y'])][1] >= STRONG_HIT_FRACTION
                                    for p in top),
            'top1_margin': round(max(margins[(p['x'], p['y'])] for p in pool), 6),
        }
    methods['spatial']['ranked_candidates'] = [patch_info(p) for p in rank_spatial]
    methods['hybrid']['ranked_candidates'] = [patch_info(p) for p in rank_hybrid]
    record = {
        'slide': stem, 'reference': {'type': row['type'], 'class': row['class'],
                                     'center': row['center']},
        'config': config, 'slide_dimensions': dimensions,
        'virtual_grid_positions': grid_count,
        'eligible_tissue_positions': len(eligible),
        'unique_patches_encoded': len(union),
        'encoded_candidates': [patch_info(p) for p in union],
        'feature_file': f'features/{stem}.npz',
        'methods': methods,
        'elapsed_seconds': round(time.perf_counter() - started, 3),
    }
    write_features(args.output / record['feature_file'], vectors.numpy())
    write_json(args.output / 'local' / f'{stem}.json', record)
    print(f"  top4 mask hits: spatial={methods['spatial']['top4_mask_hits']}, "
          f"hybrid={methods['hybrid']['top4_mask_hits']}, "
          f"diverse={methods['hybrid_diverse']['top4_mask_hits']}; "
          f"{record['elapsed_seconds']:.1f}s", flush=True)


def summarize(args, rows, config, model_info=None):
    if model_info is None:
        previous = args.output / 'summary.json'
        if previous.is_file():
            prior = json.loads(previous.read_text())
            if prior.get('config') == config:
                model_info = prior.get('model_info')
    records = []
    for row in rows:
        path = args.output / 'local' / f"{Path(row['image']).stem}.json"
        saved = read_saved(path, config)
        if saved:
            records.append(saved)
    summary = {
        'purpose': 'Clean-annotation training-slide retrieval benchmark; no GPT calls',
        'selected_slides': [Path(row['image']).stem for row in rows],
        'completed': len(records), 'selected': len(rows),
        'excluded_incomplete_tumor_slides': config['incomplete_tumor_ids'],
        'positive_slides_selected': sum(r['type'] == 'tumor' for r in rows),
        'negative_slides_selected': sum(r['type'] == 'normal' for r in rows),
        'completed_labels': dict(Counter(r['reference']['class'] for r in records)),
        'quantization': config['quantization'],
        'precision': config['precision'],
        'model_info': model_info,
        'config': config,
        'total_saved_slide_seconds': round(sum(r['elapsed_seconds'] for r in records), 3),
        'methods': {},
        'limitations': [
            'The optional INT8 reference dequantizes for FP32 matmul; it is not accelerated.',
            'Only slides not flagged as non-exhaustively annotated are used.',
            'A seeded subset is exploratory; --all-clean covers all remaining training slides.',
            'Any mask pixel counts as a hit; this is retrieval, not a diagnosis.',
            'Training masks are read after ranks are fixed and do not drive selection.',
            'Top-four coordinates alone do not establish GPT classification accuracy.',
        ],
    }
    csv_rows = []
    for record in records:
        row = {'slide': record['slide'], 'label': record['reference']['type'],
               'subtype': record['reference']['class'],
               'unique_patches_encoded': record['unique_patches_encoded'],
               'seconds': record['elapsed_seconds']}
        for name in METHODS:
            method = record['methods'][name]
            row[f'{name}_candidate_hits'] = method['candidate_mask_hits']
            row[f'{name}_top4_hits'] = method['top4_mask_hits']
            row[f'{name}_top4_strong_hits'] = method['top4_strong_hits']
            row[f'{name}_top1_margin'] = method['top1_margin']
            if name != 'hybrid_diverse':
                for k in RANK_K:
                    row[f'{name}_hit_at_{k}'] = int(any(
                        p['tumor_hit'] for p in method['ranked_candidates'][:k]))
        csv_rows.append(row)
    for name in METHODS:
        positive = [r for r in records if r['reference']['type'] == 'tumor']
        micro = [r for r in positive if r['reference']['class'] == 'micro']
        macro = [r for r in positive if r['reference']['class'] == 'macro']
        entry = {
            'positive_slides': len(positive),
            'positive_candidate_hit_slides': sum(
                r['methods'][name]['candidate_mask_hits'] > 0 for r in positive),
            'positive_top4_hit_slides': sum(
                r['methods'][name]['top4_mask_hits'] > 0 for r in positive),
            'positive_top4_strong_hit_slides': sum(
                r['methods'][name]['top4_strong_hits'] > 0 for r in positive),
            'strong_hit_definition': f'at least {STRONG_HIT_FRACTION:.1%} tumor pixels in patch',
            'micro_slides': len(micro),
            'macro_slides': len(macro),
            'micro_top4_hits': sum(r['methods'][name]['top4_mask_hits'] > 0
                                   for r in micro),
            'macro_top4_hits': sum(r['methods'][name]['top4_mask_hits'] > 0
                                   for r in macro),
        }
        if name != 'hybrid_diverse':
            entry['top_k_hit_slides'] = {
                str(k): {
                    'all': sum(any(p['tumor_hit'] for p in r['methods'][name]
                                   ['ranked_candidates'][:k]) for r in positive),
                    'micro': sum(any(p['tumor_hit'] for p in r['methods'][name]
                                     ['ranked_candidates'][:k]) for r in micro),
                    'macro': sum(any(p['tumor_hit'] for p in r['methods'][name]
                                     ['ranked_candidates'][:k]) for r in macro),
                } for k in RANK_K
            }
            entry['positive_shortlist_miss_slides'] = [
                r['slide'] for r in positive
                if r['methods'][name]['candidate_mask_hits'] == 0]
            entry['positive_ranking_miss_at_4_slides'] = [
                r['slide'] for r in positive
                if r['methods'][name]['candidate_mask_hits'] > 0
                and r['methods'][name]['top4_mask_hits'] == 0]
        summary['methods'][name] = entry
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'summary.json', summary)
    if csv_rows:
        with (args.output / 'per_slide.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Saved {len(records)}/{len(rows)} slide results in {args.output}", flush=True)
    for name, method in summary['methods'].items():
        print(f"  {name}: {method['positive_top4_hit_slides']}/"
              f"{method['positive_slides']} positive slides had a top-4 mask hit", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'local', 'summary'))
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--prompts', type=Path, default=HERE / 'prompts_camelyon16.json')
    parser.add_argument('--output', type=Path,
                        help='Defaults to a separate directory for each precision mode')
    parser.add_argument('--precision', choices=('fp32', 'int8-reference'), default='fp32')
    parser.add_argument('--slides', nargs='+', help='Explicit training slide IDs')
    parser.add_argument('--all-clean', action='store_true',
                        help='Use every training slide except flagged incomplete tumor slides')
    parser.add_argument('--positives-only', action='store_true',
                        help='Restrict the selected cohort to annotated tumor slides')
    parser.add_argument('--per-group', type=int, default=4,
                        help='Seeded sample from normal, micro, macro groups (default: 4 each)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--candidates', type=int, default=128)
    parser.add_argument('--separation', type=float, default=2.0,
                        help='Minimum top-4 center distance in patch widths')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.per_group < 1 or args.candidates < TOP_K * 2 or args.candidates % 2:
        parser.error('Require --per-group >= 1 and even --candidates >= 8')
    if args.batch_size < 1 or args.separation < 0:
        parser.error('Require --batch-size >= 1 and --separation >= 0')
    if args.all_clean and args.slides:
        parser.error('--all-clean and --slides cannot be combined')
    if args.device.startswith('cuda') and args.stage == 'local' and not torch.cuda.is_available():
        parser.error(f'CUDA unavailable: {args.device}')
    if args.output is None:
        args.output = (DEFAULT_FP32_OUTPUT if args.precision == 'fp32'
                       else DEFAULT_INT8_OUTPUT)
    for name in ('data', 'model', 'prompts', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main():
    args = parse_args()
    rows = train_rows(args.data, args.slides, args.per_group, args.seed,
                      args.all_clean, args.positives_only)
    config = config_for(args)
    counts = Counter(r['class'] for r in rows)
    print(f"Selected {len(rows)} clean training slides: {dict(counts)}; "
          f"excluded {len(config['incomplete_tumor_ids'])} incomplete tumor masks",
          flush=True)
    if len(rows) <= 20:
        print(', '.join(f"{Path(r['image']).stem}({r['class']})" for r in rows),
              flush=True)
    if args.stage == 'plan':
        print('Plan only: no slide opened, model loaded, or GPT called.', flush=True)
        return
    if args.stage == 'summary':
        summarize(args, rows, config)
        return
    pending = [r for r in rows if read_saved(
        args.output / 'local' / f"{Path(r['image']).stem}.json", config) is None]
    print(f'{len(rows) - len(pending)} cached; {len(pending)} to process', flush=True)
    model_info = None
    if pending:
        model, classifier, model_info = load_model(args)
        for index, row in enumerate(pending, 1):
            print(f"Training slide {index}/{len(pending)}: {Path(row['image']).stem}", flush=True)
            run_one(args, row, config, model, classifier)
    summarize(args, rows, config, model_info)


if __name__ == '__main__':
    main()
