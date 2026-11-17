#!/usr/bin/env python3
"""Validation-only all-GPT run using FP32 CONCH-Z + the FP32 MLP.

Replays the established one-grid sampler (>=50% tissue; cap 500) against the
existing unquantized CONCH-Z/FP32-MLP validation cache, then sends its top 8
patches to GPT. No CONCH-Z embeddings are recomputed in this run. Cached tumor
fractions are inspected only after top-8 ranking to count annotation overlaps.
There is deliberately no score gate: GPT is called once for every validation
slide. Annotation fractions and slide labels are excluded from the GPT input.

The script asks for the API key once without echoing or saving it. It writes
per-slide local records, GPT responses, CSV/JSON progress, and stops after an
ambiguous API error or invalid response to avoid unintended duplicate calls.
"""
from __future__ import annotations

import argparse
import base64
import csv
import getpass
import hashlib
import io
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image



HERE = Path(__file__).resolve().parent
DEFAULT_TRAINING = Path.home() / 'wsi_patch_fraction_cache_v1'
DEFAULT_DATA = Path.home() / 'data' / 'camelyon16_full'
DEFAULT_FEATURE_CACHE = Path.home() / 'wsi_validation_low_tissue_retrieval_v1'
DEFAULT_OUTPUT = HERE.parents[1] / 'results/validation_fp32_mlp_always_gpt_v2'
DEFAULT_GATE = Path.home() / 'mlp_validation_gate_quantized_v1' / 'gate.json'

PATCH = 1024
CAP = 500
TOP_K = 8
TISSUE_CUTOFF = 0.50
IMAGE_SIDE = 512
JPEG_QUALITY = 95
GPT_MODEL = 'gpt-5.4-mini-2026-03-17'
MAX_OUTPUT_TOKENS = 32
INPUT_RATE = 0.75
CACHED_INPUT_RATE = 0.075
OUTPUT_RATE = 4.50
BUDGET_USD = 6.0
# Conservative upper allowance per request, used to stop before a request
# when the remaining run budget is too small. Actual cost is recorded from API usage.
REQUEST_BUDGET_ALLOWANCE_USD = 0.02
ENCODER_PRECISION = 'FP32; no quantization'
PROMPT = (
    'Classify the lymph-node slide from these eight H&E crops. Return TUMOR '
    'if metastatic carcinoma is visibly present in any crop; otherwise return '
    'NORMAL based on the provided crops. Do not infer tumor from ordinary '
    'lymph-node tissue. These crops are selected samples, not the complete '
    'slide. Return only JSON with one field: class, whose value is NORMAL or TUMOR.'
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_setup(args):
    manifest_path = args.training / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    checkpoint_path = args.training / 'neural_mlp_seed42' / 'model.pt'
    checkpoint_config_path = checkpoint_path.parent / 'run_config.json'
    checkpoint_config = json.loads(checkpoint_config_path.read_text())
    if checkpoint_config.get('head') != 'mlp' or checkpoint_config.get('cache_manifest') != manifest:
        raise ValueError('Saved MLP checkpoint/config does not match the training manifest.')

    encoder_path = Path(manifest['config']['model_dir'])
    encoder_config = encoder_path / 'config.json'
    if sha256(encoder_config) != manifest['config']['model_config_sha256']:
        raise ValueError('CONCH-Z configuration differs from the MLP training manifest.')

    gate = json.loads(args.protocol_reference.read_text())
    if gate.get('checkpoint_sha256') != sha256(checkpoint_path):
        raise ValueError('Reference metadata uses a different MLP checkpoint.')
    if gate.get('encoder_config_sha256') != sha256(encoder_config):
        raise ValueError('Reference metadata uses a different CONCH-Z config.')
    if gate.get('mlp') != 'FP32, original saved current MLP':
        raise ValueError('Reference metadata does not match the saved FP32 MLP.')
    expected_sampler = {
        'patch': PATCH,
        'grid': 'one (0,0) grid, stride 1024; complete crops only',
        'tissue_cutoff': TISSUE_CUTOFF,
        'candidate_cap': CAP,
        'cap_rule': 'farthest-point spatial spread seeded by highest tissue fraction',
    }
    if gate.get('sampler') != expected_sampler:
        raise ValueError('Reference metadata sampler does not match this script.')

    feature_config_path = args.feature_cache / 'config.json'
    feature_config = json.loads(feature_config_path.read_text())
    if feature_config.get('source_manifest_sha256') != sha256(manifest_path):
        raise ValueError('Cached embeddings were built from a different slide manifest.')
    if feature_config.get('training_checkpoint_sha256') != sha256(checkpoint_path):
        raise ValueError('Cached MLP scores were built from a different checkpoint.')
    if feature_config.get('encoder_config_sha256') != sha256(encoder_config):
        raise ValueError('Cached embeddings use a different CONCH-Z model configuration.')
    if 'unquantized FP32' not in feature_config.get('encoder', ''):
        raise ValueError('Expected existing unquantized FP32 CONCH-Z embeddings.')
    if 'original FP32 weights' not in feature_config.get('mlp', ''):
        raise ValueError('Expected scores from the saved unquantized FP32 MLP.')

    data = args.data or Path(manifest['config']['data'])
    reference = data / 'evaluation' / 'reference.csv'
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if sha256(reference) != manifest['config']['reference_sha256']:
        raise ValueError('Dataset reference labels do not match the saved development manifest.')
    with reference.open(newline='') as f:
        reference_rows = {Path(r['image']).stem: r for r in csv.DictReader(f)}
    rows = [r for r in manifest['slides'] if r['split'] == 'validation']
    rows.sort(key=lambda r: r['image'])
    if len(rows) != 50 or Counter(r['type'] for r in rows) != {'normal': 32, 'tumor': 18}:
        raise ValueError('Expected the fixed 50-slide validation cohort (32 normal, 18 tumor).')
    for row in rows:
        stem = Path(row['image']).stem
        if stem not in reference_rows:
            raise ValueError(f'{stem} missing from dataset reference labels.')
        ref = reference_rows[stem]
        if ref['type'] != row['type'] or ref['class'] != row['class']:
            raise ValueError(f'{stem}: manifest and reference label mismatch.')
        if not (data / 'images' / row['image']).is_file():
            raise FileNotFoundError(data / 'images' / row['image'])

    config = {
        'protocol': 'CAMELYON16 validation; FP32 CONCH-Z + frozen FP32 MLP; always GPT; no score gate',
        'split': 'development validation: 50 slides (32 normal, 18 tumor)',
        'training_manifest_sha256': sha256(manifest_path),
        'mlp_checkpoint_sha256': sha256(checkpoint_path),
        'conch_config_sha256': sha256(encoder_config),
        'encoder_precision': ENCODER_PRECISION,
        'feature_cache': str(args.feature_cache),
        'feature_cache_config_sha256': sha256(feature_config_path),
        'mlp': 'saved current MLP, FP32; no training or quantization',
        'sampler': {
            'patch_pixels_level0': PATCH,
            'grid': 'single (0,0) grid; stride 1024; complete crops only',
            'thumbnail_tissue_cutoff': TISSUE_CUTOFF,
            'candidate_cap': CAP,
            'cap_method': 'farthest-point spatial spread; seed highest tissue fraction',
        },
        'ranking': 'maximum raw MLP coverage-regression score; select top 8',
        'embedding_cache': 'reuse existing all-candidate unquantized CONCH-Z embeddings and FP32 MLP scores; no encoding in this run',
        'gpt': {
            'model': GPT_MODEL,
            'calls': 'exactly one per validation slide; no score gate',
            'source_crop': [PATCH, PATCH],
            'uploaded_image': [IMAGE_SIDE, IMAGE_SIDE],
            'jpeg_quality': JPEG_QUALITY,
            'image_detail': 'high',
            'reasoning_effort': 'none',
            'max_output_tokens': MAX_OUTPUT_TOKENS,
            'prompt': PROMPT,
        },
        'ground_truth': 'slide labels score GPT; cached tumor fractions score top-8 retrieval after ranking; neither is sent to GPT',
        'budget_usd': BUDGET_USD,
        'budget_allowance_per_call_usd': REQUEST_BUDGET_ALLOWANCE_USD,
        'script_sha256': sha256(Path(__file__)),
    }
    for row in rows:
        feature_path = args.feature_cache / 'features' / f"{Path(row['image']).stem}.npz"
        if not feature_path.is_file():
            raise FileNotFoundError(f'Missing existing validation feature cache: {feature_path}')
    return manifest, rows, data, checkpoint_path, config, feature_config


def cap_cached_candidates(xy, tissue, width, height):
    """Apply the final single-grid, >=50%-tissue, spatial-cap sampler to cached rows."""
    on_grid = ((xy[:, 0] % PATCH == 0) & (xy[:, 1] % PATCH == 0)
               & (tissue >= TISSUE_CUTOFF))
    pool = np.flatnonzero(on_grid)
    if len(pool) <= CAP:
        return pool, pool
    coords = xy[pool]
    centers = np.column_stack(((coords[:, 0] + PATCH / 2) / width,
                               (coords[:, 1] + PATCH / 2) / height))
    pool_tissue = tissue[pool]
    used = np.zeros(len(pool), dtype=bool)
    min_distance = np.full(len(pool), np.inf)
    selected = np.empty(CAP, dtype=np.int64)
    current = int(np.argmax(pool_tissue))
    for n in range(CAP):
        selected[n] = current
        used[current] = True
        min_distance = np.minimum(min_distance, ((centers - centers[current]) ** 2).sum(axis=1))
        min_distance[used] = -1
        current = int(np.argmax(min_distance + 1e-12 * pool_tissue))
    return pool, pool[selected]


def run_local(row, data, feature_cache, output, config, feature_config):
    stem = Path(row['image']).stem
    image_path = data / 'images' / row['image']
    started = time.perf_counter()
    cache_path = feature_cache / 'features' / f'{stem}.npz'
    with np.load(cache_path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata'].item()))
        if metadata.get('config') != feature_config or metadata.get('slide') != row:
            raise ValueError(f'{stem}: validation feature cache metadata mismatch.')
        xy = z['xy'].astype(np.int64)
        features = z['features'].astype(np.float32)
        tissue = z['tissue_fraction'].astype(np.float32)
        scores_all = z['scores'].astype(np.float32)
        if features.shape != (len(xy), 768) or tissue.shape != (len(xy),) or scores_all.shape != (len(xy),):
            raise ValueError(f'{stem}: cached embeddings, tissue fractions, or MLP scores have invalid shapes.')
        with openslide.OpenSlide(str(image_path)) as slide:
            width, height = slide.dimensions
            mpp_x = mpp_y = None
            try:
                mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
                mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
            except (KeyError, ValueError):
                pass
        pool_idx, candidate_idx = cap_cached_candidates(xy, tissue, width, height)
        if len(candidate_idx) < TOP_K:
            raise ValueError(f'{stem}: only {len(candidate_idx)} eligible cached candidates; need {TOP_K}.')
        candidate_scores = scores_all[candidate_idx]
        if not np.isfinite(candidate_scores).all():
            raise ValueError(f'{stem}: non-finite cached MLP scores.')
        ranked_local = np.argsort(-candidate_scores, kind='stable')
        top_local = ranked_local[:TOP_K]
        top_indices = candidate_idx[top_local]
        top8 = []
        for rank, cache_index in enumerate(top_indices, 1):
            top8.append({'rank': rank, 'x_level0': int(xy[cache_index, 0]),
                         'y_level0': int(xy[cache_index, 1]),
                         'mlp_score': float(scores_all[cache_index]),
                         'tissue_fraction': float(tissue[cache_index])})
        # Ranking is fixed before the mask-derived fields are accessed.
        tumor_fraction = z['tumor_fraction'].astype(np.float32)
        if tumor_fraction.shape != (len(xy),):
            raise ValueError(f'{stem}: cached mask-fraction vector has invalid shape.')
        top_mask_fractions = tumor_fraction[top_indices]
        for patch, fraction in zip(top8, top_mask_fractions):
            patch['tumor_fraction_eval_only'] = float(fraction)
        top8_mask_hits = int(np.count_nonzero(top_mask_fractions > 0))
        top8_strong_mask_hits = int(np.count_nonzero(top_mask_fractions >= 0.001))

    record = {
        'config': config,
        'slide': stem,
        'truth_for_scoring_only': 'TUMOR' if row['type'] == 'tumor' else 'NORMAL',
        'source_image': row['image'],
        'slide_dimensions_level0': [width, height],
        'mpp': [mpp_x, mpp_y],
        'candidate_pool_before_cap': int(len(pool_idx)),
        'candidates_encoded': len(candidate_idx),
        'max_raw_mlp_score': float(np.max(candidate_scores)),
        'top8': top8,
        'local_seconds': time.perf_counter() - started,
        'local_timing': {'cache_load_and_ranking_seconds': time.perf_counter() - started},
        'mask_fraction_read_after_ranking': True,
        'mask_used_for_selection': False,
        'top8_mask_hit_count_any_overlap': top8_mask_hits,
        'top8_mask_hit_count_at_least_0_1_percent': top8_strong_mask_hits,
        'source_feature_cache': str(cache_path),
        'source_encoder_precision': feature_config.get('encoder'),
        'source_embedding_storage_dtype': 'float16 cache of normalized unquantized FP32 CONCH-Z output',
    }
    atomic_json(output / 'local' / f'{stem}.json', record)
    return record


def make_gpt_content(local, data):
    content = [{'type': 'input_text', 'text': PROMPT}]
    hashes = []
    width, height = local['slide_dimensions_level0']
    source = data / 'images' / local['source_image']
    with openslide.OpenSlide(str(source)) as slide:
        if list(slide.dimensions) != [width, height]:
            raise ValueError(f'{local["slide"]}: source WSI dimensions changed.')
        for p in local['top8']:
            x, y = int(p['x_level0']), int(p['y_level0'])
            rgb = np.asarray(slide.read_region((x, y), 0, (PATCH, PATCH)).convert('RGB'))
            resized = cv2.resize(rgb, (IMAGE_SIDE, IMAGE_SIDE), interpolation=cv2.INTER_AREA)
            buffer = io.BytesIO()
            Image.fromarray(resized).save(buffer, format='JPEG', quality=JPEG_QUALITY)
            image_bytes = buffer.getvalue()
            hashes.append(hashlib.sha256(image_bytes).hexdigest())
            b64 = base64.b64encode(image_bytes).decode('ascii')
            if local['mpp'][0] is None or local['mpp'][1] is None:
                scale = (f'Field {p["rank"]}; source crop 1024x1024 level-0 pixels; '
                         'physical pixel spacing unavailable.')
            else:
                mx, my = local['mpp']
                size_x, size_y = PATCH * mx / 1000, PATCH * my / 1000
                origin_x, origin_y = x * mx / 1000, y * my / 1000
                slide_mm_x, slide_mm_y = width * mx / 1e6, height * my / 1e6
                scale = (f'Field {p["rank"]}; source crop 1024x1024 level-0 pixels '
                         f'({size_x:.3f} x {size_y:.3f} mm); crop origin '
                         f'({origin_x:.2f}, {origin_y:.2f}) mm from slide top-left; '
                         f'whole-slide size ({slide_mm_x:.1f} x {slide_mm_y:.1f}) mm; '
                         f'uploaded resized image is {IMAGE_SIDE}x{IMAGE_SIDE}.')
            content.extend([{'type': 'input_text', 'text': scale},
                            {'type': 'input_image', 'detail': 'high',
                             'image_url': 'data:image/jpeg;base64,' + b64}])
    return content, hashes


def parse_answer(raw):
    try:
        value = json.loads(raw or '').get('class')
    except (json.JSONDecodeError, AttributeError):
        value = None
    if value in ('NORMAL', 'TUMOR'):
        return value
    return None


def request_gpt(client, local, data, output):
    path = output / 'gpt' / f'{local["slide"]}.json'
    content, image_hashes = make_gpt_content(local, data)
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved.get('config_sha256') != sha256(output / 'run_config.json')
                or saved.get('image_sha256') != image_hashes
                or saved.get('prompt_sha256') != prompt_hash):
            raise ValueError(f'{local["slide"]}: saved GPT record does not match inputs/config.')
        if saved.get('state') == 'COMPLETE' and saved.get('answer') in ('NORMAL', 'TUMOR'):
            return saved
        raise RuntimeError(f'{local["slide"]}: prior GPT state is {saved.get("state")}; '
                           'inspect the record rather than risk a duplicate charge.')

    saved = {'slide': local['slide'], 'state': 'IN_FLIGHT', 'model': GPT_MODEL,
             'config_sha256': sha256(output / 'run_config.json'),
             'image_sha256': image_hashes, 'prompt_sha256': prompt_hash}
    atomic_json(path, saved)
    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=GPT_MODEL,
            input=[{'role': 'user', 'content': content}],
            store=False,
            reasoning={'effort': 'none'},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            text={'format': {'type': 'json_schema', 'name': 'camelyon_binary_class',
                 'strict': True, 'schema': {'type': 'object', 'properties': {
                     'class': {'type': 'string', 'enum': ['NORMAL', 'TUMOR']}},
                     'required': ['class'], 'additionalProperties': False}}},
        )
    except Exception as exc:
        saved.update(state='ERROR_OR_UNKNOWN_BILLING', error_type=type(exc).__name__,
                     status_code=getattr(exc, 'status_code', None),
                     elapsed_seconds=time.perf_counter() - started)
        atomic_json(path, saved)
        raise RuntimeError('GPT request failed; stopped with no automatic retry to avoid a duplicate charge.') from None

    usage = getattr(response, 'usage', None)
    if usage is None:
        saved.update(state='MISSING_USAGE', response_id=getattr(response, 'id', None),
                     raw_answer=getattr(response, 'output_text', None),
                     elapsed_seconds=time.perf_counter() - started)
        atomic_json(path, saved)
        raise RuntimeError('GPT response had no usage record; stopped before another request.')
    usage_dict = usage.model_dump()
    input_tokens = int(usage.input_tokens or 0)
    output_tokens = int(usage.output_tokens or 0)
    cached = int(getattr(getattr(usage, 'input_tokens_details', None), 'cached_tokens', 0) or 0)
    cost = ((input_tokens - cached) * INPUT_RATE + cached * CACHED_INPUT_RATE
            + output_tokens * OUTPUT_RATE) / 1_000_000
    raw = getattr(response, 'output_text', '') or ''
    answer = parse_answer(raw)
    status = getattr(response, 'status', None)
    incomplete = getattr(getattr(response, 'incomplete_details', None), 'reason', None)
    complete = answer is not None and incomplete is None and status in (None, 'completed')
    saved.update(
        state='COMPLETE' if complete else 'INVALID_OUTPUT',
        answer=answer or 'INVALID', raw_answer=raw,
        response_id=getattr(response, 'id', None), response_model=getattr(response, 'model', None),
        usage=usage_dict, cost_usd=cost, gpt_seconds=time.perf_counter() - started,
        response_status=status, incomplete_reason=incomplete,
    )
    atomic_json(path, saved)
    if not complete:
        raise RuntimeError(f'{local["slide"]}: invalid/incomplete GPT output saved; stopped before next call.')
    return saved


def aggregate(rows, local_records, gpt_records):
    joined = []
    for row in rows:
        stem = Path(row['image']).stem
        local = local_records.get(stem)
        gpt = gpt_records.get(stem)
        if local is None:
            continue
        truth = 'TUMOR' if row['type'] == 'tumor' else 'NORMAL'
        prediction = gpt.get('answer') if gpt and gpt.get('state') == 'COMPLETE' else None
        joined.append({
            'slide': stem, 'truth': truth, 'prediction': prediction,
            'correct': prediction == truth if prediction else None,
            'max_raw_mlp_score': local['max_raw_mlp_score'],
            'candidate_pool_before_cap': local['candidate_pool_before_cap'],
            'candidates_encoded': local['candidates_encoded'],
            'top8': json.dumps(local['top8'], separators=(',', ':')),
            'local_seconds': local['local_seconds'],
            'gpt_seconds': gpt.get('gpt_seconds') if gpt else None,
            'input_tokens': (gpt.get('usage') or {}).get('input_tokens') if gpt else None,
            'output_tokens': (gpt.get('usage') or {}).get('output_tokens') if gpt else None,
            'cost_usd': gpt.get('cost_usd') if gpt else None,
            'gpt_status': gpt.get('state') if gpt else 'PENDING',
            'top8_mask_hit_count_any_overlap': local['top8_mask_hit_count_any_overlap'],
            'top8_mask_hit_count_at_least_0_1_percent': local['top8_mask_hit_count_at_least_0_1_percent'],
            'top8_mask_fractions_eval_only': json.dumps(
                [p['tumor_fraction_eval_only'] for p in local['top8']], separators=(',', ':')),
        })
    completed = [r for r in joined if r['prediction'] in ('NORMAL', 'TUMOR')]
    tp = sum(r['truth'] == 'TUMOR' and r['prediction'] == 'TUMOR' for r in completed)
    tn = sum(r['truth'] == 'NORMAL' and r['prediction'] == 'NORMAL' for r in completed)
    fp = sum(r['truth'] == 'NORMAL' and r['prediction'] == 'TUMOR' for r in completed)
    fn = sum(r['truth'] == 'TUMOR' and r['prediction'] == 'NORMAL' for r in completed)
    n_normal = sum(r['truth'] == 'NORMAL' for r in completed)
    n_tumor = sum(r['truth'] == 'TUMOR' for r in completed)
    sum_cost = sum(float(r['cost_usd'] or 0) for r in joined)
    positive_rows = [r for r in joined if r['truth'] == 'TUMOR']
    total_positive_top8 = sum(r['top8_mask_hit_count_any_overlap'] for r in positive_rows)
    total_positive_patches = TOP_K * len(positive_rows)
    summary = {
        'protocol': 'validation all-GPT; cached unquantized FP32 CONCH-Z + FP32 MLP top-8; no gate',
        'planned_slides': len(rows), 'local_results': len(joined),
        'completed_gpt_responses': len(completed), 'invalid_or_pending_responses': len(joined) - len(completed),
        'gpt_calls': sum(r['gpt_status'] in ('COMPLETE', 'INVALID_OUTPUT', 'MISSING_USAGE',
                                               'ERROR_OR_UNKNOWN_BILLING', 'IN_FLIGHT') for r in joined),
        'always_call_gpt': True,
        'accuracy': sum(r['correct'] is True for r in completed) / len(completed) if completed else None,
        'false_positives': fp,
        'false_positive_rate_on_normal_slides': fp / n_normal if n_normal else None,
        'false_negative_count': fn,
        'sensitivity_tumor': tp / n_tumor if n_tumor else None,
        'specificity_normal': tn / n_normal if n_normal else None,
        'precision_tumor': tp / (tp + fp) if (tp + fp) else None,
        'positive_slides_with_any_tumor_mask_hit_in_top8': sum(
            r['top8_mask_hit_count_any_overlap'] > 0 for r in positive_rows),
        'positive_slides_total': len(positive_rows),
        'positive_slides_with_strong_top8_mask_hit_ge_0_1_percent': sum(
            r['top8_mask_hit_count_at_least_0_1_percent'] > 0 for r in positive_rows),
        'positive_slide_top8_mask_overlapping_crops': total_positive_top8,
        'positive_slide_top8_mask_overlap_crop_rate': total_positive_top8 / total_positive_patches
            if total_positive_patches else None,
        'positive_slide_top8_strong_mask_overlapping_crops_ge_0_1_percent': sum(
            r['top8_mask_hit_count_at_least_0_1_percent'] for r in positive_rows),
        'confusion_matrix_truth_rows_prediction_columns': ['NORMAL', 'TUMOR'],
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'api_cost_estimate_usd': sum_cost,
        'api_input_tokens': sum(int((r['input_tokens'] or 0)) for r in joined),
        'api_output_tokens': sum(int((r['output_tokens'] or 0)) for r in joined),
        'interpretation_note': 'Validation labels score predictions only; tumor-mask fractions are read after MLP top-8 ranking for retrieval analysis and are never sent to GPT. '
                              'This split was also used for gate calibration and prior development, '
                              'so this is a development comparison, not an independent test.',
    }
    return joined, summary


def save_progress(output, rows, local_records, gpt_records):
    joined, summary = aggregate(rows, local_records, gpt_records)
    fields = ['slide', 'truth', 'prediction', 'correct', 'max_raw_mlp_score',
              'candidate_pool_before_cap', 'candidates_encoded', 'top8',
              'top8_mask_hit_count_any_overlap', 'top8_mask_hit_count_at_least_0_1_percent',
              'top8_mask_fractions_eval_only', 'local_seconds', 'gpt_seconds', 'input_tokens', 'output_tokens',
              'cost_usd', 'gpt_status']
    atomic_csv(output / 'per_slide.csv', joined, fields)
    atomic_json(output / 'summary.json', summary)
    return summary


def load_records(output, rows, config):
    local_records, gpt_records = {}, {}
    for row in rows:
        stem = Path(row['image']).stem
        lp = output / 'local' / f'{stem}.json'
        gp = output / 'gpt' / f'{stem}.json'
        if lp.exists():
            rec = json.loads(lp.read_text())
            if rec.get('config') != config:
                raise ValueError(f'{stem}: local cache belongs to a different run config.')
            source_feature_cache = Path(rec.get('source_feature_cache', ''))
            if not source_feature_cache.is_file():
                raise FileNotFoundError(f'{stem}: expected reusable source feature cache is missing: {source_feature_cache}')
            local_records[stem] = rec
        if gp.exists():
            rec = json.loads(gp.read_text())
            if rec.get('config_sha256') != sha256(output / 'run_config.json'):
                raise ValueError(f'{stem}: GPT cache belongs to a different run config.')
            gpt_records[stem] = rec
    return local_records, gpt_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=DEFAULT_TRAINING)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--feature-cache', type=Path, default=DEFAULT_FEATURE_CACHE,
                        help='Existing unquantized CONCH-Z validation cache; embeddings are reused, not regenerated.')
    parser.add_argument('--protocol-reference', type=Path, default=DEFAULT_GATE,
                        help='Prior metadata used to verify the matching MLP and sampler; no score gate is applied.')
    parser.add_argument('--budget-usd', type=float, default=BUDGET_USD)
    parser.add_argument('--request-budget-allowance-usd', type=float,
                        default=REQUEST_BUDGET_ALLOWANCE_USD)
    args = parser.parse_args()
    args.training = args.training.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.feature_cache = args.feature_cache.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.protocol_reference = args.protocol_reference.expanduser().resolve()
    if not args.output.is_relative_to(HERE.parents[1]):
        parser.error(f'Output must be under the repository: {HERE.parents[1]}')
    if args.budget_usd <= 0 or args.request_budget_allowance_usd <= 0:
        parser.error('Budget values must be positive.')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    manifest, rows, data, checkpoint_path, config, feature_config = read_setup(args)
    args.output.mkdir(parents=True, exist_ok=True)
    run_config_path = args.output / 'run_config.json'
    if run_config_path.exists():
        saved = json.loads(run_config_path.read_text())
        if saved != config:
            raise ValueError('Output directory contains a different run config; choose a new output path.')
    else:
        atomic_json(run_config_path, config)

    local_records, gpt_records = load_records(args.output, rows, config)
    pending_gpt = [r for r in rows
                   if not (Path(r['image']).stem in gpt_records
                           and gpt_records[Path(r['image']).stem].get('state') == 'COMPLETE'
                           and gpt_records[Path(r['image']).stem].get('answer') in ('NORMAL', 'TUMOR'))]
    if pending_gpt and not sys.stdin.isatty():
        raise RuntimeError('Run in an interactive terminal; API key is prompted once and never saved.')
    if pending_gpt:
        api_key = getpass.getpass('OpenAI API key (hidden; not saved): ')
        if not api_key.strip():
            raise RuntimeError('No API key entered.')
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip())
        del api_key
    else:
        client = None

    logging.info('Validation all-GPT run: 50 slides (32 normal, 18 tumor); one GPT request per slide.')
    logging.info('Reusing existing unquantized CONCH-Z embeddings + FP32 MLP scores; top-8; no score gate.')
    logging.info('Mask fractions are evaluation-only and are read after top-8 ranking.')
    budget_allowance = args.request_budget_allowance_usd

    for index, row in enumerate(rows, 1):
        stem = Path(row['image']).stem
        local = local_records.get(stem)
        if local is None:
            local = run_local(row, data, args.feature_cache, args.output,
                              config, feature_config)
            local_records[stem] = local
            logging.info('%d/50 %s local results reused: candidates=%d max_score=%.6f; '
                         'top8 mask hits=%d (>=0.1%%: %d)',
                         index, stem, local['candidates_encoded'], local['max_raw_mlp_score'],
                         local['top8_mask_hit_count_any_overlap'],
                         local['top8_mask_hit_count_at_least_0_1_percent'])
            save_progress(args.output, rows, local_records, gpt_records)

        gpt = gpt_records.get(stem)
        if gpt is None or gpt.get('state') != 'COMPLETE':
            _, current_summary = aggregate(rows, local_records, gpt_records)
            if current_summary['api_cost_estimate_usd'] + budget_allowance > args.budget_usd:
                raise RuntimeError(
                    f'Budget guard stopped before {stem}: recorded estimated API cost plus '
                    f'${budget_allowance:.3f} request allowance exceeds ${args.budget_usd:.2f}.'
                )
            logging.info('GPT request %d/50: %s (always-call mode)', index, stem)
            gpt = request_gpt(client, local, data, args.output)
            gpt_records[stem] = gpt
            summary = save_progress(args.output, rows, local_records, gpt_records)
            logging.info('%s GPT=%s cost=$%.6f; val accuracy so far=%s; false positives=%d/%d',
                         stem, gpt['answer'], gpt['cost_usd'], summary['accuracy'],
                         summary['false_positives'],
                         sum(r['truth'] == 'NORMAL' and r['prediction'] in ('NORMAL', 'TUMOR')
                             for r in aggregate(rows, local_records, gpt_records)[0]))
        else:
            logging.info('%d/50 %s reusing completed GPT response', index, stem)
        if gpt.get('state') != 'COMPLETE':
            raise RuntimeError(f'{stem}: GPT result state={gpt.get("state")}; stopped.')

    summary = save_progress(args.output, rows, local_records, gpt_records)
    if summary['completed_gpt_responses'] != 50:
        raise RuntimeError(f'Run ended with only {summary["completed_gpt_responses"]}/50 valid responses.')
    logging.info('Finished: accuracy=%.3f; false positives=%d/32 (%.3f); cost estimate=$%.4f',
                 summary['accuracy'], summary['false_positives'],
                 summary['false_positive_rate_on_normal_slides'], summary['api_cost_estimate_usd'])
    logging.info('Saved CSV/JSON results under %s', args.output)


if __name__ == '__main__':
    main()
