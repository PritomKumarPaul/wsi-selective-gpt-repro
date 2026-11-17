#!/usr/bin/env python3
"""Random-eight-patch CAMELYON16 baseline: single grid -> GPT, no CONCH/MLP.

For each official test slide, create the same single (0, 0)-offset grid used by
the prior test sampler (1024 px level-0 crops, stride 1024, complete crops
only). Keep all crops with >=50% tissue according to the existing HSV +
morphology thumbnail mask, uniformly sample eight without replacement, and
ask GPT to classify from those images. Repeat independently for each seed.

No segmentation masks, CONCH-Z model, MLP, ranking, or candidate cap are used.
Selected crops exist only in memory; coordinates, image hashes, responses,
usage, and predictions are written for reproducibility and safe resumption.
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
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openslide
from PIL import Image


DEFAULT_DATA = Path.home() / 'data' / 'camelyon16_full'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'results' / 'random8_gpt'
PATCH = 1024
THUMBNAIL_MAX = 2048
TISSUE_CUTOFF = 0.50
IMAGE_SIDE = 512
TOP_K = 8
DEFAULT_SEEDS = (1, 2, 3)
GPT_MODEL = 'gpt-5.4-mini-2026-03-17'
MAX_OUTPUT_TOKENS = 32
API_BUDGET_CAP_USD = 6.00
# A conservative preflight reserve based on the project's observed per-slide
# usage estimates. The guard checks before every new request; a single request
# can still exceed this reserve if provider pricing/usage differs.
API_CALL_RESERVE_USD = 0.01

# Kept consistent with the prior project's GPT cost estimates. Final cost is
# calculated from the usage returned by each API response, not from this rate
# table alone. Confirm account billing in the OpenAI Usage dashboard.
INPUT_RATE_USD_PER_MILLION = 0.75
CACHED_INPUT_RATE_USD_PER_MILLION = 0.075
OUTPUT_RATE_USD_PER_MILLION = 4.50

PROMPT = (
    'These are eight H&E image patches from a lymph-node slide. Based only on '
    'the visible tissue in these patches, is metastatic tumor visibly present '
    'in at least one patch? Return TUMOR if visible in any patch; otherwise '
    'return NORMAL. Do not infer tumor from normal lymph-node tissue. '
    'Return only the requested JSON.'
)


class SafetyStop(RuntimeError):
    """Stop before issuing further API requests after a safety condition."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tissue_mask(overview: np.ndarray) -> np.ndarray:
    """Exact HSV/morphology mask used by the project's existing WSI code."""
    hsv = cv2.cvtColor(overview, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 250)).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def candidates_from_slide(slide: openslide.OpenSlide) -> list[dict[str, Any]]:
    """Return every qualifying full crop from one origin-aligned grid."""
    width, height = slide.dimensions
    overview = np.asarray(
        slide.get_thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX)).convert('RGB'))
    mask = tissue_mask(overview)
    thumb_h, thumb_w = mask.shape
    candidates = []
    # Same single (0,0) grid and full-crop rule as the final test sampler.
    for y in range(0, height - PATCH + 1, PATCH):
        y1 = max(0, round(y * thumb_h / height))
        y2 = min(thumb_h, max(y1 + 1, round((y + PATCH) * thumb_h / height)))
        for x in range(0, width - PATCH + 1, PATCH):
            x1 = max(0, round(x * thumb_w / width))
            x2 = min(thumb_w, max(x1 + 1, round((x + PATCH) * thumb_w / width)))
            fraction = float(mask[y1:y2, x1:x2].mean())
            if fraction >= TISSUE_CUTOFF:
                candidates.append({'x': x, 'y': y, 'tissue_fraction': fraction})
    return candidates


def seeded_indices(seed: int, slide_id: str, count: int) -> np.ndarray:
    """Stable per-slide RNG, independent of processing order or resume point."""
    digest = hashlib.sha256(f'{seed}:{slide_id}'.encode('utf-8')).digest()
    derived_seed = int.from_bytes(digest[:8], 'little', signed=False)
    rng = np.random.default_rng(derived_seed)
    return rng.choice(count, size=TOP_K, replace=False)


def encode_patch(slide: openslide.OpenSlide, patch: dict[str, Any]) -> tuple[bytes, str]:
    rgb = np.asarray(
        slide.read_region((int(patch['x']), int(patch['y'])), 0,
                          (PATCH, PATCH)).convert('RGB'))
    resized = cv2.resize(rgb, (IMAGE_SIDE, IMAGE_SIDE), interpolation=cv2.INTER_AREA)
    buffer = io.BytesIO()
    Image.fromarray(resized).save(buffer, format='JPEG', quality=95)
    data = buffer.getvalue()
    return data, sha256_bytes(data)


def read_test_rows(data: Path) -> list[dict[str, str]]:
    reference = data / 'evaluation' / 'reference.csv'
    if not reference.is_file():
        raise FileNotFoundError(f'Missing split/reference file: {reference}')
    with reference.open(newline='') as f:
        rows = [dict(r) for r in csv.DictReader(f)
                if Path(r['image']).stem.startswith('test_')]
    rows.sort(key=lambda r: r['image'])
    counts = Counter(r['type'] for r in rows)
    if len(rows) != 129 or counts != {'normal': 80, 'tumor': 49}:
        raise ValueError(f'Expected the 129-slide test split (80 normal, 49 tumor); '
                         f'found {len(rows)} slides and labels {dict(counts)}.')
    missing = [str(data / 'images' / r['image']) for r in rows
               if not (data / 'images' / r['image']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} test slides are missing; first: {missing[0]}')
    return rows


def config_for(args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    return {
        'protocol': 'random-eight patch selection; three independent deterministic seeds',
        'split': 'CAMELYON16 public-mirror test split; labels used only for local scoring',
        'sampler': {
            'grid': 'one origin-aligned (0,0) grid; stride 1024 level-0 pixels',
            'crop': 'complete 1024x1024 level-0 crops only',
            'tissue_mask': 'thumbnail HSV (S>35, V<250), 3x3 opening, 5x5 closing',
            'thumbnail_max_dimensions': [THUMBNAIL_MAX, THUMBNAIL_MAX],
            'minimum_tissue_fraction': TISSUE_CUTOFF,
            'candidate_cap': None,
            'selection': 'uniform random without replacement from every eligible grid crop',
            'patches_per_slide': TOP_K,
            'seeds': seeds,
        },
        'local_models': 'none; no CONCH-Z or MLP',
        'masks': 'not read',
        'gpt': {
            'model': args.gpt_model, 'detail': 'high',
            'uploaded_image_dimensions': [IMAGE_SIDE, IMAGE_SIDE],
            'source_crop_dimensions_level0': [PATCH, PATCH],
            'reasoning_effort': 'none', 'max_output_tokens': MAX_OUTPUT_TOKENS,
            'prompt': PROMPT, 'store': False,
            'output_schema': {'class': ['NORMAL', 'TUMOR']},
        },
        'pricing_usd_per_1m_tokens': {
            'input': INPUT_RATE_USD_PER_MILLION,
            'cached_input': CACHED_INPUT_RATE_USD_PER_MILLION,
            'output': OUTPUT_RATE_USD_PER_MILLION,
        },
        'api_budget_guard': {
            'cap_usd': API_BUDGET_CAP_USD,
            'reserved_per_new_request_usd': API_CALL_RESERVE_USD,
            'note': 'Preflight reserve guard; one request may exceed its reserve.',
        },
    }


def compatible_config(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Allow raising only the output-token cap when resuming the same run."""
    old_copy = json.loads(json.dumps(old))
    new_copy = json.loads(json.dumps(new))
    old_gpt = old_copy.get('gpt', {})
    new_gpt = new_copy.get('gpt', {})
    old_cap = old_gpt.pop('max_output_tokens', None)
    new_cap = new_gpt.pop('max_output_tokens', None)
    return old_copy == new_copy and old_cap in (16, new_cap) and new_cap == MAX_OUTPUT_TOKENS


def parse_answer(raw: str) -> tuple[str, str]:
    try:
        value = json.loads(raw).get('class')
    except (json.JSONDecodeError, AttributeError, TypeError):
        value = None
    if value in ('NORMAL', 'TUMOR'):
        return value, 'valid_json'
    # Recover only a complete enum when the provider returns truncated JSON.
    match = re.match(r'^\s*\{\s*"class"\s*:\s*"(NORMAL|TUMOR)(?:"|\s*[},]|$)', raw or '')
    if match:
        return match.group(1), 'recovered_truncated_json_prefix'
    return 'INVALID', 'unparseable'


def response_usage(response: Any) -> tuple[dict[str, Any] | None, float | None]:
    usage = getattr(response, 'usage', None)
    if usage is None:
        return None, None
    raw = usage.model_dump()
    cached = int(getattr(getattr(usage, 'input_tokens_details', None),
                         'cached_tokens', 0) or 0)
    input_tokens = int(getattr(usage, 'input_tokens', 0) or 0)
    output_tokens = int(getattr(usage, 'output_tokens', 0) or 0)
    cost = ((input_tokens - cached) * INPUT_RATE_USD_PER_MILLION
            + cached * CACHED_INPUT_RATE_USD_PER_MILLION
            + output_tokens * OUTPUT_RATE_USD_PER_MILLION) / 1_000_000
    return raw, cost


def result_path(output: Path, seed: int, slide_id: str) -> Path:
    return output / 'responses' / f'seed_{seed}' / f'{slide_id}.json'


def budget_booked_usd(output: Path) -> float:
    """Sum known usage costs and reserve for requests with uncertain billing."""
    total = 0.0
    response_root = output / 'responses'
    if not response_root.is_dir():
        return total
    for path in response_root.glob('seed_*/*.json'):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            total += API_CALL_RESERVE_USD
            continue
        if record.get('cost_usd') is not None:
            total += max(0.0, float(record['cost_usd']))
        elif record.get('state') in {'IN_FLIGHT', 'ERROR_OR_UNKNOWN_BILLING'}:
            total += API_CALL_RESERVE_USD
    return total


def make_request_content(slide: openslide.OpenSlide,
                         selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    content: list[dict[str, Any]] = [{'type': 'input_text', 'text': PROMPT}]
    hashes = []
    for patch in selected:
        image_bytes, digest = encode_patch(slide, patch)
        hashes.append(digest)
        payload = base64.b64encode(image_bytes).decode('ascii')
        content.append({'type': 'input_image', 'detail': 'high',
                        'image_url': f'data:image/jpeg;base64,{payload}'})
    return content, hashes


def call_gpt(client: Any, output: Path, seed: int, slide_id: str,
             image_hashes: list[str], model: str,
             content: list[dict[str, Any]],
             retry_invalid_output: bool) -> dict[str, Any]:
    path = result_path(output, seed, slide_id)
    prompt_hash = sha256_bytes(PROMPT.encode('utf-8'))
    fingerprint = {'seed': seed, 'slide': slide_id, 'model': model,
                   'prompt_sha256': prompt_hash, 'image_sha256': image_hashes}
    prior_attempt_cost = 0.0
    if path.exists():
        saved = json.loads(path.read_text())
        for key, expected in fingerprint.items():
            if saved.get(key) != expected:
                raise ValueError(f'Existing response cache has different {key}: {path}')
        if saved.get('state') == 'COMPLETE':
            return saved
        if (saved.get('state') == 'INVALID_OUTPUT' and retry_invalid_output
                and saved.get('incomplete_reason') == 'max_output_tokens'):
            prior_attempt_cost = float(saved.get('cost_usd') or 0.0)
            archive = path.with_name(
                f'{path.stem}.invalid_max_output_tokens.{time.time_ns()}.json')
            os.replace(path, archive)
            logging.warning('Archived truncated response before explicit retry: %s', archive)
        else:
        # An interrupted/in-flight or failed request may already have been billed.
        # Never repeat it automatically; inspect account usage and decide manually.
            raise SafetyStop(
                f'Cached request is {saved.get("state")}; stopped before any further API call. '
                f'Inspect the response record and account usage: {path}')

    booked = budget_booked_usd(output)
    if booked + API_CALL_RESERVE_USD > API_BUDGET_CAP_USD:
        raise SafetyStop(
            f'Budget guard stopped before a new request: ${booked:.4f} booked/reserved, '
            f'${API_CALL_RESERVE_USD:.2f} reserve needed, cap ${API_BUDGET_CAP_USD:.2f}.')

    saved = {**fingerprint, 'state': 'IN_FLIGHT',
             'prior_attempt_cost_usd': prior_attempt_cost,
             'note': 'If interrupted or failed, inspect API usage before any manual retry.'}
    atomic_json(path, saved)
    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=model,
            input=[{'role': 'user', 'content': content}],
            store=False,
            reasoning={'effort': 'none'},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            text={'format': {
                'type': 'json_schema', 'name': 'camelyon_binary_class', 'strict': True,
                'schema': {'type': 'object', 'properties': {
                    'class': {'type': 'string', 'enum': ['NORMAL', 'TUMOR']}},
                    'required': ['class'], 'additionalProperties': False},
            }},
        )
    except Exception as exc:
        saved.update(state='ERROR_OR_UNKNOWN_BILLING', error_type=type(exc).__name__,
                     status_code=getattr(exc, 'status_code', None),
                     elapsed_seconds=time.perf_counter() - started)
        atomic_json(path, saved)
        return saved

    elapsed = time.perf_counter() - started
    raw = getattr(response, 'output_text', '') or ''
    answer, parse_status = parse_answer(raw)
    usage, cost = response_usage(response)
    response_status = getattr(response, 'status', None)
    incomplete = getattr(response, 'incomplete_details', None)
    incomplete_reason = getattr(incomplete, 'reason', None) if incomplete else None
    clean = (usage is not None and parse_status == 'valid_json'
             and incomplete_reason is None and response_status in (None, 'completed'))
    saved.update(state='COMPLETE' if clean else 'INVALID_OUTPUT', answer=answer,
                 raw_answer=raw, response_id=getattr(response, 'id', None),
                 response_model=getattr(response, 'model', None), usage=usage,
                 cost_usd=cost, gpt_seconds=elapsed, parse_status=parse_status,
                 response_status=response_status, incomplete_reason=incomplete_reason)
    atomic_json(path, saved)
    return saved


def score_seed(seed: int, rows: list[dict[str, str]], data: Path,
               output: Path, client: Any, model: str,
               retry_invalid_output: bool) -> list[dict[str, Any]]:
    seed_results = []
    for index, row in enumerate(rows, 1):
        slide_id = Path(row['image']).stem
        truth = 'TUMOR' if row['type'] == 'tumor' else 'NORMAL'
        image_path = data / 'images' / row['image']
        slide_started = time.perf_counter()
        try:
            with openslide.OpenSlide(str(image_path)) as slide:
                width, height = slide.dimensions
                candidates = candidates_from_slide(slide)
                if len(candidates) < TOP_K:
                    result = {
                        'state': 'INSUFFICIENT_CANDIDATES', 'slide': slide_id,
                        'seed': seed, 'truth': truth, 'prediction': None,
                        'candidate_count': len(candidates), 'required_candidates': TOP_K,
                        'selection_seconds': time.perf_counter() - slide_started,
                        'total_seconds': time.perf_counter() - slide_started,
                    }
                else:
                    indices = seeded_indices(seed, slide_id, len(candidates))
                    selected = [candidates[int(i)] for i in indices]
                    coords = [{'rank': rank, 'x_level0': int(p['x']),
                               'y_level0': int(p['y']),
                               'tissue_fraction': float(p['tissue_fraction'])}
                              for rank, p in enumerate(selected, 1)]
                    content, image_hashes = make_request_content(slide, selected)
                    selection_seconds = time.perf_counter() - slide_started
                    selection = {
                        'slide': slide_id, 'seed': seed,
                        'candidate_count_before_random_selection': len(candidates),
                        'candidate_cap': None, 'patches_selected': TOP_K,
                        'source_image': row['image'],
                        'slide_dimensions_level0': [width, height],
                        'patches': coords, 'image_sha256': image_hashes,
                        'truth_not_sent_to_gpt': truth,
                    }
                    atomic_json(output / 'selections' / f'seed_{seed}' / f'{slide_id}.json',
                                selection)
                    gpt = call_gpt(client, output, seed, slide_id, image_hashes, model,
                                   content, retry_invalid_output)
                    pred = gpt.get('answer') if gpt.get('state') == 'COMPLETE' else None
                    result = {
                        'state': gpt.get('state'), 'slide': slide_id, 'seed': seed,
                        'truth': truth, 'prediction': pred,
                        'correct': pred == truth if pred in ('TUMOR', 'NORMAL') else None,
                        'candidate_count': len(candidates), 'patches_selected': TOP_K,
                        'selected_coordinates': coords,
                        'image_sha256': image_hashes,
                        'selection_seconds': selection_seconds,
                        'total_seconds': time.perf_counter() - slide_started,
                        'gpt_seconds': gpt.get('gpt_seconds'),
                        'input_tokens': (gpt.get('usage') or {}).get('input_tokens'),
                        'cached_input_tokens': ((gpt.get('usage') or {}).get(
                            'input_tokens_details') or {}).get('cached_tokens'),
                        'output_tokens': (gpt.get('usage') or {}).get('output_tokens'),
                        'cost_usd': gpt.get('cost_usd'),
                        'prior_attempt_cost_usd': gpt.get('prior_attempt_cost_usd', 0.0),
                        'parse_status': gpt.get('parse_status'),
                        'response_id': gpt.get('response_id'),
                    }
        except Exception as exc:
            if isinstance(exc, SafetyStop):
                raise
            result = {'state': 'LOCAL_ERROR', 'slide': slide_id, 'seed': seed,
                      'truth': truth, 'prediction': None,
                      'error_type': type(exc).__name__, 'error': str(exc),
                      'selection_seconds': time.perf_counter() - slide_started,
                      'total_seconds': time.perf_counter() - slide_started}

        seed_results.append(result)
        atomic_csv(output / f'seed_{seed}_per_slide.csv', seed_results,
                   ['state', 'slide', 'seed', 'truth', 'prediction', 'correct',
                    'candidate_count', 'patches_selected', 'selection_seconds',
                    'total_seconds', 'gpt_seconds', 'input_tokens', 'cached_input_tokens',
                    'output_tokens', 'cost_usd', 'prior_attempt_cost_usd',
                    'parse_status', 'response_id',
                    'selected_coordinates', 'image_sha256', 'error_type', 'error'])
        atomic_json(output / f'seed_{seed}_summary.json', summarize(seed, seed_results))
        logging.info('seed=%s progress=%s/%s slide=%s candidates=%s state=%s prediction=%s',
                     seed, index, len(rows), slide_id, result.get('candidate_count'),
                     result['state'], result.get('prediction'))
        if result.get('state') in {'INVALID_OUTPUT', 'ERROR_OR_UNKNOWN_BILLING'}:
            raise SafetyStop(
                f'{result["state"]} for {slide_id} (seed {seed}); stopped before the next GPT call. '
                'Inspect its response JSON and API usage before deciding whether to resume.')
        booked = budget_booked_usd(output)
        if booked >= API_BUDGET_CAP_USD:
            raise SafetyStop(
                f'Budget guard reached after {slide_id}: ${booked:.4f} booked/reserved '
                f'against the ${API_BUDGET_CAP_USD:.2f} cap; no further calls will be made.')
    return seed_results


def summarize(seed: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in rows if r.get('prediction') in ('NORMAL', 'TUMOR')]
    truth = [r['truth'] for r in completed]
    pred = [r['prediction'] for r in completed]
    tn = sum(t == p == 'NORMAL' for t, p in zip(truth, pred))
    fp = sum(t == 'NORMAL' and p == 'TUMOR' for t, p in zip(truth, pred))
    fn = sum(t == 'TUMOR' and p == 'NORMAL' for t, p in zip(truth, pred))
    tp = sum(t == p == 'TUMOR' for t, p in zip(truth, pred))
    return {
        'seed': seed, 'planned_slides': 129, 'slides_attempted': len(rows),
        'completed_predictions': len(completed),
        'failed_or_unscored': len(rows) - len(completed),
        'accuracy_over_completed': sum(r.get('correct') is True for r in completed) / len(completed)
            if completed else None,
        'sensitivity_over_completed': tp / (tp + fn) if tp + fn else None,
        'specificity_over_completed': tn / (tn + fp) if tn + fp else None,
        'confusion_matrix_truth_rows_prediction_columns_NORMAL_TUMOR': [[tn, fp], [fn, tp]],
        'gpt_calls_attempted': sum(r.get('state') in {
            'COMPLETE', 'INVALID_OUTPUT', 'ERROR_OR_UNKNOWN_BILLING'} for r in rows),
        'gpt_valid_answers': sum(r.get('state') == 'COMPLETE' for r in rows),
        'candidate_pool_min': min((r['candidate_count'] for r in rows
                                   if 'candidate_count' in r), default=None),
        'candidate_pool_mean': float(np.mean([r['candidate_count'] for r in rows
                                               if 'candidate_count' in r]))
            if any('candidate_count' in r for r in rows) else None,
        'gpt_cost_usage_estimate_usd': sum(
            float(r.get('cost_usd') or 0) + float(r.get('prior_attempt_cost_usd') or 0)
            for r in rows),
        'gpt_seconds_total': sum(float(r.get('gpt_seconds') or 0) for r in rows),
        'selection_and_crop_seconds_total': sum(float(r.get('selection_seconds') or 0)
                                                for r in rows),
        'state_counts': dict(Counter(r.get('state', 'UNKNOWN') for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA,
                        help='CAMELYON16 root containing images/ and evaluation/reference.csv')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--gpt-model', default=GPT_MODEL)
    parser.add_argument('--retry-invalid-output', action='store_true',
                        help='Explicitly retry cached responses stopped for max_output_tokens. '
                             'The earlier attempt may already have been billed.')
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error('Provide one or more distinct random seeds.')
    if not args.output.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error('Output must remain inside this organized project folder.')
    rows = read_test_rows(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    config = config_for(args, args.seeds)
    config_path = args.output / 'run_config.json'
    if config_path.exists():
        saved_config = json.loads(config_path.read_text())
        if saved_config != config and compatible_config(saved_config, config):
            atomic_json(config_path, config)
        elif saved_config != config:
            raise ValueError(f'Output folder belongs to another protocol: {args.output}')
    else:
        atomic_json(config_path, config)
    log_path = args.output / 'run.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])

    # Ask once for the whole run; the key is held in memory and never written.
    print(f'Protocol: one grid, all >=50%-tissue candidates, random eight, no cap; '
          f'{len(rows)} slides x {len(args.seeds)} seeds.')
    print('The label file is used only for local scoring; it is not sent to GPT.')
    key = getpass.getpass('OpenAI API key (hidden; not saved): ')
    if not key:
        raise ValueError('Empty API key.')
    from openai import OpenAI
    client = OpenAI(api_key=key, timeout=180, max_retries=0)
    del key

    all_summaries = []
    stopped_reason = None
    for seed in args.seeds:
        logging.info('Starting seed %d', seed)
        try:
            results = score_seed(seed, rows, args.data, args.output, client, args.gpt_model,
                                 args.retry_invalid_output)
        except SafetyStop as exc:
            stopped_reason = str(exc)
            logging.error('%s', stopped_reason)
            partial_path = args.output / f'seed_{seed}_summary.json'
            if partial_path.is_file():
                all_summaries.append(json.loads(partial_path.read_text()))
            break
        summary = summarize(seed, results)
        atomic_json(args.output / f'seed_{seed}_summary.json', summary)
        all_summaries.append(summary)
        logging.info('Finished seed %d: accuracy=%s; GPT cost estimate=$%.6f',
                     seed, summary['accuracy_over_completed'],
                     summary['gpt_cost_usage_estimate_usd'])
    atomic_json(args.output / 'all_seeds_summary.json', {
        'run_config': config,
        'run_status': 'STOPPED_SAFETY' if stopped_reason else 'COMPLETE',
        'stopped_reason': stopped_reason,
        'seed_summaries': all_summaries,
        'mean_accuracy_over_seed_results': float(np.mean([
            s['accuracy_over_completed'] for s in all_summaries
            if s['accuracy_over_completed'] is not None])) if any(
                s['accuracy_over_completed'] is not None for s in all_summaries) else None,
        'total_gpt_cost_usage_estimate_usd': sum(
            s['gpt_cost_usage_estimate_usd'] for s in all_summaries),
        'note': 'Three seeds are repeated measurements on the same 129 slides, not 387 independent slides. '
                'The official test set has prior exploratory use in this project.',
    })
    if stopped_reason:
        logging.info('Run stopped safely; completed summaries and response records were saved.')
    else:
        logging.info('All seeds finished.')


if __name__ == '__main__':
    main()
