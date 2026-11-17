#!/usr/bin/env python3
"""Final CAMELYON16 test-set run: sampler -> CONCH-Z -> MLP gate -> GPT top-8.

The validation-calibrated no-call rule is loaded from gate.json. The local
sampler uses one 1024-pixel grid, >=50% thumbnail tissue, and a maximum of
500 candidates. CONCH-Z visual Linear layers use CUDA W8A8; the saved MLP
remains FP32. Masks are never read; reference labels are used only after a
prediction is recorded. Results are resumable per slide.
"""
import argparse
import base64
import csv
import getpass
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import openslide
import torch
from PIL import Image
from transformers import AutoModel

from cuda_int8_linear import check_cuda_int8, convert_cuda_linears
from patch_fraction_regression import atomic_json
from test_patch_regression_heads import SavedScaler
from train_patch_neural_head import make_head
from train_retrieval_study import sha256
from wsi_core import tissue_mask


DEFAULT_TRAINING = Path.home() / 'wsi_patch_fraction_cache_v1'
DEFAULT_DATA = Path.home() / 'data' / 'camelyon16_full'
DEFAULT_OUTPUT = Path.home() / 'wsi_test_quantized_runtime_gpt_v1'
PATCH = 1024
CAP = 500
TOP_K = 8
TISSUE_CUTOFF = 0.50
QUANTIZATION = 'CUDA W8A8 torch._int_mm on encoder.visual nn.Linear only; all remaining CONCH ops FP32'
GPT_MODEL = 'gpt-5.4-mini-2026-03-17'
INPUT_RATE = 0.75
CACHED_INPUT_RATE = 0.075
OUTPUT_RATE = 4.50
IMAGE_SIDE = 512
MAX_OUTPUT_TOKENS = 32

PROMPT = (
    'Classify the lymph-node slide from these eight H&E crops. Return TUMOR '
    'if metastatic carcinoma is visibly present in any crop; otherwise return '
    'NORMAL based on the provided crops. Do not infer tumor from ordinary '
    'lymph-node tissue. These crops are selected samples, not the complete '
    'slide. Return only JSON with one field: class, whose value is NORMAL or TUMOR.'
)


def atomic_csv(path, rows, fields):
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
    gate_file = args.gate_file
    gate = json.loads(gate_file.read_text())
    recalibration_dir = gate_file.parent
    if gate.get('checkpoint_sha256') != sha256(checkpoint_path):
        raise ValueError('Quantized validation gate was calibrated for a different MLP checkpoint.')
    if gate.get('encoder_config_sha256') != sha256(
            Path(manifest['config']['model_dir']) / 'config.json'):
        raise ValueError('Quantized validation gate was calibrated for a different CONCH-Z config.')
    if gate.get('quantization') != QUANTIZATION:
        raise ValueError('Gate calibration does not use the required quantized CONCH-Z method.')
    if gate.get('mlp') != 'FP32, original saved current MLP':
        raise ValueError('Gate calibration did not use the required FP32 MLP.')
    if gate.get('sampler') != {
            'patch': PATCH,
            'grid': 'one (0,0) grid, stride 1024; complete crops only',
            'tissue_cutoff': TISSUE_CUTOFF,
            'candidate_cap': CAP,
            'cap_rule': 'farthest-point spatial spread seeded by highest tissue fraction'}:
        raise ValueError('Quantized gate calibration sampler does not match this test sampler.')
    gate_config = recalibration_dir / 'config.json'
    gate_scores = recalibration_dir / 'scores.csv'
    if (gate.get('validation_config_sha256') != sha256(gate_config)
            or gate.get('validation_scores_sha256') != sha256(gate_scores)):
        raise ValueError('Validation recalibration artifacts do not match the saved gate.')
    threshold = float(gate['threshold'])
    if not np.isfinite(threshold):
        raise ValueError('Calibrated threshold is not finite.')
    run_config_path = checkpoint_path.parent / 'run_config.json'
    checkpoint_config = json.loads(run_config_path.read_text())
    if checkpoint_config.get('head') != 'mlp' or checkpoint_config.get('cache_manifest') != manifest:
        raise ValueError('The saved current MLP does not match its source training manifest.')

    data = args.data or Path(manifest['config']['data'])
    encoder_path = Path(manifest['config']['model_dir'])
    reference = data / 'evaluation' / 'reference.csv'
    if not reference.is_file() or sha256(reference) != manifest['config']['reference_sha256']:
        raise ValueError(f'Dataset/reference mismatch or missing file: {reference}')
    if sha256(encoder_path / 'config.json') != manifest['config']['model_config_sha256']:
        raise ValueError('CONCH-Z config differs from the one used to train the MLP.')

    with reference.open(newline='') as f:
        rows = [dict(r) for r in csv.DictReader(f)
                if Path(r['image']).stem.startswith('test_')]
    rows.sort(key=lambda r: r['image'])
    if len(rows) != 129 or len({r['image'] for r in rows}) != 129:
        raise ValueError(f'Expected 129 distinct public-mirror test WSIs; found {len(rows)}.')
    counts = Counter(r['type'] for r in rows)
    if counts != {'normal': 80, 'tumor': 49}:
        raise ValueError(f'Unexpected test label counts: {dict(counts)}')
    for row in rows:
        image_path = data / 'images' / row['image']
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    config = {
        'protocol': 'fixed final test-set binary run; previous test retrieval diagnostics exist',
        'split': 'CAMELYON16 public-mirror official test; 129 slides',
        'training_manifest_sha256': sha256(manifest_path),
        'current_mlp_sha256': sha256(checkpoint_path),
        'conch_config_sha256': sha256(encoder_path / 'config.json'),
        'sampler': {
            'crop_pixels_level0': PATCH,
            'grid': 'single (0,0)-offset grid; stride 1024; complete crops only',
            'tissue': 'HSV/morphology tissue_mask on <=2048px thumbnail; crop fraction >=50%',
            'candidate_cap': CAP,
            'cap_method': 'farthest-point spatial spread; seed highest tissue fraction',
        },
        'local_models': 'frozen CONCH-Z with visual Linear W8A8 CUDA quantization + saved current MLP FP32; no training',
        'quantization': QUANTIZATION,
        'gate': {'slide_score': 'maximum raw MLP score among sampled candidates',
                 'threshold': threshold,
                 'rule': 'skip GPT and output NORMAL if max score <= calibrated threshold',
                 'validation_gate_file_sha256': sha256(gate_file)},
        'gpt': {'model': GPT_MODEL, 'top_k': TOP_K, 'image_pixels': [IMAGE_SIDE, IMAGE_SIDE],
                'image_detail': 'high', 'max_output_tokens': MAX_OUTPUT_TOKENS, 'reasoning_effort': 'none',
                'prompt': PROMPT, 'store': False},
        'labels_and_masks': 'reference label used only for final scoring; tumor masks are not read',
        'pricing_usd_per_1m_tokens': {'input': INPUT_RATE,
                                      'cached_input': CACHED_INPUT_RATE,
                                      'output': OUTPUT_RATE},
        'script_sha256': sha256(Path(__file__)),
    }
    return manifest, rows, data, encoder_path, checkpoint_path, config, threshold


def sample_candidates(slide):
    """One grid, same thumbnail tissue-mask threshold as the validation gate."""
    width, height = slide.dimensions
    thumb = np.asarray(slide.get_thumbnail((2048, 2048)).convert('RGB'))
    mask = tissue_mask(thumb)
    th, tw = mask.shape
    candidates = []
    for y in range(0, height - PATCH + 1, PATCH):
        y1 = max(0, round(y * th / height))
        y2 = min(th, max(y1 + 1, round((y + PATCH) * th / height)))
        for x in range(0, width - PATCH + 1, PATCH):
            x1 = max(0, round(x * tw / width))
            x2 = min(tw, max(x1 + 1, round((x + PATCH) * tw / width)))
            frac = float(mask[y1:y2, x1:x2].mean())
            if frac >= TISSUE_CUTOFF:
                candidates.append({'x': x, 'y': y, 'tissue_fraction': frac})
    return candidates


def spatial_cap(candidates, width, height):
    if len(candidates) <= CAP:
        return candidates
    xy = np.asarray([[(p['x'] + PATCH / 2) / width,
                      (p['y'] + PATCH / 2) / height] for p in candidates], dtype=np.float64)
    tissue = np.asarray([p['tissue_fraction'] for p in candidates], dtype=np.float64)
    used = np.zeros(len(candidates), dtype=bool)
    distance = np.full(len(candidates), np.inf)
    current = int(np.argmax(tissue))
    selected = []
    for _ in range(CAP):
        selected.append(candidates[current])
        used[current] = True
        distance = np.minimum(distance, ((xy - xy[current]) ** 2).sum(axis=1))
        distance[used] = -1
        current = int(np.argmax(distance + 1e-12 * tissue))
    return selected


def load_models(encoder_path, checkpoint_path, device):
    if not str(device).startswith('cuda'):
        raise RuntimeError('Quantized CONCH-Z requires an available CUDA device.')
    check_cuda_int8()
    encoder = AutoModel.from_pretrained(str(encoder_path), trust_remote_code=True,
                                        local_files_only=True).eval().float()
    converted = convert_cuda_linears(encoder.visual)
    if converted < 1:
        raise RuntimeError('No CONCH-Z visual Linear layers were converted to W8A8.')
    encoder.to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('head') != 'mlp':
        raise ValueError('Expected the saved MLP checkpoint.')
    head = make_head('mlp', checkpoint['hidden'], checkpoint['dropout'])
    head.load_state_dict(checkpoint['state_dict'], strict=True)
    head.eval().float().to(device)
    if any(p.dtype != torch.float32 for p in head.parameters()):
        raise ValueError('The saved MLP must run in FP32.')
    scaler = SavedScaler(checkpoint['scaler_mean'].numpy(),
                         checkpoint['scaler_scale'].numpy())
    return encoder, head, scaler, converted


def run_local(row, data, output, encoder, head, scaler, device, batch_size, config,
              gate_threshold):
    stem = Path(row['image']).stem
    started = time.perf_counter()
    timer = {'sampling_seconds': 0.0, 'decode_transform_seconds': 0.0,
             'conch_forward_seconds': 0.0, 'mlp_seconds': 0.0}
    with openslide.OpenSlide(str(data / 'images' / row['image'])) as slide:
        dimensions = list(slide.dimensions)
        t = time.perf_counter()
        pool = sample_candidates(slide)
        before_cap = len(pool)
        pool = spatial_cap(pool, *slide.dimensions)
        timer['sampling_seconds'] = time.perf_counter() - t
        if len(pool) < TOP_K:
            raise ValueError(f'{stem}: only {len(pool)} tissue candidates, need {TOP_K}.')

        vectors = []
        for start in range(0, len(pool), batch_size):
            batch = pool[start:start + batch_size]
            t = time.perf_counter()
            images = torch.stack([
                encoder.eval_transform(slide.read_region((p['x'], p['y']), 0,
                                                         (PATCH, PATCH)).convert('RGB'))
                for p in batch]).to(device)
            timer['decode_transform_seconds'] += time.perf_counter() - t
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            t = time.perf_counter()
            with torch.inference_mode():
                vectors.append(encoder.encode_image(images).float().cpu().numpy())
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            timer['conch_forward_seconds'] += time.perf_counter() - t
            print(f'  {stem}: encoded {min(start + len(batch), len(pool))}/{len(pool)}', flush=True)
        features = np.concatenate(vectors).astype(np.float32)
        standardized = scaler.transform(features)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        with torch.inference_mode():
            scores = head(torch.from_numpy(standardized).to(device)).flatten().float().cpu().numpy()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        timer['mlp_seconds'] = time.perf_counter() - t
        if not np.isfinite(scores).all():
            raise RuntimeError(f'{stem}: MLP produced non-finite scores.')
        best = int(np.argmax(scores))
        order = np.argsort(-scores, kind='stable')[:TOP_K]

        mpp_x = mpp_y = None
        try:
            mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        except (KeyError, ValueError):
            pass
        top8 = []
        for rank, idx in enumerate(order, 1):
            p = pool[int(idx)]
            top8.append({'rank': rank, 'x_level0': int(p['x']), 'y_level0': int(p['y']),
                         'mlp_score': float(scores[int(idx)]),
                         'tissue_fraction': float(p['tissue_fraction'])})
    score = float(scores[best])
    local_seconds = time.perf_counter() - started
    record = {
        'config': config, 'slide': stem,
        'evaluation_truth_not_sent_to_gpt': 'TUMOR' if row['type'] == 'tumor' else 'NORMAL',
        'reference_subtype_not_sent_to_gpt': row['class'],
        'slide_dimensions_level0': dimensions, 'mpp': [mpp_x, mpp_y],
        'candidate_pool_before_cap': before_cap, 'candidates_encoded': len(pool),
        'max_raw_mlp_score': score,
        'gate_threshold': gate_threshold,
        'gate_decision': 'SKIP_GPT' if score <= gate_threshold else 'CALL_GPT',
        'top8': top8, 'source_image': row['image'],
        'local_seconds': local_seconds, 'local_timing': timer,
        'mask_read': False,
    }
    atomic_json(output / 'local' / f'{stem}.json', record)
    return record


def load_local(path, config):
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get('config') != config:
        if not compatible_resume_config(record.get('config', {}), config):
            raise ValueError(f'Existing local result uses different settings: {path}')
        record['config'] = config
        atomic_json(path, record)
    if len(record.get('top8', [])) != TOP_K:
        raise ValueError(f'Incomplete local record: {path}')
    return record


def gpt_content(record, data):
    content = [{'type': 'input_text', 'text': PROMPT}]
    image_hashes = []
    mpp = record['mpp']
    width, height = record['slide_dimensions_level0']
    source_image = data / 'images' / record['source_image']
    with openslide.OpenSlide(str(source_image)) as slide:
        if list(slide.dimensions) != [width, height]:
            raise ValueError(f'{record["slide"]}: source WSI dimensions changed since local inference.')
        for patch in record['top8']:
            x, y = int(patch['x_level0']), int(patch['y_level0'])
            if x < 0 or y < 0 or x + PATCH > width or y + PATCH > height:
                raise ValueError(f'{record["slide"]}: selected crop coordinates fall outside the WSI.')
            rgb = np.asarray(slide.read_region((x, y), 0, (PATCH, PATCH)).convert('RGB'))
            resized = cv2.resize(rgb, (IMAGE_SIDE, IMAGE_SIDE), interpolation=cv2.INTER_AREA)
            buffer = io.BytesIO()
            Image.fromarray(resized).save(buffer, format='JPEG', quality=95)
            image_bytes = buffer.getvalue()
            image_hashes.append(hashlib.sha256(image_bytes).hexdigest())
            image_b64 = base64.b64encode(image_bytes).decode('ascii')
            if mpp[0] is None or mpp[1] is None:
                context = (f'Field {patch["rank"]}; source crop 1024x1024 level-0 pixels; '
                           'physical pixel spacing unavailable.')
            else:
                sx, sy = 1024 * mpp[0] / 1000, 1024 * mpp[1] / 1000
                ox, oy = patch['x_level0'] * mpp[0] / 1000, patch['y_level0'] * mpp[1] / 1000
                sw, sh = width * mpp[0] / 1e6, height * mpp[1] / 1e6
                context = (f'Field {patch["rank"]}; source crop 1024x1024 level-0 pixels '
                           f'({sx:.3f} x {sy:.3f} mm); crop origin ({ox:.2f}, {oy:.2f}) mm '
                           f'from the slide top-left; whole-slide size ({sw:.1f} x {sh:.1f}) mm; '
                           f'uploaded resized image is {IMAGE_SIDE}x{IMAGE_SIDE}.')
            content.extend([{'type': 'input_text', 'text': context},
                            {'type': 'input_image', 'detail': 'high',
                             'image_url': 'data:image/jpeg;base64,' + image_b64}])
    return content, image_hashes


def usage_and_cost(response):
    usage = getattr(response, 'usage', None)
    if usage is None:
        return None, None
    raw = usage.model_dump()
    cached = int(getattr(getattr(usage, 'input_tokens_details', None), 'cached_tokens', 0) or 0)
    input_tokens = int(usage.input_tokens or 0)
    output_tokens = int(usage.output_tokens or 0)
    cost = ((input_tokens - cached) * INPUT_RATE + cached * CACHED_INPUT_RATE
            + output_tokens * OUTPUT_RATE) / 1_000_000
    return raw, cost


def parse_class_answer(raw):
    """Parse valid JSON, or recover a complete enum from a truncated JSON prefix."""
    try:
        value = json.loads(raw or '').get('class')
    except (json.JSONDecodeError, AttributeError):
        value = None
    if value in ('NORMAL', 'TUMOR'):
        return value, 'valid_json'
    # Only recover when the full allowed enum value is present unambiguously.
    match = re.match(r'^\s*\{\s*"class"\s*:\s*"(NORMAL|TUMOR)(?:"|\s*[},]|$)', raw or '')
    if match:
        return match.group(1), 'recovered_truncated_json_prefix'
    return 'INVALID', 'unparseable'


def compatible_resume_config(saved, current):
    """Allow prompt/output-cap edits while preserving already-computed local crops."""
    def normalized(config):
        value = json.loads(json.dumps(config))
        value.pop('script_sha256', None)
        gpt = value.get('gpt', {})
        if gpt.get('max_output_tokens') not in (16, MAX_OUTPUT_TOKENS):
            return None
        gpt['max_output_tokens'] = 16
        gpt.pop('prompt', None)
        return value
    old, new = normalized(saved), normalized(current)
    return old is not None and old == new


def request_gpt(client, local, output, model, data):
    path = output / 'gpt' / f'{local["slide"]}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()
    crop_prepare_started = time.perf_counter()
    content, image_hashes = gpt_content(local, data)
    crop_prepare_seconds = time.perf_counter() - crop_prepare_started
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('image_sha256') != image_hashes or saved.get('model') != model:
            raise ValueError(f'Existing GPT record does not match current inputs: {path}')
        if saved.get('prompt_sha256') != prompt_hash:
            if saved.get('state') != 'COMPLETE':
                raise RuntimeError(f'{local["slide"]}: an earlier request with a different prompt '
                                   f'has state {saved.get("state")}; inspect it before retrying.')
            archive = output / 'gpt' / 'archive' / (
                f'{local["slide"]}.{saved.get("prompt_sha256", "legacy")[:12]}.json')
            if archive.exists():
                raise FileExistsError(f'Archived response already exists: {archive}')
            archive.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, archive)
            print(f'  Archived prior-prompt response: {archive.name}', flush=True)
            saved = None
        if saved is not None and saved.get('state') == 'COMPLETE':
            if saved.get('answer') == 'INVALID':
                answer, parse_status = parse_class_answer(saved.get('raw_answer', ''))
                if answer in ('NORMAL', 'TUMOR'):
                    saved.update(answer=answer, parse_status=parse_status)
                    atomic_json(path, saved)
            return saved
        if saved is not None:
            raise RuntimeError(f'{local["slide"]}: prior request is {saved.get("state")}; inspect '
                               'the response record before manually deciding whether to retry.')

    saved = {'slide': local['slide'], 'model': model,
             'image_sha256': image_hashes, 'prompt_sha256': prompt_hash,
             'runtime_crop_prepare_seconds': crop_prepare_seconds,
             'state': 'IN_FLIGHT'}
    atomic_json(path, saved)
    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=model,
            input=[{'role': 'user', 'content': content}],
            store=False, reasoning={'effort': 'none'}, max_output_tokens=MAX_OUTPUT_TOKENS,
            text={'format': {'type': 'json_schema', 'name': 'binary_camelyon_answer',
                'strict': True, 'schema': {'type': 'object', 'properties': {
                    'class': {'type': 'string', 'enum': ['NORMAL', 'TUMOR']}},
                    'required': ['class'], 'additionalProperties': False}}},
        )
    except Exception as exc:
        saved.update(state='ERROR_OR_UNKNOWN_BILLING', error_type=type(exc).__name__,
                     status_code=getattr(exc, 'status_code', None),
                     elapsed_seconds=time.perf_counter() - started)
        atomic_json(path, saved)
        raise RuntimeError('Request failed. Saved status without automatic retry to avoid '
                           'accidental duplicate charges.') from None
    elapsed = time.perf_counter() - started
    usage, cost = usage_and_cost(response)
    if usage is None:
        saved.update(state='MISSING_USAGE', response_id=response.id,
                     raw_answer=response.output_text, elapsed_seconds=elapsed)
        atomic_json(path, saved)
        raise RuntimeError('Response usage missing; stopped rather than guessing cost.')
    answer, parse_status = parse_class_answer(response.output_text or '')
    response_status = getattr(response, 'status', None)
    incomplete = getattr(response, 'incomplete_details', None)
    incomplete_reason = getattr(incomplete, 'reason', None) if incomplete else None
    clean_completion = (parse_status == 'valid_json' and incomplete_reason is None
                       and response_status in (None, 'completed'))
    saved.update(state='COMPLETE', answer=answer, raw_answer=response.output_text,
                 response_id=response.id, response_model=response.model,
                 usage=usage, cost_usd=cost, gpt_seconds=elapsed,
                 runtime_crop_prepare_seconds=crop_prepare_seconds,
                 parse_status=parse_status, response_status=response_status,
                 incomplete_reason=incomplete_reason)
    if not clean_completion:
        saved['state'] = 'INVALID_OUTPUT'
    atomic_json(path, saved)
    if not clean_completion:
        raise RuntimeError(f'{local["slide"]}: GPT returned incomplete/unparseable output '
                           f'(status={response_status}, parse={parse_status}, '
                           f'reason={incomplete_reason}). Saved response and stopping before '
                           'the next slide; inspect the record before resuming.')
    return saved


def has_current_gpt_result(path, model):
    if not path.is_file():
        return False
    saved = json.loads(path.read_text())
    expected_prompt = hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()
    return (saved.get('model') == model
            and saved.get('prompt_sha256') == expected_prompt
            and saved.get('state') == 'COMPLETE')


def make_slide_result(local, gpt_record=None):
    skipped = local['gate_decision'] == 'SKIP_GPT'
    truth = local['evaluation_truth_not_sent_to_gpt']
    prediction = 'NORMAL' if skipped else (gpt_record or {}).get('answer')
    gpt_seconds = 0.0 if skipped else (gpt_record or {}).get('gpt_seconds', 0.0)
    crop_prepare_seconds = (0.0 if skipped else
                            (gpt_record or {}).get('runtime_crop_prepare_seconds', 0.0))
    tokens = (gpt_record or {}).get('usage') or {}
    return {
        'slide': local['slide'], 'truth': truth, 'prediction': prediction,
        'correct': prediction == truth if prediction else None,
        'gate_score': local['max_raw_mlp_score'], 'gate_decision': local['gate_decision'],
        'gate_threshold': local['gate_threshold'],
        'candidate_pool_before_cap': local['candidate_pool_before_cap'],
        'candidates_encoded': local['candidates_encoded'],
        'top8': local['top8'], 'local_seconds': local['local_seconds'],
        'runtime_crop_prepare_seconds': crop_prepare_seconds,
        'gpt_seconds': gpt_seconds,
        'total_sample_seconds': local['local_seconds'] + crop_prepare_seconds + gpt_seconds,
        'input_tokens': tokens.get('input_tokens', 0),
        'cached_input_tokens': (tokens.get('input_tokens_details') or {}).get('cached_tokens', 0),
        'output_tokens': tokens.get('output_tokens', 0),
        'cost_usd': (gpt_record or {}).get('cost_usd', 0.0),
        'gpt_status': 'SKIPPED_BY_GATE' if skipped else (gpt_record or {}).get('state', 'PENDING'),
        'gpt_parse_status': (gpt_record or {}).get('parse_status'),
    }


def summarize(results, rows, output, startup_seconds, key_prompt_seconds, started):
    by_id = {r['slide']: r for r in results}
    ordered = [by_id[Path(r['image']).stem] for r in rows if Path(r['image']).stem in by_id]
    completed = [r for r in ordered if r['prediction'] is not None]
    called = [r for r in ordered if r['gate_decision'] == 'CALL_GPT']
    gpt_answered = [r for r in called if r['gpt_status'] == 'COMPLETE']
    skipped = [r for r in ordered if r['gate_decision'] == 'SKIP_GPT']
    threshold_misses = [r for r in skipped if r['truth'] == 'TUMOR']
    archived_calls = []
    archive_dir = output / 'gpt' / 'archive'
    if archive_dir.is_dir():
        archived_calls = [json.loads(p.read_text()) for p in archive_dir.glob('*.json')]
    archived_cost = sum(float(r.get('cost_usd', 0.0)) for r in archived_calls)
    current_cost = sum(r['cost_usd'] for r in ordered)
    tp = sum(r['truth'] == r['prediction'] == 'TUMOR' for r in completed)
    tn = sum(r['truth'] == r['prediction'] == 'NORMAL' for r in completed)
    fp = sum(r['truth'] == 'NORMAL' and r['prediction'] == 'TUMOR' for r in completed)
    fn = sum(r['truth'] == 'TUMOR' and r['prediction'] == 'NORMAL' for r in completed)
    output_json = {
        'planned_test_slides': 129, 'slides_processed_so_far': len(ordered),
        'slides_with_local_results': len(ordered),
        'gate_threshold_used': ordered[0]['gate_threshold'] if ordered else None,
        'completed_predictions': len(completed), 'gpt_called': len(called),
        'gpt_responses_completed': len(gpt_answered),
        'invalid_gpt_answers': sum(r['prediction'] not in ('NORMAL', 'TUMOR')
                                   for r in gpt_answered),
        'recovered_truncated_gpt_answers': sum(
            r.get('gpt_parse_status') == 'recovered_truncated_json_prefix'
            for r in gpt_answered),
        'gpt_call_accuracy_so_far': (sum(r['correct'] is True for r in gpt_answered)
                                     / len(gpt_answered) if gpt_answered else None),
        'overall_accuracy_so_far': (sum(r['correct'] is True for r in completed)
                                   / len(completed) if completed else None),
        'gpt_skipped': len(skipped), 'skip_rate': len(skipped) / len(rows) if rows else None,
        'skipped_normal': sum(r['truth'] == 'NORMAL' for r in skipped),
        'threshold_missed_positive_count': len(threshold_misses),
        'threshold_missed_positive_ids': [r['slide'] for r in threshold_misses],
        'skipped_tumor_false_negatives': len(threshold_misses),
        'gate_positive_call_recall': (sum(r['truth'] == 'TUMOR' for r in called) / 49),
        'confusion_matrix_truth_rows_prediction_columns': ['NORMAL', 'TUMOR'],
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'accuracy': sum(r['correct'] is True for r in completed) / len(completed) if completed else None,
        'precision_tumor': tp / (tp + fp) if tp + fp else None,
        'recall_tumor': tp / (tp + fn) if tp + fn else None,
        'specificity': tn / (tn + fp) if tn + fp else None,
        'gpt_cost_usd_current_prompt': current_cost,
        'superseded_prompt_calls_archived': len(archived_calls),
        'superseded_prompt_cost_usd': archived_cost,
        'total_api_cost_estimate_including_archived_calls_usd': current_cost + archived_cost,
        'api_input_tokens': sum(r['input_tokens'] for r in ordered),
        'api_cached_input_tokens': sum(r['cached_input_tokens'] for r in ordered),
        'api_output_tokens': sum(r['output_tokens'] for r in ordered),
        'local_seconds_total': sum(r['local_seconds'] for r in ordered),
        'runtime_crop_prepare_seconds_total': sum(r['runtime_crop_prepare_seconds'] for r in ordered),
        'gpt_seconds_total': sum(r['gpt_seconds'] for r in ordered),
        'mean_local_seconds_per_slide': float(np.mean([r['local_seconds'] for r in ordered])) if ordered else None,
        'mean_runtime_crop_prepare_seconds_per_called_slide': float(np.mean(
            [r['runtime_crop_prepare_seconds'] for r in called])) if called else None,
        'mean_gpt_seconds_per_called_slide': float(np.mean([r['gpt_seconds'] for r in called])) if called else None,
        'model_startup_seconds_excluded_from_per_slide': startup_seconds,
        'hidden_key_prompt_seconds': key_prompt_seconds,
        'wall_seconds_this_invocation': time.perf_counter() - started,
        'interpretation_note': 'Official test labels were not used by the sampler, MLP, gate, or GPT prompt. '
                              'The quantized-encoder gate was calibrated on development validation; '
                              'this test set has '
                              'prior exploratory retrieval diagnostics, so it is not wholly untouched.',
        'cost_note': 'USD is estimated from response-reported tokens and published GPT-5.4 mini rates; '
                     'confirm account charges in the OpenAI Usage dashboard.',
    }
    atomic_json(output / 'summary.json', output_json)
    fields = ['slide', 'truth', 'prediction', 'correct', 'gate_score', 'gate_decision',
              'gate_threshold', 'candidate_pool_before_cap', 'candidates_encoded', 'local_seconds',
              'runtime_crop_prepare_seconds', 'gpt_seconds', 'total_sample_seconds',
              'input_tokens', 'cached_input_tokens',
              'output_tokens', 'cost_usd', 'gpt_status', 'gpt_parse_status']
    atomic_csv(output / 'per_slide.csv', ordered, fields)
    print(json.dumps(output_json, indent=2), flush=True)
    print(f'Progress: {len(ordered)}/129 slides | overall accuracy '
          f'{output_json["overall_accuracy_so_far"]} | GPT calls {len(gpt_answered)}/'
          f'{len(called)} completed | GPT-call accuracy '
          f'{output_json["gpt_call_accuracy_so_far"]} | threshold-missed positives '
          f'{len(threshold_misses)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=DEFAULT_TRAINING)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--gate-file', type=Path,
                        default=Path.home() / 'mlp_validation_gate_quantized_v1' / 'gate.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpt-model', default=GPT_MODEL)
    args = parser.parse_args()
    args.training, args.data, args.output, args.gate_file = [p.expanduser().resolve()
        for p in (args.training, args.data, args.output, args.gate_file)]
    if not args.output.is_relative_to(Path.home().resolve()):
        parser.error('Output must be under the user home directory.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('Quantized CONCH-Z requires CUDA; specify an available --device.')

    started = time.perf_counter()
    manifest, rows, data, encoder_path, checkpoint_path, config, gate_threshold = read_setup(args)
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / 'run_config.json'
    if run_path.exists():
        saved_config = json.loads(run_path.read_text())
        if saved_config != config and not compatible_resume_config(saved_config, config):
            raise ValueError('Output directory belongs to a different run configuration; choose a new --output.')
        if saved_config != config:
            atomic_json(run_path, config)
    else:
        atomic_json(run_path, config)
    existing = [load_local(args.output / 'local' / f'{Path(r["image"]).stem}.json',
                           config) for r in rows]
    pending_gpt = [r for r, local in zip(rows, existing)
                   if local and local['gate_decision'] == 'CALL_GPT'
                   and not has_current_gpt_result(
                       args.output / 'gpt' / f'{Path(r["image"]).stem}.json', args.gpt_model)]
    needs_key = any(local is None for local in existing) or bool(pending_gpt)
    if needs_key and not sys.stdin.isatty():
        raise RuntimeError('Run in an interactive terminal for the single hidden API-key prompt.')

    print(f'Official test: {len(rows)} slides (80 normal, 49 tumor).', flush=True)
    print(f'Validation-calibrated quantized gate: skip iff maximum raw MLP score <= '
          f'{gate_threshold:.6f}; top-{TOP_K} sent otherwise.', flush=True)
    if args.output == args.training or args.output.is_relative_to(args.training) \
            or args.training.is_relative_to(args.output):
        raise ValueError('Output must be separate from the training cache.')
    client = None
    key_prompt_seconds = 0.0
    if needs_key:
        print('The API key is requested once now, hidden, and is not written to disk.', flush=True)
        key_started = time.perf_counter()
        key = getpass.getpass('OpenAI API key (hidden; not saved): ')
        key_prompt_seconds = time.perf_counter() - key_started
        if not key:
            raise ValueError('Empty API key.')
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=180, max_retries=0)
        del key
    else:
        print('All local and GPT results are already complete; no API key is needed.', flush=True)
    model_started = time.perf_counter()
    print(f'Loading W8A8-quantized CONCH-Z visual layers and current FP32 MLP on '
          f'{args.device}...', flush=True)
    encoder, head, scaler, quantized_linears = load_models(
        encoder_path, checkpoint_path, args.device)
    startup_seconds = time.perf_counter() - model_started
    atomic_json(args.output / 'quantization.json', {
        'method': QUANTIZATION, 'visual_linear_layers_quantized': quantized_linears,
        'mlp': 'original saved FP32 current MLP; not quantized',
        'quantization_speedup_measured': False,
    })
    os.umask(0o077)
    results = []
    for i, row in enumerate(rows, 1):
        stem = Path(row['image']).stem
        local_path = args.output / 'local' / f'{stem}.json'
        local = load_local(local_path, config)
        if local is None:
            print(f'[{i}/{len(rows)}] Local pipeline: {stem}', flush=True)
            local = run_local(row, data, args.output, encoder, head, scaler,
                              args.device, args.batch_size, config, gate_threshold)
            print(f'  candidates={local["candidates_encoded"]}; max score='
                  f'{local["max_raw_mlp_score"]:.4f}; {local["gate_decision"]}; '
                  f'local {local["local_seconds"]:.1f}s', flush=True)
        else:
            print(f'[{i}/{len(rows)}] Reusing local result: {stem}', flush=True)

        gpt_record = None
        if local['gate_decision'] == 'CALL_GPT':
            print(f'  GPT top-8: {stem}', flush=True)
            if client is None:
                raise RuntimeError('A GPT request is pending but the API client was not initialized.')
            gpt_record = request_gpt(client, local, args.output, args.gpt_model, data)
            print(f'  GPT={gpt_record.get("answer")}; '
                  f'{gpt_record.get("gpt_seconds", 0):.1f}s; '
                  f'${gpt_record.get("cost_usd", 0):.5f}', flush=True)
        else:
            print('  GPT skipped by fixed gate; local prediction=NORMAL.', flush=True)
        results.append(make_slide_result(local, gpt_record))
        summarize(results, rows[:i], args.output, startup_seconds,
                  key_prompt_seconds, started)

    del encoder, head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'Final outputs: {args.output / "summary.json"} and {args.output / "per_slide.csv"}', flush=True)


if __name__ == '__main__':
    main()
