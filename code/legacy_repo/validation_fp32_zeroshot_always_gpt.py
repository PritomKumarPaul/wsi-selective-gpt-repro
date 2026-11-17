#!/usr/bin/env python3
"""Always-GPT CAMELYON16 validation run using cached CONCH-Z zero-shot ranking.

Reuses the existing unquantized CONCH-Z validation image embeddings, applies
the same one-grid/50%-tissue/500-candidate sampler as the MLP run, ranks crops
by tumor-vs-normal text-image cosine margin, and calls GPT for every slide.
Tumor annotation fractions are read only after ranking for retrieval metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

import validation_fp32_mlp_always_gpt as common


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_PROMPTS = HERE / 'prompts_camelyon16.json'
DEFAULT_OUTPUT = PROJECT / 'results/validation_fp32_zeroshot_always_gpt_v1'
FEATURE_DIM = 768


def get_prompt_classifier(model_dir: Path, prompt_path: Path, device: torch.device,
                          output: Path) -> tuple[np.ndarray, float, str]:
    prompt_hash = common.sha256(prompt_path)
    model_config_hash = common.sha256(model_dir / 'config.json')
    prototype_path = output / 'zero_shot_text_prototypes.npz'
    if prototype_path.is_file():
        with np.load(prototype_path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata'].item()))
            if metadata.get('prompt_sha256') == prompt_hash and metadata.get('model_config_sha256') == model_config_hash:
                prototypes = data['prototypes'].astype(np.float32)
                if prototypes.shape == (FEATURE_DIM, 2):
                    logging.info('Reusing cached CONCH-Z tumor/normal text prototypes.')
                    return prototypes, 0.0, prompt_hash

    if not (model_dir / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Local CONCH-Z weights not found in {model_dir}; no download attempted.')
    prompt_spec = json.loads(prompt_path.read_text())
    logging.info('Encoding the fixed zero-shot text prompts once on %s.', device)
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    ).eval().to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prototypes = model.zero_shot_classifier(
            [prompt_spec['classes']['tumor'], prompt_spec['classes']['normal']],
            prompt_spec['templates'], device=device
        ).float().cpu()
    elapsed = time.perf_counter() - started
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if tuple(prototypes.shape) != (FEATURE_DIM, 2):
        raise ValueError(f'Unexpected CONCH-Z text prototype shape {tuple(prototypes.shape)}')
    prototypes = torch.nn.functional.normalize(prototypes, dim=0).numpy().astype(np.float32)
    metadata = {'prompt_sha256': prompt_hash, 'model_config_sha256': model_config_hash,
                'encoder_precision': 'unquantized FP32',
                'class_order': ['tumor', 'normal']}
    prototype_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = prototype_path.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, prototypes=prototypes,
                            metadata=np.asarray(json.dumps(metadata)))
    tmp.replace(prototype_path)
    return prototypes, elapsed, prompt_hash


def run_local(row: dict, data_dir: Path, feature_cache: Path, output: Path,
              config: dict, feature_config: dict, prototypes: np.ndarray) -> dict:
    stem = Path(row['image']).stem
    feature_path = feature_cache / 'features' / f'{stem}.npz'
    started = time.perf_counter()
    with np.load(feature_path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
        if metadata.get('config') != feature_config or metadata.get('slide') != row:
            raise ValueError(f'{stem}: existing validation feature-cache metadata mismatch.')
        xy = saved['xy'].astype(np.int64)
        features = saved['features'].astype(np.float32)
        tissue = saved['tissue_fraction'].astype(np.float32)
        if features.shape != (len(xy), FEATURE_DIM) or tissue.shape != (len(xy),):
            raise ValueError(f'{stem}: cached feature or tissue-fraction shape mismatch.')

        image_path = data_dir / 'images' / row['image']
        import openslide
        with openslide.OpenSlide(str(image_path)) as slide:
            width, height = slide.dimensions
            mpp_x = mpp_y = None
            try:
                mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
                mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
            except (KeyError, ValueError):
                pass

        pool_idx, candidate_idx = common.cap_cached_candidates(
            xy, tissue, width, height)
        if len(candidate_idx) < common.TOP_K:
            raise ValueError(f'{stem}: only {len(candidate_idx)} eligible candidates; need {common.TOP_K}.')
        vectors = features[candidate_idx]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        similarities = vectors @ prototypes
        margins = similarities[:, 0] - similarities[:, 1]
        if not np.isfinite(margins).all():
            raise ValueError(f'{stem}: non-finite zero-shot similarity margins.')
        top_local = np.argsort(-margins, kind='stable')[:common.TOP_K]
        top_indices = candidate_idx[top_local]
        top8 = []
        for rank, (cache_idx, margin, pair) in enumerate(
                zip(top_indices, margins[top_local], similarities[top_local]), 1):
            top8.append({
                'rank': rank,
                'x_level0': int(xy[cache_idx, 0]),
                'y_level0': int(xy[cache_idx, 1]),
                'zero_shot_margin': float(margin),
                'tumor_similarity': float(pair[0]),
                'normal_similarity': float(pair[1]),
                'tissue_fraction': float(tissue[cache_idx]),
            })

        # The retrieval list is now fixed. Read cached mask-derived labels only
        # for evaluation; they are never passed to GPT or used in ranking.
        tumor_fraction = saved['tumor_fraction'].astype(np.float32)
        if tumor_fraction.shape != (len(xy),):
            raise ValueError(f'{stem}: cached tumor-mask fraction shape mismatch.')
        selected_fractions = tumor_fraction[top_indices]
        for patch, fraction in zip(top8, selected_fractions):
            patch['tumor_fraction_eval_only'] = float(fraction)

    hits = int(np.count_nonzero(selected_fractions > 0))
    strong_hits = int(np.count_nonzero(selected_fractions >= 0.001))
    record = {
        'config': config,
        'slide': stem,
        'truth_for_scoring_only': 'TUMOR' if row['type'] == 'tumor' else 'NORMAL',
        'source_image': row['image'],
        'slide_dimensions_level0': [width, height],
        'mpp': [mpp_x, mpp_y],
        'candidate_pool_before_cap': int(len(pool_idx)),
        'candidates_encoded': int(len(candidate_idx)),
        'max_zero_shot_margin': float(np.max(margins)),
        'top8': top8,
        'local_seconds': time.perf_counter() - started,
        'mask_fraction_read_after_ranking': True,
        'mask_used_for_selection': False,
        'top8_mask_hit_count_any_overlap': hits,
        'top8_mask_hit_count_at_least_0_1_percent': strong_hits,
        'source_feature_cache': str(feature_path),
        'source_embedding_storage_dtype': 'float16 cache of normalized unquantized FP32 CONCH-Z output',
    }
    common.atomic_json(output / 'local' / f'{stem}.json', record)
    return record


def aggregate(rows: list[dict], local_records: dict, gpt_records: dict) -> tuple[list[dict], dict]:
    joined = []
    for row in rows:
        stem = Path(row['image']).stem
        local, gpt = local_records.get(stem), gpt_records.get(stem)
        if local is None:
            continue
        truth = 'TUMOR' if row['type'] == 'tumor' else 'NORMAL'
        prediction = gpt.get('answer') if gpt and gpt.get('state') == 'COMPLETE' else None
        joined.append({
            'slide': stem, 'truth': truth, 'prediction': prediction,
            'correct': prediction == truth if prediction else None,
            'max_zero_shot_margin': local['max_zero_shot_margin'],
            'candidate_pool_before_cap': local['candidate_pool_before_cap'],
            'candidates_reused': local['candidates_encoded'],
            'top8': json.dumps(local['top8'], separators=(',', ':')),
            'top8_mask_hit_count_any_overlap': local['top8_mask_hit_count_any_overlap'],
            'top8_mask_hit_count_at_least_0_1_percent': local['top8_mask_hit_count_at_least_0_1_percent'],
            'top8_mask_fractions_eval_only': json.dumps(
                [p['tumor_fraction_eval_only'] for p in local['top8']], separators=(',', ':')),
            'local_seconds': local['local_seconds'],
            'gpt_seconds': gpt.get('gpt_seconds') if gpt else None,
            'input_tokens': (gpt.get('usage') or {}).get('input_tokens') if gpt else None,
            'output_tokens': (gpt.get('usage') or {}).get('output_tokens') if gpt else None,
            'cost_usd': gpt.get('cost_usd') if gpt else None,
            'gpt_status': gpt.get('state') if gpt else 'PENDING',
        })
    completed = [r for r in joined if r['prediction'] in ('NORMAL', 'TUMOR')]
    tp = sum(r['truth'] == 'TUMOR' and r['prediction'] == 'TUMOR' for r in completed)
    tn = sum(r['truth'] == 'NORMAL' and r['prediction'] == 'NORMAL' for r in completed)
    fp = sum(r['truth'] == 'NORMAL' and r['prediction'] == 'TUMOR' for r in completed)
    fn = sum(r['truth'] == 'TUMOR' and r['prediction'] == 'NORMAL' for r in completed)
    normal_n = sum(r['truth'] == 'NORMAL' for r in completed)
    tumor_n = sum(r['truth'] == 'TUMOR' for r in completed)
    positives = [r for r in joined if r['truth'] == 'TUMOR']
    hit_crops = sum(r['top8_mask_hit_count_any_overlap'] for r in positives)
    strong_crops = sum(r['top8_mask_hit_count_at_least_0_1_percent'] for r in positives)
    summary = {
        'protocol': 'validation all-GPT; cached unquantized CONCH-Z zero-shot top-8; no gate',
        'planned_slides': len(rows), 'local_results': len(joined),
        'completed_gpt_responses': len(completed), 'gpt_calls': sum(
            r['gpt_status'] in ('COMPLETE', 'INVALID_OUTPUT', 'MISSING_USAGE',
                                'ERROR_OR_UNKNOWN_BILLING', 'IN_FLIGHT') for r in joined),
        'always_call_gpt': True,
        'accuracy': sum(r['correct'] is True for r in completed) / len(completed) if completed else None,
        'false_positives': fp,
        'false_positive_rate_on_normal_slides': fp / normal_n if normal_n else None,
        'false_negative_count': fn,
        'sensitivity_tumor': tp / tumor_n if tumor_n else None,
        'specificity_normal': tn / normal_n if normal_n else None,
        'precision_tumor': tp / (tp + fp) if tp + fp else None,
        'positive_slides_with_any_tumor_mask_hit_in_top8': sum(
            r['top8_mask_hit_count_any_overlap'] > 0 for r in positives),
        'positive_slides_total': len(positives),
        'positive_slides_with_strong_top8_mask_hit_ge_0_1_percent': sum(
            r['top8_mask_hit_count_at_least_0_1_percent'] > 0 for r in positives),
        'positive_slide_top8_mask_overlapping_crops': hit_crops,
        'positive_slide_top8_strong_mask_overlapping_crops_ge_0_1_percent': strong_crops,
        'positive_slide_top8_mask_overlap_crop_rate': hit_crops / (common.TOP_K * len(positives)) if positives else None,
        'confusion_matrix_truth_rows_prediction_columns': ['NORMAL', 'TUMOR'],
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'api_cost_estimate_usd': sum(float(r['cost_usd'] or 0) for r in joined),
        'api_input_tokens': sum(int(r['input_tokens'] or 0) for r in joined),
        'api_output_tokens': sum(int(r['output_tokens'] or 0) for r in joined),
        'interpretation_note': 'The same validation split was used for development; this is not an independent test. Tumor mask fractions are read only after zero-shot ranking and never sent to GPT.',
    }
    return joined, summary


def save_progress(output: Path, rows: list[dict], local: dict, gpt: dict) -> dict:
    joined, summary = aggregate(rows, local, gpt)
    fields = ['slide', 'truth', 'prediction', 'correct', 'max_zero_shot_margin',
              'candidate_pool_before_cap', 'candidates_reused', 'top8',
              'top8_mask_hit_count_any_overlap', 'top8_mask_hit_count_at_least_0_1_percent',
              'top8_mask_fractions_eval_only', 'local_seconds', 'gpt_seconds',
              'input_tokens', 'output_tokens', 'cost_usd', 'gpt_status']
    common.atomic_csv(output / 'per_slide.csv', joined, fields)
    common.atomic_json(output / 'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=common.DEFAULT_TRAINING)
    parser.add_argument('--data', type=Path, default=common.DEFAULT_DATA)
    parser.add_argument('--feature-cache', type=Path, default=common.DEFAULT_FEATURE_CACHE)
    parser.add_argument('--protocol-reference', type=Path, default=common.DEFAULT_GATE)
    parser.add_argument('--prompts', type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cpu', help='Device used only for the small prompt-prototype encoding step.')
    parser.add_argument('--budget-usd', type=float, default=common.BUDGET_USD)
    parser.add_argument('--request-budget-allowance-usd', type=float,
                        default=common.REQUEST_BUDGET_ALLOWANCE_USD)
    args = parser.parse_args()
    for name in ('training', 'data', 'feature_cache', 'protocol_reference', 'prompts', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.output.is_relative_to(PROJECT):
        parser.error(f'Output must remain inside the project repo: {PROJECT}')
    if args.budget_usd <= 0 or args.request_budget_allowance_usd <= 0:
        parser.error('Budget values must be positive.')
    if args.device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    manifest, rows, data_dir, _checkpoint, config, feature_config = common.read_setup(args)
    model_dir = Path(manifest['config']['model_dir'])
    prototypes, prompt_seconds, prompt_hash = get_prompt_classifier(
        model_dir, args.prompts, torch.device(args.device), args.output)
    prompt_spec = json.loads(args.prompts.read_text())
    config.update({
        'protocol': 'CAMELYON16 validation; cached unquantized CONCH-Z zero-shot retrieval; always GPT; no gate',
        'ranking': 'cosine(image, tumor prototype) - cosine(image, normal prototype); select top 8',
        'zero_shot_prompt_file': str(args.prompts),
        'zero_shot_prompt_sha256': prompt_hash,
        'zero_shot_prompt_spec': prompt_spec,
        'zero_shot_text_prototype_cache': str(args.output / 'zero_shot_text_prototypes.npz'),
        'script_sha256': common.sha256(Path(__file__)),
    })
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'run_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Output directory contains a different run configuration; use a new output path.')
    else:
        common.atomic_json(config_path, config)

    local_records, gpt_records = common.load_records(args.output, rows, config)
    pending = [r for r in rows if Path(r['image']).stem not in gpt_records
               or gpt_records[Path(r['image']).stem].get('state') != 'COMPLETE']
    if pending and not sys.stdin.isatty():
        raise RuntimeError('Run in an interactive terminal; API key is prompted once and never saved.')
    client = None
    if pending:
        import getpass
        key = getpass.getpass('OpenAI API key (hidden; not saved): ')
        if not key.strip():
            raise RuntimeError('No API key entered.')
        from openai import OpenAI
        client = OpenAI(api_key=key.strip())
        del key

    logging.info('All-GPT zero-shot validation: 50 slides; same candidates/top-8/GPT settings as MLP run.')
    logging.info('Prompt prototype encoding/reuse setup took %.2fs; all image embeddings are reused.', prompt_seconds)
    logging.info('Reusing unquantized CONCH-Z image features and cached MLP-independent candidate pool; no image encoding.')
    for i, row in enumerate(rows, 1):
        stem = Path(row['image']).stem
        local = local_records.get(stem)
        if local is None:
            local = run_local(row, data_dir, args.feature_cache, args.output,
                              config, feature_config, prototypes)
            local_records[stem] = local
            logging.info('%d/50 %s prompt top8 mask hits=%d (>=0.1%%: %d)', i, stem,
                         local['top8_mask_hit_count_any_overlap'],
                         local['top8_mask_hit_count_at_least_0_1_percent'])
            save_progress(args.output, rows, local_records, gpt_records)
        gpt = gpt_records.get(stem)
        if gpt is None or gpt.get('state') != 'COMPLETE':
            _, sofar = aggregate(rows, local_records, gpt_records)
            if sofar['api_cost_estimate_usd'] + args.request_budget_allowance_usd > args.budget_usd:
                raise RuntimeError(f'Budget guard stopped before {stem}; estimated spend plus request allowance exceeds ${args.budget_usd:.2f}.')
            gpt = common.request_gpt(client, local, data_dir, args.output)
            gpt_records[stem] = gpt
            summary = save_progress(args.output, rows, local_records, gpt_records)
            logging.info('%s GPT=%s; accuracy so far=%s; false positives=%d; cost=$%.6f',
                         stem, gpt['answer'], summary['accuracy'],
                         summary['false_positives'], gpt['cost_usd'])
        if gpt.get('state') != 'COMPLETE':
            raise RuntimeError(f'{stem}: GPT state={gpt.get("state")}; stopped.')

    _, summary = aggregate(rows, local_records, gpt_records)
    if summary['completed_gpt_responses'] != len(rows):
        raise RuntimeError('Not all validation slides have completed GPT responses.')
    logging.info('Finished: accuracy=%.3f, FP=%d, FN=%d, estimated cost=$%.4f',
                 summary['accuracy'], summary['false_positives'],
                 summary['false_negative_count'], summary['api_cost_estimate_usd'])
    logging.info('Outputs: %s', args.output)


if __name__ == '__main__':
    main()
