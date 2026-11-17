"""Training-free CONCH-Z patch retrieval on annotated CAMELYON16 slides.

This is a research pipeline check, not a medical diagnosis or a benchmark.
The tumor mask is opened only after patch coordinates and model scores are fixed.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel

from wsi_core import patch_grid, tissue_mask


DEFAULT_DATA = Path.home() / 'data' / 'camelyon16_subset'
DEFAULT_MODEL = Path.home() / 'models' / 'conch-z'
MODEL_REVISION = '60211fced9c25a014494e248eedb6a6be80109f7'


def candidates_from_thumbnail(slide, patch_size, min_tissue):
    """Return level-0 grid positions that pass thumbnail HSV/morphology filtering."""
    width, height = slide.dimensions
    thumb = np.asarray(slide.get_thumbnail((2048, 2048)).convert('RGB'))
    mask = tissue_mask(thumb)
    positions = patch_grid(width, height, thumb.shape, mask, np.zeros_like(mask), patch_size)
    eligible = [p for p in positions if p['tissue_fraction'] >= min_tissue]
    return eligible, len(positions), thumb, mask


def spatial_shortlist(eligible, limit, width, height):
    """Deterministic farthest-point sampling, seeded by the strongest tissue tile."""
    if not eligible:
        raise ValueError('No tiles passed the tissue filter')
    xy = np.asarray([[(p['x'] + 512) / width, (p['y'] + 512) / height]
                     for p in eligible], dtype=np.float64)
    fractions = np.asarray([p['tissue_fraction'] for p in eligible])
    chosen = []
    used = np.zeros(len(eligible), dtype=bool)
    distances = np.full(len(eligible), np.inf)
    index = int(np.argmax(fractions))
    for _ in range(min(limit, len(eligible))):
        chosen.append(eligible[index].copy())
        used[index] = True
        distances = np.minimum(distances, ((xy - xy[index]) ** 2).sum(axis=1))
        distances[used] = -1
        # Tissue fraction breaks equal-distance ties without consulting masks.
        index = int(np.argmax(distances + 1e-9 * fractions))
    for idx, item in enumerate(chosen):
        item['candidate_id'] = idx
    return chosen


def encode_candidates(slide, model, items, patch_size, device, batch_size):
    embeddings = []
    for offset in range(0, len(items), batch_size):
        batch = items[offset:offset + batch_size]
        tensors = []
        for item in batch:
            patch = slide.read_region((item['x'], item['y']), 0,
                                      (patch_size, patch_size)).convert('RGB')
            tensors.append(model.eval_transform(patch))
        images = torch.stack(tensors).to(device)
        with torch.inference_mode():
            embeddings.append(model.encode_image(images).float().cpu())
        print(f'  encoded {min(offset + batch_size, len(items))}/{len(items)} patches', flush=True)
    return torch.cat(embeddings)


def mask_measure(mask_slide, item, patch_size):
    labels = np.asarray(mask_slide.read_region((item['x'], item['y']), 0,
                                                (patch_size, patch_size)).convert('RGB'))[:, :, 0]
    pixels = int(np.count_nonzero(labels == 2))
    return pixels, pixels / labels.size


def save_montage(slide, items, out_path, patch_size, title):
    # Downsampled display copies; coordinates and reported scores are unchanged.
    cell = 240
    cols = min(4, max(1, len(items)))
    rows = math.ceil(len(items) / cols)
    canvas = Image.new('RGB', (cols * cell, rows * (cell + 42) + 40), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(20, 30, 45))
    for j, item in enumerate(items):
        patch = slide.read_region((item['x'], item['y']), 0,
                                  (patch_size, patch_size)).convert('RGB')
        patch.thumbnail((cell - 8, cell - 8))
        px = (j % cols) * cell + 4
        py = (j // cols) * (cell + 42) + 40
        canvas.paste(patch, (px, py))
        draw.text((px + 3, py + cell),
                  f"#{item['candidate_id']} score {item['score_margin']:+.3f}",
                  fill=(20, 30, 45))
        draw.text((px + 3, py + cell + 17),
                  f"mask tumor {item['tumor_fraction']:.1%}", fill=(20, 30, 45))
    canvas.save(out_path)


def run(args):
    if args.patch_size != 1024:
        raise ValueError('This pilot fixes the native patch size at 1024 pixels')
    if args.top_k < 1 or args.candidates < args.top_k:
        raise ValueError('Require 1 <= top-k <= candidates')
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_spec = json.loads(args.prompts.read_text())
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    start = time.perf_counter()
    model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True,
                                      local_files_only=True).eval().to(device)
    with torch.inference_mode():
        classifier = model.zero_shot_classifier(
            [prompt_spec['classes']['tumor'], prompt_spec['classes']['normal']],
            prompt_spec['templates'], device=device).float().cpu()
    model_seconds = time.perf_counter() - start

    results = {
        'purpose': 'Exploratory local CONCH-Z ranking; not a diagnostic evaluation',
        'model': 'MahmoodLab/CONCH-Z', 'model_revision': MODEL_REVISION,
        'model_path': str(args.model), 'prompts_path': str(args.prompts),
        'prompt_spec': prompt_spec, 'device': str(device),
        'settings': {'patch_size_level0': args.patch_size, 'thumbnail_max_size': 2048,
                     'min_tissue_fraction': args.min_tissue,
                     'shortlist_method': 'fixed farthest-point spread, seed=max thumbnail tissue fraction',
                     'candidates_per_slide': args.candidates, 'top_k': args.top_k,
                     'mask_positive_label': 2, 'random_seed': args.seed,
                     'score': 'cosine(tumor class average) - cosine(normal class average)'},
        'model_load_and_text_seconds': round(model_seconds, 3), 'slides': []
    }
    rng = np.random.default_rng(args.seed)
    for stem in args.slides:
        slide_path = args.data / 'images' / f'{stem}.tif'
        mask_path = args.data / 'masks' / f'{stem}_mask.tif'
        if not slide_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f'Need complete slide/mask pair: {stem}')
        print(f'Processing {stem}', flush=True)
        slide_start = time.perf_counter()
        with openslide.OpenSlide(str(slide_path)) as slide:
            eligible, grid_count, _thumb, _tissue = candidates_from_thumbnail(
                slide, args.patch_size, args.min_tissue)
            candidates = spatial_shortlist(eligible, args.candidates, *slide.dimensions)
            features = encode_candidates(slide, model, candidates,
                                         args.patch_size, device, args.batch_size)
            similarities = (features @ classifier).numpy()
            for j, item in enumerate(candidates):
                item['similarity_tumor'] = round(float(similarities[j, 0]), 6)
                item['similarity_normal'] = round(float(similarities[j, 1]), 6)
                item['score_margin'] = round(float(similarities[j, 0] - similarities[j, 1]), 6)
            ranked = sorted(candidates, key=lambda p: (-p['score_margin'], p['candidate_id']))
            selected = ranked[:args.top_k]

            # Evaluation starts here. No mask data influenced candidate generation or ranking.
            with openslide.OpenSlide(str(mask_path)) as truth:
                if slide.dimensions != truth.dimensions:
                    raise ValueError(f'Level-0 slide/mask size mismatch: {stem}')
                for item in candidates:
                    pixels, fraction = mask_measure(truth, item, args.patch_size)
                    item['tumor_pixels'] = pixels
                    item['tumor_fraction'] = round(fraction, 6)
                    item['tumor_hit'] = pixels > 0
            random_ids = rng.choice(len(candidates), size=args.top_k, replace=False).tolist()
            random_selected = [candidates[j] for j in random_ids]
            selected_hits = sum(p['tumor_hit'] for p in selected)
            candidate_hits = sum(p['tumor_hit'] for p in candidates)
            random_hits = sum(p['tumor_hit'] for p in random_selected)
            result = {
                'slide': stem, 'known_class_from_filename': 'tumor' if stem.startswith('tumor') else 'normal',
                'dimensions': list(slide.dimensions), 'virtual_grid_positions': grid_count,
                'eligible_tissue_positions': len(eligible), 'decoded_candidate_patches': len(candidates),
                'candidate_tumor_hits': candidate_hits,
                'candidate_tumor_hit_fraction': round(candidate_hits / len(candidates), 4),
                'top_k_tumor_hits': selected_hits,
                'top_k_any_tumor_hit': selected_hits > 0,
                'top_k_hit_precision': round(selected_hits / len(selected), 4),
                'random_k_tumor_hits': random_hits,
                'random_k_expected_hits': round(args.top_k * candidate_hits / len(candidates), 4),
                'top_k': selected, 'random_k': random_selected,
                'all_candidates': candidates,
                'elapsed_seconds': round(time.perf_counter() - slide_start, 3),
            }
            save_montage(slide, selected, args.output / f'{stem}_top{args.top_k}.jpg',
                         args.patch_size, f'{stem}: CONCH-Z top {args.top_k}')
            result['montage'] = f'{stem}_top{args.top_k}.jpg'
            results['slides'].append(result)
            print(f"  {candidate_hits}/{len(candidates)} candidates touch tumor; "
                  f"top {args.top_k}: {selected_hits}; random: {random_hits}", flush=True)
    results['total_seconds'] = round(time.perf_counter() - start, 3)
    output = args.output / 'report.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Wrote {output}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--prompts', type=Path,
                        default=Path(__file__).with_name('prompts_camelyon16.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('outputs') / 'camelyon_conchz_pilot')
    parser.add_argument('--slides', nargs='+', default=['normal_108', 'tumor_091'])
    parser.add_argument('--patch-size', type=int, default=1024)
    parser.add_argument('--min-tissue', type=float, default=0.5)
    parser.add_argument('--candidates', type=int, default=128)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
