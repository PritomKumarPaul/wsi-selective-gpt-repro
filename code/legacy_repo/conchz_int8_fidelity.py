"""Training-free CONCH-Z vision INT8 weight-storage fidelity experiment.

This uses a reference dequantize-on-forward Linear, so it measures score and
ranking changes plus persistent weight storage, not optimized INT8 speed.
No model or data download is performed. Precomputed text class embeddings are
kept in FP32; only visual-tower Linear weights are quantized per output row.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import openslide
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from camelyon_conchz_pilot import DEFAULT_DATA, DEFAULT_MODEL, encode_candidates


HERE = Path(__file__).resolve().parent
DEFAULT_REPORTS = [HERE / 'examples/conchz_pilot/pair1_report.json',
                   HERE / 'examples/conchz_pilot/pair2_report.json']


class Int8StoredLinear(nn.Module):
    """Symmetric per-output-channel INT8 weights; FP32 dequantized GEMM.

    This is a reference implementation for fidelity, not an accelerated kernel.
    """

    def __init__(self, layer):
        super().__init__()
        weight = layer.weight.detach()
        scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127
        qweight = (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer('qweight', qweight)
        self.register_buffer('scale', scale)
        if layer.bias is None:
            self.bias = None
        else:
            self.register_buffer('bias', layer.bias.detach().clone())

    def forward(self, x):
        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)[:, None]
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def replace_linears(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8StoredLinear(child))
            count += 1
        else:
            count += replace_linears(child)
    return count


def state_bytes(module):
    return sum(t.nelement() * t.element_size() for t in module.state_dict().values())


def load_candidates(paths):
    records = []
    for path in paths:
        report = json.loads(path.read_text())
        for slide in report['slides']:
            records.append({'stem': slide['slide'], 'dimensions': slide['dimensions'],
                            'candidates': slide['all_candidates'], 'report': str(path)})
    return records


def score_slides(model, classifier, records, data, device, batch_size):
    outputs = {}
    start = time.perf_counter()
    for record in records:
        stem = record['stem']
        path = data / 'images' / f'{stem}.tif'
        if not path.is_file():
            raise FileNotFoundError(path)
        print('Encoding', stem, flush=True)
        with openslide.OpenSlide(str(path)) as slide:
            if list(slide.dimensions) != record['dimensions']:
                raise ValueError(f'Dimensions changed for {stem}')
            vectors = encode_candidates(slide, model, record['candidates'],
                                        1024, device, batch_size)
        sims = (vectors @ classifier).numpy()
        outputs[stem] = {'features': vectors.numpy(),
                         'margins': sims[:, 0] - sims[:, 1]}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return outputs, time.perf_counter() - start


def top_ids(margins, record, k=4):
    order = sorted(range(len(margins)), key=lambda i: (-float(margins[i]),
                                                       record['candidates'][i]['candidate_id']))
    return order[:k]


def run(args):
    if not (args.model / 'model.safetensors').is_file():
        raise FileNotFoundError(f'CONCH-Z weights missing locally: {args.model}')
    torch.manual_seed(0)
    device = torch.device(args.device)
    records = load_candidates(args.reports)
    prompts = json.loads(args.prompts.read_text())
    print('Loading CONCH-Z...', flush=True)
    model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True,
                                      local_files_only=True).eval().to(device)
    with torch.inference_mode():
        classifier = model.zero_shot_classifier(
            [prompts['classes']['tumor'], prompts['classes']['normal']],
            prompts['templates'], device=device).float().cpu()
    fp32_vision_bytes = state_bytes(model.visual)
    if args.reuse_archived_baseline:
        print('Using archived FP32 margins as the reference...', flush=True)
        baseline = {record['stem']: {
            'margins': np.asarray([p['score_margin'] for p in record['candidates']], dtype=np.float32),
            'features': None} for record in records}
        baseline_seconds = None
    else:
        print('FP32 reference scoring...', flush=True)
        baseline, baseline_seconds = score_slides(model, classifier, records, args.data,
                                                   device, args.batch_size)
    print('Replacing vision Linear weights with INT8 storage...', flush=True)
    converted = replace_linears(model.visual)
    int8_vision_bytes = state_bytes(model.visual)
    print('INT8 weight-storage reference scoring...', flush=True)
    quantized, quantized_seconds = score_slides(model, classifier, records, args.data,
                                                 device, args.batch_size)
    result = {
        'purpose': 'Training-free INT8 weight fidelity, not accelerated deployment',
        'model': 'MahmoodLab/CONCH-Z', 'device': str(device),
        'model_path': str(args.model), 'prompts_path': str(args.prompts),
        'method': 'vision nn.Linear symmetric per-output-channel INT8 weights; dequantize to input dtype before dense F.linear; text embeddings cached from FP32 model',
        'baseline_source': 'archived report score margins' if args.reuse_archived_baseline else 'fresh FP32 encoding',
        'vision_linear_modules_converted': converted,
        'vision_fp32_state_bytes': fp32_vision_bytes,
        'vision_int8_stored_state_bytes': int8_vision_bytes,
        'vision_storage_reduction_fraction': 1 - int8_vision_bytes / fp32_vision_bytes,
        'baseline_scoring_seconds': round(baseline_seconds, 3) if baseline_seconds is not None else None,
        'int8_reference_scoring_seconds': round(quantized_seconds, 3),
        'top_k': 4, 'slides': []
    }
    for record in records:
        stem = record['stem']
        base, quant = baseline[stem], quantized[stem]
        base_top = top_ids(base['margins'], record)
        quant_top = top_ids(quant['margins'], record)
        items = record['candidates']
        archived_top = sorted(range(len(items)), key=lambda i: (-items[i]['score_margin'],
                                                                 items[i]['candidate_id']))[:4]
        def hit_count(indices):
            return sum(bool(items[i]['tumor_hit']) for i in indices)
        emb_cos = (F.cosine_similarity(torch.from_numpy(base['features']),
                                      torch.from_numpy(quant['features']), dim=1).numpy()
                   if base['features'] is not None else None)
        entry = {
            'slide': stem, 'candidates': len(items),
            'archived_top4_ids': [items[i]['candidate_id'] for i in archived_top],
            'baseline_top4_ids': [items[i]['candidate_id'] for i in base_top],
            'int8_top4_ids': [items[i]['candidate_id'] for i in quant_top],
            'baseline_matches_archived': base_top == archived_top,
            'top4_set_overlap': len(set(base_top) & set(quant_top)),
            'baseline_top4_mask_hits': hit_count(base_top),
            'int8_top4_mask_hits': hit_count(quant_top),
            'margin_mean_absolute_error': float(np.mean(np.abs(base['margins'] - quant['margins']))),
            'margin_max_absolute_error': float(np.max(np.abs(base['margins'] - quant['margins']))),
            'embedding_cosine_mean': float(np.mean(emb_cos)) if emb_cos is not None else None,
            'embedding_cosine_min': float(np.min(emb_cos)) if emb_cos is not None else None,
        }
        result['slides'].append(entry)
        print(stem, 'top4 overlap', entry['top4_set_overlap'],
              'hits', entry['baseline_top4_mask_hits'], '->', entry['int8_top4_mask_hits'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Wrote', args.output, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--prompts', type=Path, default=HERE / 'prompts_camelyon16.json')
    parser.add_argument('--reports', type=Path, nargs='+', default=DEFAULT_REPORTS)
    parser.add_argument('--output', type=Path, default=HERE / 'outputs/conchz_int8_fidelity/report.json')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--reuse-archived-baseline', action='store_true',
                        help='Compare INT8 scores with FP32 margins already saved in input reports')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
