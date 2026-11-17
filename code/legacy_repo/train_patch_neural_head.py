#!/usr/bin/env python3
"""Train PyTorch linear/MLP coverage heads on existing CONCH-Z features.

No image reads, CONCH-Z loading, downloads or cloud calls. Select epoch count
using an inner WSI split from the 200 training slides, then refit on all 200.
The original 50 validation slides are evaluated once after final training.
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from patch_fraction_regression import (
    atomic_json, evaluate_regressor, read_cache, storage_guard,
)


def make_head(kind, hidden=128, dropout=0.2):
    if kind == 'linear':
        return nn.Linear(768, 1)
    return nn.Sequential(nn.Linear(768, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))


def arrays(records):
    xs, ys, ws = [], [], []
    for _row, cache in records:
        y = cache['fraction'].astype(np.float32)
        w = np.full(len(y), 1 / len(y), dtype=np.float32)
        if (y > 0).any() and (y == 0).any():
            w[y > 0] = 0.5 / (y > 0).sum()
            w[y == 0] = 0.5 / (y == 0).sum()
        xs.append(cache['features'].astype(np.float32)); ys.append(y); ws.append(w)
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    w *= len(w) / w.sum()
    return x, y, w


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_on(model, x, y, weights, device, batch_size):
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            stop = start + batch_size
            prediction = model(torch.from_numpy(x[start:stop]).to(device)).flatten()
            target = torch.from_numpy(y[start:stop]).to(device)
            w = torch.from_numpy(weights[start:stop]).to(device)
            total += float(((prediction-target).square()*w).sum().cpu())
            count += len(target)
    return total / count


def train_epochs(args, train, validation=None, epochs=None, phase='inner'):
    set_seed(args.seed)
    x, y, weights = arrays(train)
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x).astype(np.float32)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y),
                                     torch.from_numpy(weights)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    if validation is not None:
        vx, vy, vw = arrays(validation)
        vx = scaler.transform(vx).astype(np.float32)
    model = make_head(args.head, args.hidden, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    history, best, best_epoch, stale = [], float('inf'), 1, 0
    for epoch in range(1, (epochs if epochs is not None else args.epochs) + 1):
        model.train()
        for batch_x, batch_y, batch_w in loader:
            batch_x, batch_y, batch_w = (v.to(args.device) for v in (batch_x, batch_y, batch_w))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x).flatten()
            loss = ((prediction-batch_y).square()*batch_w).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; reduce --lr')
            loss.backward()
            optimizer.step()
        train_loss = loss_on(model, x, y, weights, args.device, args.batch_size)
        val_loss = (loss_on(model, vx, vy, vw, args.device, args.batch_size)
                    if validation is not None else None)
        history.append({'phase': phase, 'epoch': epoch, 'train_weighted_mse': train_loss,
                        'inner_validation_weighted_mse': val_loss})
        message = f'{phase} epoch {epoch:03d}: train MSE={train_loss:.6f}'
        if val_loss is not None:
            message += f' | inner validation MSE={val_loss:.6f}'
        print(message, flush=True)
        if val_loss is not None:
            if val_loss < best:
                best, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
            if stale >= args.patience:
                print(f'Early stop; selected epoch {best_epoch} from inner validation.', flush=True)
                break
    return model, scaler, history, best_epoch


class Predictor:
    """Minimal predict interface for the existing common evaluation code."""
    def __init__(self, model, device, batch_size):
        self.model, self.device, self.batch_size = model.eval(), device, batch_size

    def predict(self, features):
        result = []
        with torch.inference_mode():
            for start in range(0, len(features), self.batch_size):
                x = torch.as_tensor(features[start:start+self.batch_size], dtype=torch.float32,
                                    device=self.device)
                result.append(self.model(x).flatten().cpu().numpy())
        return np.concatenate(result)


def save_history(output, history):
    atomic_json(output / 'history.json', history)
    with (output / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, phase in zip(axes, ('inner', 'refit')):
        rows = [r for r in history if r['phase'] == phase]
        ax.plot([r['epoch'] for r in rows], [r['train_weighted_mse'] for r in rows], label='Training')
        if phase == 'inner':
            ax.plot([r['epoch'] for r in rows], [r['inner_validation_weighted_mse'] for r in rows],
                    label='Inner validation')
        ax.set(xlabel='Epoch', ylabel='Weighted MSE', title=phase.capitalize())
        ax.legend(); ax.grid(alpha=0.2)
    fig.savefig(output / 'loss_curves.png', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan', 'fit'))
    p.add_argument('--head', choices=('linear', 'mlp'), default='mlp')
    p.add_argument('--cache', type=Path, default=Path.home() / 'wsi_patch_fraction_cache_v1')
    p.add_argument('--output', type=Path)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--inner-val-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if (min(args.epochs, args.patience, args.batch_size, args.hidden, args.threads) < 1
            or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1
            or not 0.1 <= args.inner_val_fraction <= 0.4):
        p.error('Invalid positive counts, optimizer settings, dropout or inner validation fraction')
    args.cache = args.cache.expanduser().resolve()
    args.output = (args.output or args.cache / f'neural_{args.head}_seed{args.seed}').expanduser().resolve()
    if (not args.output.is_relative_to(Path.home()) or args.output == args.cache
            or args.output.is_relative_to(args.cache / 'features')):
        p.error('Use a separate result directory under the user home directory, outside the feature folder')
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    train_rows = [r for r in manifest['slides'] if r['split'] == 'train']
    val_rows = [r for r in manifest['slides'] if r['split'] == 'validation']
    strata = [f'{r["class"]}:{r["center"]}' for r in train_rows]
    inner_train, inner_val = train_test_split(train_rows, test_size=args.inner_val_fraction,
                                            stratify=strata, random_state=args.seed)
    model = make_head(args.head, args.hidden, args.dropout)
    count = sum(v.numel() for v in model.parameters())
    print(f'{args.head}: {model}\nTrainable head parameters: {count:,}; CONCH-Z frozen.', flush=True)
    print(f'Inner split {len(inner_train)}/{len(inner_val)} WSIs; final refit {len(train_rows)} WSIs; '
          f'final validation {len(val_rows)} WSIs. Output: {args.output}', flush=True)
    if args.stage == 'plan':
        print('Plan only; no training or files written.')
        return
    if (args.output / 'run_config.json').exists():
        raise FileExistsError('This run already exists; use a different --output for another run.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda'):
        # Required by deterministic CUDA matrix multiplication on supported setups.
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # Guard the whole cache tree as well as the individual result directory.
    guard = SimpleNamespace(output=args.cache, max_cache_gb=2.0, min_free_gb=2.0)
    storage_guard(guard, 32 * 1024**2)
    cache_args = SimpleNamespace(output=args.cache)
    records = []
    for row in manifest['slides']:
        item = read_cache(cache_args, row, manifest)
        if item is None:
            raise FileNotFoundError(f'Incomplete feature cache for {row["image"]}; run prepare first')
        records.append((row, item))
    train_names = {r['image'] for r in train_rows}
    inner_names = {r['image'] for r in inner_train}
    inner_val_names = {r['image'] for r in inner_val}
    outer_names = {r['image'] for r in val_rows}
    if inner_names & inner_val_names or train_names & outer_names:
        raise ValueError('Slide leakage in split')
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'run_config.json', {
        'head': args.head, 'architecture': str(model), 'trainable_parameters': count,
        'epochs_maximum': args.epochs, 'patience': args.patience, 'batch_size': args.batch_size,
        'lr': args.lr, 'weight_decay': args.weight_decay, 'seed': args.seed,
        'device': args.device, 'hidden': args.hidden, 'dropout': args.dropout,
        'objective': 'same slide/class-weighted MSE coverage target as Ridge',
        'cache_manifest': manifest,
        'inner_train_slides': sorted(inner_names), 'inner_validation_slides': sorted(inner_val_names),
        'outer_validation_slides': sorted(outer_names),
    })
    _, _, history, selected_epoch = train_epochs(
        args, [r for r in records if r[0]['image'] in inner_names],
        [r for r in records if r[0]['image'] in inner_val_names])
    # Restart from a fixed random initialization and use all original training WSIs.
    trained, scaler, final_history, _ = train_epochs(
        args, [r for r in records if r[0]['image'] in train_names],
        epochs=selected_epoch, phase='refit')
    save_history(args.output, history + final_history)
    torch.save({'state_dict': {k: v.cpu() for k, v in trained.state_dict().items()},
                'head': args.head, 'hidden': args.hidden, 'dropout': args.dropout,
                'scaler_mean': torch.tensor(scaler.mean_), 'scaler_scale': torch.tensor(scaler.scale_),
                'selected_epoch': selected_epoch}, args.output / 'model.pt')
    # state_dict cpu() above does not move the actual model; keep device explicit.
    trained.to(args.device)
    _x, targets, weights = arrays([r for r in records if r[0]['image'] in train_names])
    evaluate_regressor(args, manifest, records, Predictor(trained, args.device, args.batch_size),
                       scaler, targets, weights, info={
                           'objective': f'weighted MSE to tumor fraction; frozen CONCH-Z + PyTorch {args.head}',
                           'head': args.head, 'trainable_parameters': count,
                           'selected_epochs': selected_epoch,
                           'epoch_selection': 'inner WSI split only; refit on all original training WSIs',
                       })
    ridge_path = args.cache / 'summary.json'
    if ridge_path.exists():
        ridge = json.loads(ridge_path.read_text())
        neural = json.loads((args.output / 'summary.json').read_text())
        atomic_json(args.output / 'ridge_comparison.json', {'ridge': ridge, args.head: neural})
    print('Saved model, epoch history, loss curves and held-out metrics:', args.output, flush=True)


if __name__ == '__main__':
    main()
