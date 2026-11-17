#!/usr/bin/env python3
"""Parallel CONCH-Z feature extraction for CLAM, then one CLAM train/test run.

Three worker processes each load CONCH-Z on one GPU. They receive distinct
slides from a shared FIFO queue, so a worker that finishes early claims the
next slide. Per-slide caches use the same metadata and atomic NPZ format as
clam_sb_baseline.py. Once all feature bags exist, the original baseline script
runs once; it sees the complete cache and skips feature extraction.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import traceback
from typing import Any


BASELINE = Path(__file__).with_name('clam_sb_baseline.py').resolve()


def import_baseline():
    sys.path.insert(0, str(BASELINE.parent))
    import clam_sb_baseline as clam
    return clam


def find_active_baseline() -> list[tuple[int, str]]:
    """Find an already-running original baseline process to avoid duplicate writes."""
    active = []
    proc_root = Path('/proc')
    for proc in proc_root.glob('[0-9]*'):
        try:
            pid = int(proc.name)
            if pid == os.getpid():
                continue
            exe = (proc / 'exe').resolve().name
            if 'python' not in exe.lower():
                continue
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            args = [part.decode(errors='replace') for part in argv if part]
            if str(BASELINE) in args:
                active.append((pid, ' '.join(args)))
        except (OSError, ValueError, PermissionError):
            continue
    return active


def worker(gpu: int, first_row: dict[str, Any] | None, task_queue: Any,
           status_queue: Any, data_root: str, model_dir: str, output_dir: str,
           config_sha: str, batch_size: int, min_free_gb: float,
           cpu_threads: int) -> None:
    """Keep one CONCH-Z model resident on this GPU and consume slides in order."""
    log_path = Path(output_dir) / f'feature_worker_gpu{gpu}.log'
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s GPU{gpu} %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    try:
        import cv2
        import numpy as np
        import torch
        from transformers import AutoModel

        torch.set_num_threads(cpu_threads)
        cv2.setNumThreads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this worker.')
        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')
        logging.info('Loading local CONCH-Z on %s', device)
        encoder = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        ).eval().float().to(device)
        if not hasattr(encoder, 'eval_transform') or not hasattr(encoder, 'encode_image'):
            raise AttributeError('Loaded model lacks CONCH-Z eval_transform/encode_image API.')
        logging.info('Model ready; batch_size=%d', batch_size)

        clam = import_baseline()
        row = first_row
        while row is not None:
            slide_name = Path(row['image']).stem
            status_queue.put(('started', gpu, slide_name, time.time()))
            started = time.perf_counter()
            try:
                out = Path(output_dir)
                cache = clam.cache_path(out, row)
                # A cache may have been completed before this worker claimed it
                # (for example after a manual resume). Validate it, then skip.
                if clam.read_feature_cache(cache, config_sha, row) is None:
                    clam.disk_guard(out, min_free_gb)
                    item = clam.encode_slide(
                        Path(data_root) / 'images' / row['image'], row,
                        encoder, device, batch_size,
                    )
                    metadata = {
                        'config_sha256': config_sha,
                        'image': row['image'], 'split': row['split'],
                        'label': row['label'], 'class_name': row['class_name'],
                    }
                    clam.safe_save_npz(
                        cache, features=item['features'], xy=item['xy'],
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                    )
                    count = len(item['features'])
                else:
                    count = -1
                elapsed = time.perf_counter() - started
                logging.info('Cached %s patches=%s elapsed=%.1fs',
                             slide_name, count if count >= 0 else 'already-present', elapsed)
                status_queue.put(('completed', gpu, slide_name, count, elapsed))
            except Exception as exc:
                logging.exception('Failed slide %s', slide_name)
                status_queue.put(('failed', gpu, slide_name, repr(exc)))

            row = task_queue.get()
        del encoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        logging.info('Worker finished.')
    except Exception as exc:
        logging.exception('Worker startup/fatal failure')
        status_queue.put(('fatal', gpu, repr(exc), traceback.format_exc()))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=Path.home() / 'data' / 'camelyon16_full')
    parser.add_argument('--split-manifest', type=Path,
                        default=Path.home() / 'wsi_patch_fraction_cache_v1' / 'manifest.json')
    parser.add_argument('--model', type=Path, default=None,
                        help='Local CONCH-Z model folder; defaults to the split manifest setting.')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'results' / 'clam_sb_fullslide')
    parser.add_argument('--gpus', default='0,1,2',
                        help='GPU IDs for concurrent feature workers. GPU 3 is deliberately excluded.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Per-GPU CONCH-Z batch size.')
    parser.add_argument('--min-free-gb', type=float, default=10.0,
                        help='Stop workers before free space drops below this reserve.')
    parser.add_argument('--cpu-threads-per-worker', type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data = args.data.expanduser().resolve()
    args.split_manifest = args.split_manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    if not args.output.is_relative_to(project_root):
        raise SystemExit(f'Output must remain inside {project_root}')
    args.output.mkdir(parents=True, exist_ok=True)
    lock_file = (args.output / '.parallel_run.lock').open('w')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit('Another parallel CLAM launcher is already using this output folder.') from exc
    if args.batch_size < 1 or args.cpu_threads_per_worker < 1:
        raise SystemExit('Batch size and CPU threads per worker must be positive.')
    if args.min_free_gb < 0:
        raise SystemExit('--min-free-gb cannot be negative.')
    try:
        gpus = [int(part.strip()) for part in args.gpus.split(',') if part.strip()]
    except ValueError as exc:
        raise SystemExit('--gpus must be comma-separated GPU IDs, e.g. 0,1,2') from exc
    if not gpus or len(gpus) != len(set(gpus)) or any(g < 0 or g > 2 for g in gpus):
        raise SystemExit('Use unique GPU IDs from 0,1,2; GPU 3 is reserved.')

    active = find_active_baseline()
    if active:
        details = '\n'.join(f'  PID {pid}: {cmd}' for pid, cmd in active)
        raise SystemExit(
            'The original single-GPU CLAM script is still running. Stop it after '
            'the current slide reports "cached ...", then rerun this launcher.\n' + details
        )

    clam = import_baseline()
    split_manifest_obj = json.loads(args.split_manifest.read_text())
    model_dir = (args.model.expanduser().resolve() if args.model
                 else Path(split_manifest_obj['config']['model_dir']).expanduser().resolve())
    reference = args.data / 'evaluation' / 'reference.csv'
    if not (model_dir / 'config.json').is_file():
        raise FileNotFoundError(model_dir / 'config.json')
    rows, split_summary = clam.make_rows(args.data, args.split_manifest)
    config = clam.make_config(args.data, model_dir, args.split_manifest, reference)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f'Existing output has a different configuration: {args.output}')
    if not config_path.exists():
        clam.atomic_json(config_path, config)
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    clam.atomic_json(args.output / 'split_summary.json', split_summary)

    missing = []
    for row in rows:
        if clam.read_feature_cache(clam.cache_path(args.output, row), config_sha, row) is None:
            missing.append(row)

    log_path = args.output / 'parallel_feature_cache.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info('Splits: %s', json.dumps(split_summary, sort_keys=True))
    logging.info('Missing feature bags: %d; valid existing bags will be reused.', len(missing))
    logging.info('Worker GPUs: %s; per-GPU batch size: %d', gpus, args.batch_size)
    if missing:
        context = mp.get_context('spawn')
        task_queue = context.Queue()
        status_queue = context.Queue()
        # Assign the first three missing slides explicitly so the initial
        # worker assignments are deterministic; thereafter workers claim the
        # next slide from the shared FIFO queue as soon as they become free.
        initial = [missing[i] if i < len(missing) else None for i in range(len(gpus))]
        for row in missing[len(gpus):]:
            task_queue.put(row)
        for _ in gpus:
            task_queue.put(None)
        processes = []
        for gpu, first_row in zip(gpus, initial):
            proc = context.Process(
                target=worker,
                args=(gpu, first_row, task_queue, status_queue,
                      str(args.data), str(model_dir), str(args.output), config_sha,
                      args.batch_size, args.min_free_gb, args.cpu_threads_per_worker),
                name=f'clam-feature-gpu{gpu}',
            )
            proc.start()
            processes.append(proc)

        failures = []
        finished = 0
        while any(proc.is_alive() for proc in processes):
            try:
                event = status_queue.get(timeout=2.0)
            except Exception:
                continue
            if event[0] == 'completed':
                _, gpu, slide, count, elapsed = event
                logging.info('Progress: GPU%s finished %s (%s patches, %.1fs)',
                             gpu, slide, count if count >= 0 else 'cached', elapsed)
                finished += 1
            elif event[0] == 'failed':
                failures.append(event)
                logging.error('GPU%s failed %s: %s', event[1], event[2], event[3])
            elif event[0] == 'fatal':
                failures.append(event)
                logging.error('GPU%s worker failed at startup: %s', event[1], event[2])

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                failures.append(('exit', proc.name, proc.exitcode))
        # A worker can finish between the loop condition and queue read; drain
        # its final messages after joining so a late failure is not lost.
        while True:
            try:
                event = status_queue.get_nowait()
            except queue.Empty:
                break
            if event[0] == 'completed':
                _, gpu, slide, count, elapsed = event
                logging.info('Progress: GPU%s finished %s (%s patches, %.1fs)',
                             gpu, slide, count if count >= 0 else 'cached', elapsed)
                finished += 1
            elif event[0] == 'failed':
                failures.append(event)
                logging.error('GPU%s failed %s: %s', event[1], event[2], event[3])
            elif event[0] == 'fatal':
                failures.append(event)
                logging.error('GPU%s worker failed at startup: %s', event[1], event[2])
        task_queue.close()
        status_queue.close()
        if failures:
            logging.error('Feature extraction had %d failure(s); training will not start. '
                          'Rerun this command to resume from completed slide caches.', len(failures))
            raise SystemExit(2)
        logging.info('Parallel feature extraction complete: %d slide bags completed this run.', finished)
    else:
        logging.info('All slide feature bags already exist; skipping worker startup.')

    logging.info('Starting the original CLAM training and one test evaluation on cuda:%d.', gpus[0])
    command = [
        sys.executable, str(BASELINE),
        '--data', str(args.data), '--split-manifest', str(args.split_manifest),
        '--output', str(args.output), '--device', f'cuda:{gpus[0]}',
        '--batch-size', str(args.batch_size), '--min-free-gb', str(args.min_free_gb),
    ]
    if args.model:
        command.extend(['--model', str(model_dir)])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
