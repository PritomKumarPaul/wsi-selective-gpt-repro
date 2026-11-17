#!/usr/bin/env python3
"""Download extra annotated CAMELYON16 train slides and official test slides.

Default fixed sample: 10 new normal + 10 new tumor training slides; 10 normal
+ 10 tumor official test slides (five micro, five macro). All have matching
masks, and positive slides have XML annotations. The first 20 slides are
excluded. The bundled manifest pins source sizes and checksums.

--start launches a detached worker on the server. --stop is graceful. On the
next --start, incomplete .part files are identified, deleted, and downloaded
again from byte zero; completed files with valid MD5s are kept. No big file
is downloaded by --plan or --status.
"""

import argparse
import csv
import fcntl
import io
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

from download_camelyon16_subset import (
    BASE, ROOT_PREFIX, NON_EXHAUSTIVE_TUMOR_IDS, get_bytes,
    list_objects, parse_checksums, relative_path, md5_file,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DEST = Path.home() / 'data' / 'camelyon16_expansion'
DEFAULT_EXISTING = Path.home() / 'data' / 'camelyon16_subset' / 'manifest.json'
BUNDLED_MANIFEST = HERE / 'examples/camelyon16_expansion_manifest.json'
MAX_NEW_GB = 75.0
SPACE_BUFFER = 2_000_000_000
CHUNK = 4 * 1024 * 1024
META_KEYS = [ROOT_PREFIX + name for name in
             ('README.md', 'license.txt', 'checksums.md5', 'evaluation/reference.csv')]


def make_plan(args):
    objects = list_objects()
    checksums = parse_checksums(get_bytes(ROOT_PREFIX + 'checksums.md5'))
    reference_bytes = get_bytes(ROOT_PREFIX + 'evaluation/reference.csv')
    reference = {PurePosixPath(row['image']).stem: row for row in
                 csv.DictReader(io.StringIO(reference_bytes.decode('utf-8-sig')))}
    if not args.existing_manifest.is_file():
        raise FileNotFoundError(f'Existing 20-slide manifest is needed to exclude repeats: '
                                f'{args.existing_manifest}')
    old = json.loads(args.existing_manifest.read_text())
    excluded = {item['slide'] for item in old['selected_slides']}

    def eligible(group):
        candidates = []
        for key, size in objects.items():
            if not key.startswith(ROOT_PREFIX + 'images/') or not key.endswith('.tif'):
                continue
            stem = PurePosixPath(key).stem
            if stem in excluded:
                continue
            row = reference.get(stem)
            if group == 'train_normal' and not stem.startswith('normal_'):
                continue
            if group == 'train_tumor':
                if not stem.startswith('tumor_'):
                    continue
                if int(stem.rsplit('_', 1)[1]) in NON_EXHAUSTIVE_TUMOR_IDS:
                    continue
            if group.startswith('test_'):
                if not stem.startswith('test_') or row is None:
                    continue
                if group == 'test_normal' and row['type'] != 'normal':
                    continue
                if group == 'test_micro' and row['class'] != 'micro':
                    continue
                if group == 'test_macro' and row['class'] != 'macro':
                    continue
            mask = ROOT_PREFIX + 'masks/' + stem + '_mask.tif'
            positive = group in ('train_tumor', 'test_micro', 'test_macro')
            xml = ROOT_PREFIX + 'annotations/' + stem + '.xml'
            needed = [key, mask] + ([xml] if positive else [])
            if not all(part in objects and part in checksums for part in needed):
                continue
            candidates.append({
                'slide': stem, 'split': 'test_official' if group.startswith('test_') else 'train_extra',
                'label': 'tumor' if positive else 'normal',
                'subtype': row['class'] if row else None,
                'center': row['center'] if row else None,
                'image_bytes': size, 'total_bytes': sum(objects[p] for p in needed),
                'keys': needed,
            })
        return sorted(candidates, key=lambda x: (x['total_bytes'], x['slide']))

    groups = {
        'train_normal': eligible('train_normal')[:args.train_normal],
        'train_tumor': eligible('train_tumor')[:args.train_tumor],
        'test_normal': eligible('test_normal')[:args.test_normal],
        'test_micro': eligible('test_micro')[:(args.test_tumor + 1) // 2],
        'test_macro': eligible('test_macro')[:args.test_tumor // 2],
    }
    requested = {'train_normal': args.train_normal, 'train_tumor': args.train_tumor,
                 'test_normal': args.test_normal,
                 'test_micro': (args.test_tumor + 1) // 2,
                 'test_macro': args.test_tumor // 2}
    for group, chosen in groups.items():
        if len(chosen) != requested[group]:
            raise RuntimeError(f'Only {len(chosen)} eligible slides for {group}; '
                               f'{requested[group]} requested')
    # Interleave labels/splits so a balanced partial dataset arrives early.
    test_tumors = sorted(groups['test_micro'] + groups['test_macro'],
                         key=lambda x: (x['total_bytes'], x['slide']))
    ordered = []
    for batch in itertools.zip_longest(groups['train_normal'], groups['train_tumor'],
                                       groups['test_normal'], test_tumors):
        ordered.extend(x for x in batch if x is not None)
    keys = META_KEYS[:]
    for item in ordered:
        keys.extend(item['keys'][1:] + item['keys'][:1])  # mask/XML before large image
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate file in plan')
    total = sum(objects[key] for key in keys)
    if total > int(args.max_gb * 1e9):
        raise RuntimeError(f'Plan requires {total/1e9:.3f} GB, beyond the '
                           f'{args.max_gb:.1f} GB new-download cap')
    return {
        'format_version': 1, 'source': BASE,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'selection': 'smallest fully annotated extra train slides; smallest official test '
                     'slides balanced by normal/tumor and micro/macro',
        'excluded_existing_slides': sorted(excluded),
        'excluded_non_exhaustive_tumor_ids': sorted(NON_EXHAUSTIVE_TUMOR_IDS),
        'counts': {'train_normal': args.train_normal, 'train_tumor': args.train_tumor,
                   'test_normal': args.test_normal, 'test_tumor': args.test_tumor},
        'budget_bytes': int(args.max_gb * 1e9), 'planned_bytes': total,
        'selected_slides': ordered,
        'files': [{'key': key, 'size': objects[key], 'md5': checksums.get(key)} for key in keys],
    }


def validate_plan(plan, cap):
    if plan.get('format_version') != 1 or plan.get('source') != BASE:
        raise ValueError('Unknown manifest format or source')
    files = plan['files']
    keys = [item['key'] for item in files]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate manifest key')
    for item in files:
        relative_path(item['key'])
        if not isinstance(item['size'], int) or item['size'] < 0:
            raise ValueError(f'Invalid size for {item["key"]}')
        if item['key'] not in META_KEYS and (not isinstance(item['md5'], str)
                                             or len(item['md5']) != 32):
            raise ValueError(f'Missing MD5 for {item["key"]}')
    if sum(item['size'] for item in files) != plan['planned_bytes']:
        raise ValueError('Manifest size total mismatch')
    if plan['planned_bytes'] > cap or plan['budget_bytes'] > cap:
        raise ValueError('Manifest exceeds requested cap')
    if len({s['slide'] for s in plan['selected_slides']}) != len(plan['selected_slides']):
        raise ValueError('Duplicate slide in manifest')
    if set(s['slide'] for s in plan['selected_slides']) & set(plan['excluded_existing_slides']):
        raise ValueError('Manifest repeats an existing slide')
    if any(not set(slide['keys']).issubset(keys) for slide in plan['selected_slides']):
        raise ValueError('Manifest has a slide with missing files')
    return plan


def manifest_for(dest, cap, create=False):
    path = dest / 'manifest.json'
    if not path.exists() and create:
        if any(dest.iterdir()):
            raise RuntimeError(f'Destination has files but no manifest: {dest}')
        if not BUNDLED_MANIFEST.is_file():
            raise FileNotFoundError(f'Bundled manifest missing: {BUNDLED_MANIFEST}; run --plan')
        shutil.copyfile(BUNDLED_MANIFEST, path)
    if not path.is_file():
        raise FileNotFoundError(f'No manifest at {path}; run --plan first')
    return validate_plan(json.loads(path.read_text()), cap)


def bytes_used(dest):
    return sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())


def lock_is_held(dest):
    with (dest / '.download.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


def file_valid(path, item, md5=True):
    return (path.is_file() and path.stat().st_size == item['size']
            and (not md5 or not item['md5'] or md5_file(path) == item['md5']))


def report_status(dest, plan, verify=False):
    complete, partial = set(), []
    for item in plan['files']:
        path = dest / relative_path(item['key'])
        if file_valid(path, item, md5=verify):
            complete.add(item['key'])
        elif path.with_name(path.name + '.part').is_file():
            part = path.with_name(path.name + '.part')
            partial.append((item['key'], part.stat().st_size, item['size']))
    slides = Counter((s['split'], s['label']) for s in plan['selected_slides']
                     if set(s['keys']).issubset(complete))
    print(f"Plan: {len(plan['selected_slides'])} slides, {plan['planned_bytes']/1e9:.3f} GB "
          f"under {plan['budget_bytes']/1e9:.1f} GB new-data cap")
    print(f"Files: {len(complete)}/{len(plan['files'])} complete"
          f"{' (MD5 checked)' if verify else ' (size checked)'}; "
          f"on disk {bytes_used(dest)/1e9:.3f} GB")
    print('Complete slide sets:',
          ', '.join(f'{split}/{label}={slides[(split,label)]}' for split in
                    ('train_extra', 'test_official') for label in ('normal', 'tumor')))
    if partial:
        for key, current, expected in partial:
            print(f'PARTIAL {key}: {current/1e9:.3f}/{expected/1e9:.3f} GB '
                  '(deleted and restarted on next --start)')
    print('Running:', lock_is_held(dest), 'Stop requested:', (dest / 'STOP').exists())
    return len(complete) == len(plan['files'])


def cleanup_partials(dest, plan):
    for item in plan['files']:
        path = dest / relative_path(item['key'])
        part = path.with_name(path.name + '.part')
        if part.exists():
            print(f'DISCARD partial {part} ({part.stat().st_size:,} bytes)', flush=True)
            part.unlink()


def download_one(dest, item, cap, stop):
    path = dest / relative_path(item['key'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_valid(path, item):
        print('OK', item['key'], flush=True)
        return
    if path.exists():
        print('DISCARD invalid complete file', path, flush=True)
        path.unlink()
    part = path.with_name(path.name + '.part')
    for attempt in range(1, 4):
        if stop.exists():
            raise InterruptedError('Stop requested')
        if part.exists():
            print('DISCARD incomplete retry', part, flush=True)
            part.unlink()
        if bytes_used(dest) + item['size'] > cap:
            raise RuntimeError('Destination would exceed the configured data cap')
        if shutil.disk_usage(dest).free < item['size'] + SPACE_BUFFER:
            raise RuntimeError('Less than file size plus 2 GB free-space reserve')
        try:
            request = Request(BASE + quote(item['key'], safe='/'))
            with urlopen(request, timeout=90) as response, part.open('xb') as output:
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != item['size']:
                    raise RuntimeError('Source Content-Length differs from pinned manifest')
                since_check = 0
                while True:
                    if stop.exists():
                        raise InterruptedError('Stop requested; partial retained until next --start')
                    block = response.read(CHUNK)
                    if not block:
                        break
                    output.write(block)
                    since_check += len(block)
                    if output.tell() > item['size']:
                        raise RuntimeError('Transfer exceeded pinned file size')
                    if since_check >= 64 * 1024 * 1024:
                        if bytes_used(dest) > cap:
                            raise RuntimeError('Destination exceeded data cap')
                        if shutil.disk_usage(dest).free < SPACE_BUFFER:
                            raise RuntimeError('Free space fell below 2 GB reserve')
                        since_check = 0
            if not file_valid(part, item):
                raise RuntimeError('Downloaded file failed size or MD5 verification')
            part.replace(path)
            print('DONE', item['key'], f"{item['size']/1e9:.3f} GB", flush=True)
            return
        except InterruptedError:
            raise
        except Exception as error:
            part.unlink(missing_ok=True)  # Each retry starts from byte zero.
            print(f'RETRY {attempt}/3 {item["key"]}: {error}', flush=True)
            if attempt == 3:
                raise
            for _ in range(min(2 ** attempt, 10)):
                if stop.exists():
                    raise InterruptedError('Stop requested')
                time.sleep(1)


def worker(dest, cap):
    plan = manifest_for(dest, cap)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (dest / '.download.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another downloader already holds the destination lock')
        (dest / 'download.pid').write_text(str(os.getpid()) + '\n')
        stop = dest / 'STOP'
        try:
            print('START', datetime.now(timezone.utc).isoformat(), flush=True)
            cleanup_partials(dest, plan)
            for item in plan['files']:
                if stop.exists():
                    raise InterruptedError('Stop requested')
                download_one(dest, item, cap, stop)
            print('COMPLETE all files downloaded and MD5 checked', flush=True)
        except InterruptedError as error:
            print('STOPPED', error, '; restart with --start', flush=True)
        finally:
            (dest / 'download.pid').unlink(missing_ok=True)
            print('END', datetime.now(timezone.utc).isoformat(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ('plan', 'start', 'status', 'stop', 'verify', 'worker'):
        actions.add_argument('--' + name, action='store_true',
                             help=argparse.SUPPRESS if name == 'worker' else None)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--existing-manifest', type=Path, default=DEFAULT_EXISTING,
                        help='Existing 20-slide manifest to exclude when creating a new plan')
    parser.add_argument('--max-gb', type=float, default=55.0,
                        help='Cap for this NEW destination, decimal GB (default 55; maximum 75)')
    parser.add_argument('--train-normal', type=int, default=10)
    parser.add_argument('--train-tumor', type=int, default=10)
    parser.add_argument('--test-normal', type=int, default=10)
    parser.add_argument('--test-tumor', type=int, default=10)
    args = parser.parse_args()
    if not (0 < args.max_gb <= MAX_NEW_GB):
        parser.error(f'--max-gb must be between 0 and {MAX_NEW_GB:g}')
    if any(x < 0 for x in (args.train_normal, args.train_tumor,
                           args.test_normal, args.test_tumor)):
        parser.error('Slide counts cannot be negative')
    cap = int(args.max_gb * 1e9)
    dest = args.dest.expanduser().resolve()
    if args.plan:
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / 'manifest.json'
        if path.exists():
            raise SystemExit(f'Manifest already exists: {path}; use another destination')
        if any(dest.iterdir()):
            raise SystemExit(f'Destination is not empty: {dest}')
        plan = make_plan(args)
        path.write_text(json.dumps(plan, indent=2) + '\n')
        report_status(dest, plan)
        print('Only metadata was read; no slide or mask downloaded.')
        return
    if args.stop:
        if not dest.is_dir():
            raise SystemExit(f'No destination exists: {dest}')
        (dest / 'STOP').touch()
        print(f'Stop requested. Worker will finish the current read and exit; '
              f'partial file will be deleted on the next --start.')
        return
    if args.start:
        dest.mkdir(parents=True, exist_ok=True)
        plan = manifest_for(dest, cap, create=True)
        if lock_is_held(dest):
            print('Downloader is already running; use --status')
            return
        (dest / 'STOP').unlink(missing_ok=True)  # Explicit --start resumes a stopped run.
        log_path = dest / 'download.log'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--dest', str(dest), '--max-gb', str(args.max_gb)]
        with log_path.open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        print(f'Started detached server worker PID {process.pid}.')
        print(f'Plan: {len(plan["selected_slides"])} slides, '
              f'{plan["planned_bytes"]/1e9:.3f} GB. Log: {log_path}')
        print('You may disconnect. Check --status or tail -f the log; use --stop to pause.')
        return
    if not dest.is_dir():
        raise SystemExit(f'No destination exists: {dest}')
    plan = manifest_for(dest, cap)
    if args.worker:
        worker(dest, cap)
    else:
        complete = report_status(dest, plan, verify=args.verify)
        if args.verify and not complete:
            raise SystemExit('Some files are missing or fail MD5; use --start')


if __name__ == '__main__':
    main()
