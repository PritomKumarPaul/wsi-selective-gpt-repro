#!/usr/bin/env python3
"""Download the complete public CAMELYON16 mirror into a bounded /net folder.

The bundled manifest pins every object's size and available source checksum.
--start launches a detached server worker; --stop is graceful. On restart,
incomplete .part files are deleted and fetched again from byte zero. Valid
completed files are kept. No slide is downloaded while making the plan.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from download_camelyon16_subset import (
    BASE, NS, ROOT_PREFIX, get_bytes, md5_file, parse_checksums, relative_path,
)
from download_camelyon16_expansion import (
    SPACE_BUFFER, cleanup_partials, file_valid,
    lock_is_held,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DEST = Path.home() / 'data' / 'camelyon16_full'
BUNDLED_MANIFEST = HERE / 'examples/camelyon16_full_manifest.json'
SEED_DIRS = (
    Path.home() / 'data' / 'camelyon16_subset',
    Path.home() / 'data' / 'camelyon16_expansion',
)
CAP_BYTES = 900_000_000_000  # 900 decimal GB; the mirror currently needs ~762 GB.
HASH_JOURNAL = '.verified_files.json'
HEX32 = re.compile(r'^[0-9a-fA-F]{32}$')


def bytes_used(dest):
    """Count destination bytes while other transfers atomically rename files."""
    total = 0
    for path in dest.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # A .part or journal temp file may have been renamed meanwhile.
            pass
    return total


def list_catalog():
    """Read S3 XML listings only; no slide content is transferred."""
    result = {}
    token = None
    while True:
        query = {'list-type': '2', 'prefix': ROOT_PREFIX, 'max-keys': '1000'}
        if token:
            query['continuation-token'] = token
        with urlopen(BASE + '?' + urlencode(query), timeout=60) as response:
            root = ET.fromstring(response.read())
        for element in root.findall('s:Contents', NS):
            key = element.findtext('s:Key', namespaces=NS)
            if key.endswith('/'):  # S3 directory marker, not a file.
                continue
            result[key] = {
                'size': int(element.findtext('s:Size', namespaces=NS)),
                'etag': (element.findtext('s:ETag', namespaces=NS) or '').strip('"'),
            }
        token = root.findtext('s:NextContinuationToken', namespaces=NS)
        if not token:
            return result


def ordered_keys(catalog):
    """Put small metadata first, then each slide's related files and image."""
    images = sorted(k for k in catalog if k.startswith(ROOT_PREFIX + 'images/'))
    remaining = set(catalog)
    order = []

    def add(key):
        if key in remaining:
            order.append(key)
            remaining.remove(key)

    for key in sorted(remaining):
        if '/' not in key[len(ROOT_PREFIX):]:
            add(key)
    for key in sorted(remaining):
        if key.startswith(ROOT_PREFIX + 'evaluation/'):
            add(key)
    for image in images:
        stem = PurePosixPath(image).stem
        add(ROOT_PREFIX + 'annotations/' + stem + '.xml')
        add(ROOT_PREFIX + 'background_tissue/' + stem + '_tissue.tif')
        add(ROOT_PREFIX + 'masks/' + stem + '_mask.tif')
        add(image)
    for key in sorted(remaining):
        add(key)
    return order


def make_plan():
    catalog = list_catalog()
    published = parse_checksums(get_bytes(ROOT_PREFIX + 'checksums.md5'))
    files = []
    for key in ordered_keys(catalog):
        relative_path(key)
        obj = catalog[key]
        etag = obj['etag']
        # A single-part S3 ETag is an MD5; a multipart ETag contains a dash.
        md5 = published.get(key)
        method = 'published_md5' if md5 else 'single_part_etag_md5'
        if not md5 and HEX32.fullmatch(etag):
            md5 = etag.lower()
        if not md5:
            method = 'size_and_local_sha256_only'
        files.append({'key': key, 'size': obj['size'], 'md5': md5,
                      'etag': etag, 'verification': method})
    total = sum(item['size'] for item in files)
    if total > CAP_BYTES:
        raise RuntimeError(f'Public mirror now needs {total/1e9:.3f} GB; '
                           f'exceeds the fixed {CAP_BYTES/1e9:.0f} GB cap')
    counts = Counter(k[len(ROOT_PREFIX):].split('/')[0] for k in catalog)
    return {'format_version': 1, 'source': BASE,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'cap_bytes': CAP_BYTES, 'planned_bytes': total,
            'object_counts': dict(counts), 'files': files}


def validate_plan(plan):
    if plan.get('format_version') != 1 or plan.get('source') != BASE:
        raise ValueError('Unknown manifest format or source')
    files = plan.get('files', [])
    if not files or len({f['key'] for f in files}) != len(files):
        raise ValueError('Empty plan or duplicate keys')
    for item in files:
        relative_path(item['key'])
        if not isinstance(item['size'], int) or item['size'] < 0:
            raise ValueError(f'Invalid size: {item["key"]}')
        if item['md5'] is not None and not HEX32.fullmatch(item['md5']):
            raise ValueError(f'Invalid MD5: {item["key"]}')
        if item['md5'] is None and item['verification'] != 'size_and_local_sha256_only':
            raise ValueError(f'Missing verification method: {item["key"]}')
    if sum(f['size'] for f in files) != plan['planned_bytes']:
        raise ValueError('Manifest size total mismatch')
    if plan['planned_bytes'] > CAP_BYTES or plan['cap_bytes'] != CAP_BYTES:
        raise ValueError('Manifest exceeds fixed 900 GB cap')
    return plan


def manifest_for(dest, create=False):
    path = dest / 'manifest.json'
    if create and not path.exists():
        if any(dest.iterdir()):
            raise RuntimeError(f'Destination has files but no manifest: {dest}')
        if not BUNDLED_MANIFEST.is_file():
            raise FileNotFoundError(f'Bundled manifest missing: {BUNDLED_MANIFEST}')
        shutil.copyfile(BUNDLED_MANIFEST, path)
    if not path.is_file():
        raise FileNotFoundError(f'No manifest: {path}; use --start or --plan')
    return validate_plan(json.loads(path.read_text()))


def load_journal(dest):
    path = dest / HASH_JOURNAL
    return json.loads(path.read_text()) if path.exists() else {}


def save_journal(dest, journal):
    path = dest / HASH_JOURNAL
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(journal, separators=(',', ':')) + '\n')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def journal_entry(path, item, local_sha256=None):
    stat = path.stat()
    entry = {'size': item['size'], 'mtime_ns': stat.st_mtime_ns}
    if local_sha256:
        entry['local_sha256'] = local_sha256
    return entry


def already_verified(path, item, journal):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    old = journal.get(item['key'])
    if old and old.get('size') == item['size'] and old.get('mtime_ns') == path.stat().st_mtime_ns:
        return True  # Its content was checked when this exact file was completed.
    if item['md5']:
        return md5_file(path) == item['md5']
    # No source checksum exists for two small multipart tissue files. Keep a
    # local fingerprint for detecting changes after their first transfer.
    return True


def seed_existing(dest, item):
    target = dest / relative_path(item['key'])
    if target.exists():
        return False
    for source_dir in SEED_DIRS:
        source = source_dir / relative_path(item['key'])
        if not file_valid(source, item):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if bytes_used(dest) + item['size'] > CAP_BYTES:
            raise RuntimeError('Target would exceed 900 GB cap while seeding')
        try:
            os.link(source, target)  # Same /net filesystem: no extra copy.
            mode = 'hardlink'
        except OSError:
            if shutil.disk_usage(dest).free < item['size'] + SPACE_BUFFER:
                raise RuntimeError('Not enough free space to copy existing file')
            partial = target.with_name(target.name + '.part')
            shutil.copyfile(source, partial)
            if not file_valid(partial, item):
                partial.unlink(missing_ok=True)
                raise RuntimeError(f'Copied source failed verification: {source}')
            partial.replace(target)
            mode = 'copy'
        print('REUSED', mode, item['key'], flush=True)
        return True
    return False


def transfer(dest, item, stop):
    """Stream one object with an in-flight checksum; retry from byte zero."""
    target = dest / relative_path(item['key'])
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    for attempt in range(1, 4):
        if stop.exists():
            raise InterruptedError('Stop requested')
        part.unlink(missing_ok=True)
        used_before = bytes_used(dest)
        if used_before + item['size'] > CAP_BYTES:
            raise RuntimeError('Destination would exceed the fixed 900 GB cap')
        if shutil.disk_usage(dest).free < item['size'] + SPACE_BUFFER:
            raise RuntimeError('Less than file size plus 2 GB free-space reserve')
        try:
            request = Request(BASE + quote(item['key'], safe='/'))
            md5 = hashlib.md5()
            sha = hashlib.sha256() if not item['md5'] else None
            written = 0
            since_space_check = 0
            with urlopen(request, timeout=90) as response, part.open('xb') as output:
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != item['size']:
                    raise RuntimeError('Source Content-Length differs from manifest')
                while True:
                    if stop.exists():
                        raise InterruptedError('Stop requested; partial retained until next --start')
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    md5.update(block)
                    if sha:
                        sha.update(block)
                    written += len(block)
                    since_space_check += len(block)
                    if written > item['size'] or used_before + written > CAP_BYTES:
                        raise RuntimeError('Transfer exceeded pinned size or 900 GB cap')
                    if since_space_check >= 256 * 1024 * 1024:
                        if shutil.disk_usage(dest).free < SPACE_BUFFER:
                            raise RuntimeError('Free space fell below 2 GB reserve')
                        since_space_check = 0
            if written != item['size']:
                raise RuntimeError('Incomplete transfer')
            if item['md5'] and md5.hexdigest() != item['md5']:
                raise RuntimeError('Downloaded MD5 differs from source checksum')
            part.replace(target)
            print('DONE', item['key'], f"{written/1e9:.3f} GB", flush=True)
            return sha.hexdigest() if sha else None
        except InterruptedError:
            raise
        except Exception as error:
            part.unlink(missing_ok=True)
            print(f'RETRY {attempt}/3 {item["key"]}: {error}', flush=True)
            if attempt == 3:
                raise
            for _ in range(min(2 ** attempt, 10)):
                if stop.exists():
                    raise InterruptedError('Stop requested')
                time.sleep(1)


def finish_one(dest, item, stop):
    """Process one independent file; the coordinator owns journal writes."""
    if stop.exists():
        raise InterruptedError('Stop requested')
    path = dest / relative_path(item['key'])
    if path.exists():
        if file_valid(path, item):
            return sha256_file(path) if not item['md5'] else None
        print('DISCARD invalid complete file', path, flush=True)
        path.unlink()
    if seed_existing(dest, item):
        return sha256_file(path) if not item['md5'] else None
    return transfer(dest, item, stop)


def status(dest, plan, verify=False):
    journal = load_journal(dest)
    complete = Counter()
    partials = []
    failed = []
    for item in plan['files']:
        path = dest / relative_path(item['key'])
        valid = file_valid(path, item) if verify else path.is_file() and path.stat().st_size == item['size']
        if verify and valid and not item['md5']:
            saved = journal.get(item['key'], {}).get('local_sha256')
            valid = bool(saved) and sha256_file(path) == saved
        if valid:
            category = item['key'][len(ROOT_PREFIX):].split('/')[0]
            complete[category] += 1
        elif path.exists():
            failed.append(item['key'])
        part = path.with_name(path.name + '.part')
        if part.is_file():
            partials.append((item['key'], part.stat().st_size))
    print(f"Plan: {len(plan['files'])} files, {plan['planned_bytes']/1e9:.3f} GB; "
          f"hard cap {CAP_BYTES/1e9:.0f} GB")
    print(f"Complete: {sum(complete.values())}/{len(plan['files'])} "
          f"({'source/hash checked' if verify else 'size checked'}); "
          f"on disk {bytes_used(dest)/1e9:.3f} GB")
    print('Images:', complete['images'], '/ 399; masks:', complete['masks'], '/ 399')
    print('Running:', lock_is_held(dest), 'Stop requested:', (dest / 'STOP').exists())
    for key, size in partials:
        print(f'PARTIAL {key}: {size/1e6:.1f} MB; deleted on next --start')
    for key in failed[:10]:
        print('INVALID', key)
    return sum(complete.values()) == len(plan['files'])


def worker(dest, workers=1):
    plan = manifest_for(dest)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (dest / '.download.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another full downloader already holds the lock')
        (dest / 'download.pid').write_text(str(os.getpid()) + '\n')
        stop = dest / 'STOP'
        try:
            print('START', datetime.now(timezone.utc).isoformat(),
                  f'parallel_files={workers}', flush=True)
            cleanup_partials(dest, plan)
            journal = load_journal(dest)
            pending = []
            for item in plan['files']:
                if stop.exists():
                    raise InterruptedError('Stop requested')
                path = dest / relative_path(item['key'])
                if already_verified(path, item, journal):
                    if item['key'] not in journal or journal[item['key']].get('mtime_ns') != path.stat().st_mtime_ns:
                        journal[item['key']] = journal_entry(
                            path, item, sha256_file(path) if not item['md5'] else None)
                        save_journal(dest, journal)
                    continue
                pending.append(item)
            # Reserve the entire remaining plan before concurrent work starts.
            # Its pinned total is below the cap, so in-flight transfers cannot
            # collectively push the destination past that cap.
            if bytes_used(dest) + sum(item['size'] for item in pending) > CAP_BYTES:
                raise RuntimeError('Existing and pending files exceed the 900 GB cap')
            print(f'PENDING {len(pending)} files; {workers} concurrent transfer(s)', flush=True)
            queue = iter(pending)
            completed = len(plan['files']) - len(pending)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                active = {}

                def submit_next():
                    if stop.exists():
                        return
                    item = next(queue, None)
                    if item is not None:
                        active[pool.submit(finish_one, dest, item, stop)] = item

                for _ in range(workers):
                    submit_next()
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = active.pop(future)
                        try:
                            local_sha = future.result()
                        except Exception:
                            stop.touch()  # Ask other in-flight transfers to exit.
                            raise
                        path = dest / relative_path(item['key'])
                        journal[item['key']] = journal_entry(path, item, local_sha)
                        save_journal(dest, journal)
                        completed += 1
                        if completed % 25 == 0:
                            print(f'PROGRESS {completed}/{len(plan["files"])} files', flush=True)
                        submit_next()
            if stop.exists():
                raise InterruptedError('Stop requested')
            print('COMPLETE all planned CAMELYON16 files', flush=True)
        except InterruptedError as error:
            print('STOPPED', error, '; resume with --start', flush=True)
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
    parser.add_argument('--workers', type=int, default=1,
                        help='Concurrent files for --start (1-8; try 4 after stopping the old worker)')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    dest = args.dest.expanduser().resolve()
    if args.plan:
        dest.mkdir(parents=True, exist_ok=True)
        if any(dest.iterdir()):
            raise SystemExit(f'Plan destination must be empty: {dest}')
        plan = validate_plan(make_plan())
        (dest / 'manifest.json').write_text(json.dumps(plan, indent=2) + '\n')
        print(f"Planned {len(plan['files'])} files, {plan['planned_bytes']/1e9:.3f} GB; "
              'no slides downloaded')
        return
    if args.stop:
        if not dest.is_dir():
            raise SystemExit(f'No destination: {dest}')
        (dest / 'STOP').touch()
        print('Stop requested. Any incomplete .part is deleted on next --start.')
        return
    if args.start:
        dest.mkdir(parents=True, exist_ok=True)
        plan = manifest_for(dest, create=True)
        if lock_is_held(dest):
            print('Downloader is already running; use --status')
            return
        (dest / 'STOP').unlink(missing_ok=True)
        log_path = dest / 'download.log'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--dest', str(dest),
                   '--workers', str(args.workers)]
        with log_path.open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        print(f'Started detached server worker PID {process.pid}.')
        print(f"Plan: {len(plan['files'])} files, {plan['planned_bytes']/1e9:.3f} GB, "
              f'{args.workers} concurrent transfer(s). '
              f'Log: {log_path}')
        return
    if not dest.is_dir():
        raise SystemExit(f'No destination: {dest}')
    plan = manifest_for(dest)
    if args.worker:
        worker(dest, args.workers)
    else:
        complete = status(dest, plan, verify=args.verify)
        if args.verify and not complete:
            raise SystemExit('Some files are incomplete or invalid; use --start')


if __name__ == '__main__':
    main()
