#!/usr/bin/env python3
"""Plan and download a bounded CAMELYON16 subset from the public AWS mirror.

The default plan is 10 normal and 10 fully annotated tumor training slides,
their matching mask TIFFs, tumor XML files, and source metadata. Transfers are
sequential, resumable, checksummed, and limited to at most 100 decimal GB.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

BASE = "https://camelyon-dataset.s3.us-west-2.amazonaws.com/"
ROOT_PREFIX = "CAMELYON16/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
CHUNK = 4 * 1024 * 1024
MAX_BUDGET = 100_000_000_000
SPACE_BUFFER = 2_000_000_000
NON_EXHAUSTIVE_TUMOR_IDS = {
    10, 15, 18, 20, 25, 29, 33, 34, 44, 46, 51, 54, 55, 56,
    67, 79, 85, 92, 95, 110,
}


def get_bytes(key):
    with urlopen(BASE + quote(key, safe="/"), timeout=60) as response:
        return response.read()


def list_objects():
    objects = {}
    token = None
    while True:
        params = {"list-type": "2", "prefix": ROOT_PREFIX, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        with urlopen(BASE + "?" + urlencode(params), timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s:Contents", NS):
            key = item.findtext("s:Key", namespaces=NS)
            objects[key] = int(item.findtext("s:Size", namespaces=NS))
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            return objects


def parse_checksums(raw):
    checksums = {}
    for line in raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{32})\s+\*?(.+)", line.strip())
        if match:
            checksums[ROOT_PREFIX + match.group(2)] = match.group(1).lower()
    return checksums


def make_plan(budget):
    objects = list_objects()
    checksums = parse_checksums(get_bytes(ROOT_PREFIX + "checksums.md5"))
    selected = []
    for label in ("normal", "tumor"):
        eligible = []
        for key, size in objects.items():
            if not (key.startswith(ROOT_PREFIX + "images/" + label + "_")
                    and key.endswith(".tif")):
                continue
            stem = PurePosixPath(key).stem
            if label == "tumor" and int(stem.rsplit("_", 1)[1]) in NON_EXHAUSTIVE_TUMOR_IDS:
                continue
            mask = ROOT_PREFIX + "masks/" + stem + "_mask.tif"
            xml = ROOT_PREFIX + "annotations/" + stem + ".xml"
            needed = [key, mask] + ([xml] if label == "tumor" else [])
            if not all(part in objects and part in checksums for part in needed):
                continue
            eligible.append({
                "slide": stem, "label": label, "image_bytes": size,
                "keys": needed,
                "total_bytes": sum(objects[part] for part in needed),
            })
        eligible.sort(key=lambda item: (item["image_bytes"], item["slide"]))
        if len(eligible) < 10:
            raise RuntimeError(f"Only {len(eligible)} usable {label} slides found")
        selected.extend(eligible[:10])

    metadata = [ROOT_PREFIX + name for name in
                ("README.md", "license.txt", "checksums.md5")]
    keys = metadata[:]
    # Interleave classes so a normal/tumor pair is available early.
    for pair in zip(selected[:10], selected[10:]):
        for item in pair:
            # Small annotations and masks arrive before the large image.
            keys.extend(item["keys"][1:] + item["keys"][:1])
    total = sum(objects[key] for key in keys)
    if total > budget:
        raise RuntimeError(f"Selected {total / 1e9:.2f} GB exceeds {budget / 1e9:.2f} GB budget")
    files = [{"key": key, "size": objects[key], "md5": checksums.get(key)}
             for key in keys]
    return {
        "source": BASE, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "10 smallest eligible normal and 10 smallest fully annotated tumor training slides",
        "excluded_non_exhaustive_tumor_ids": sorted(NON_EXHAUSTIVE_TUMOR_IDS),
        "budget_bytes": budget, "planned_bytes": total,
        "selected_slides": selected, "files": files,
    }


def relative_path(key):
    path = PurePosixPath(key)
    if not key.startswith(ROOT_PREFIX) or ".." in path.parts:
        raise ValueError(f"Unexpected S3 key: {key}")
    return Path(*path.parts[1:])


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def is_complete(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    return item["md5"] is None or md5_file(path) == item["md5"]


def used_bytes(dest):
    return sum(path.stat().st_size for path in dest.rglob("*") if path.is_file())


def transfer(dest, item, budget, stop):
    target = dest / relative_path(item["key"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_complete(target, item):
        print("OK", item["key"], flush=True)
        return
    if target.exists():
        raise RuntimeError(f"Existing file is incomplete or has wrong checksum: {target}")
    part = target.with_name(target.name + ".part")
    expected = item["size"]
    if part.exists() and part.stat().st_size > expected:
        raise RuntimeError(f"Oversized partial file: {part}")
    for attempt in range(1, 6):
        if stop.exists():
            raise InterruptedError("STOP marker found")
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected:
            if item["md5"] and md5_file(part) != item["md5"]:
                raise RuntimeError(f"Completed partial has wrong MD5; inspect or remove {part}")
            part.replace(target)
            print("DONE", item["key"], f"{expected / 1e9:.3f} GB", flush=True)
            return
        if used_bytes(dest) + expected - offset > budget:
            raise RuntimeError("Download would exceed configured disk budget")
        if shutil.disk_usage(dest).free < expected - offset + SPACE_BUFFER:
            raise RuntimeError("Not enough free space with 2 GB reserve")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = Request(BASE + quote(item["key"], safe="/"), headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    raise RuntimeError("Server did not honor the resume byte range")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned an unexpected byte range")
                with part.open("ab" if offset else "wb") as handle:
                    bytes_since_budget_check = 0
                    while True:
                        if stop.exists():
                            raise InterruptedError("STOP marker found")
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        bytes_since_budget_check += len(block)
                        if handle.tell() > expected:
                            raise RuntimeError("Transfer exceeded the expected size or budget")
                        if bytes_since_budget_check >= 64 * 1024 * 1024:
                            if used_bytes(dest) > budget:
                                raise RuntimeError("Transfer exceeded the configured disk budget")
                            bytes_since_budget_check = 0
            if part.stat().st_size != expected:
                raise OSError(f"Incomplete transfer: {part.stat().st_size}/{expected}")
            if item["md5"] and md5_file(part) != item["md5"]:
                raise RuntimeError(f"MD5 mismatch: {part}")
            part.replace(target)
            print("DONE", item["key"], f"{expected / 1e9:.3f} GB", flush=True)
            return
        except InterruptedError:
            raise
        except Exception as error:
            print(f"Retry {attempt}/5 for {item['key']}: {error}", file=sys.stderr, flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))


def status(dest, plan, check_md5=False):
    complete = []
    partial = []
    for item in plan["files"]:
        target = dest / relative_path(item["key"])
        if (is_complete(target, item) if check_md5 else
                target.is_file() and target.stat().st_size == item["size"]):
            complete.append(item)
        elif target.with_name(target.name + ".part").exists():
            partial.append(item["key"])
    print(f"Planned: {len(plan['selected_slides'])} slides, "
          f"{plan['planned_bytes'] / 1e9:.3f} GB of {plan['budget_bytes'] / 1e9:.1f} GB cap")
    print(f"Size-complete files: {len(complete)}/{len(plan['files'])}; "
          f"partial files: {len(partial)}; on disk: {used_bytes(dest) / 1e9:.3f} GB")
    file_sizes = {item["key"]: item["size"] for item in plan["files"]}
    slides_done = {label: 0 for label in ("normal", "tumor")}
    for slide in plan["selected_slides"]:
        image_key = slide["keys"][0]
        image_path = dest / relative_path(image_key)
        if image_path.is_file() and image_path.stat().st_size == file_sizes[image_key]:
            slides_done[slide["label"]] += 1
    print(f"Slides complete: {sum(slides_done.values())}/20 "
          f"(normal {slides_done['normal']}/10, tumor {slides_done['tumor']}/10)")
    if partial:
        print("Partial:", ", ".join(partial))
    print("Stop requested:", (dest / "STOP").exists())
    with (dest / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            running = False
            fcntl.flock(lock, fcntl.LOCK_UN)
        except BlockingIOError:
            running = True
    print("Downloader running:", running)
    return len(complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True,
                        help="Dedicated output directory, outside the Git repository")
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("plan", "download", "status", "stop", "verify"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--max-gb", type=float, default=100.0,
                        help="Decimal GB cap; cannot exceed 100 (default: 100)")
    args = parser.parse_args()
    if not (0 < args.max_gb <= 100):
        parser.error("--max-gb must be between 0 and 100")
    budget = int(args.max_gb * 1e9)
    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    plan_path = dest / "manifest.json"
    stop = dest / "STOP"
    if args.stop:
        stop.touch()
        print(f"Stop requested via {stop}; partial file will remain resumable")
        return
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["planned_bytes"] > budget or plan["budget_bytes"] > budget:
            parser.error("Existing manifest exceeds requested --max-gb")
        budget = min(budget, plan["budget_bytes"])
    elif args.plan or args.download:
        plan = make_plan(budget)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    else:
        parser.error(f"No manifest at {plan_path}; run --plan first")
    if args.plan or args.status or args.verify:
        complete_count = status(dest, plan, check_md5=args.verify)
        if args.verify and complete_count != len(plan["files"]):
            raise SystemExit("Some files are absent or fail size/MD5 verification")
        return

    lock_path = dest / ".download.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another downloader is already running for this destination")
        stop.unlink(missing_ok=True)  # An explicit --download resumes a stopped run.
        try:
            for item in plan["files"]:
                if stop.exists():
                    print("Stopped; run --download again to resume", flush=True)
                    return
                transfer(dest, item, budget, stop)
        except (InterruptedError, KeyboardInterrupt):
            print("Stopped safely; run --download again to resume", flush=True)
            return
    status(dest, plan)


if __name__ == "__main__":
    main()
