#!/usr/bin/env bash
set -euo pipefail

# Warm the Linux page cache for the default Stage3 test weights, or for the
# explicit files supplied on the command line. This reads files only; it does
# not copy, rewrite, or pin them in RAM.
ASSET_ROOT="${SPARSEOCC_ASSET_ROOT:-/mnt/data/SparseOccVLA_assets}"

if (($#)); then
    WEIGHT_FILES=("$@")
else
    WEIGHT_FILES=(
        "$ASSET_ROOT/ckpts/stage3_4d_600q_forecasting.pth"
        "$ASSET_ROOT/InternVL2-2B/model.safetensors"
    )
fi

python3 - "${WEIGHT_FILES[@]}" <<'PY'
import os
import sys
import time

paths = sys.argv[1:]
missing = [path for path in paths if not os.path.isfile(path)]
if missing:
    for path in missing:
        print('ERROR: weight file not found: {}'.format(path), file=sys.stderr)
    sys.exit(2)

entries = [(path, os.path.getsize(path)) for path in paths]
total_bytes = sum(size for _, size in entries)
read_total = 0
started = time.monotonic()
chunk_size = 16 * 1024 * 1024

print('Prewarming {} weight file(s), total {:.2f} GiB'.format(
    len(entries), total_bytes / (1024 ** 3)), flush=True)

for path, expected_size in entries:
    file_started = time.monotonic()
    file_read = 0
    last_report = file_started
    print('\n[START] {} ({:.2f} GiB)'.format(
        path, expected_size / (1024 ** 3)), flush=True)

    with open(path, 'rb', buffering=0) as source:
        while True:
            block = source.read(chunk_size)
            if not block:
                break
            count = len(block)
            file_read += count
            read_total += count
            now = time.monotonic()
            if now - last_report >= 2 or file_read == expected_size:
                elapsed = now - file_started
                speed = file_read / max(elapsed, 1e-9) / (1024 ** 2)
                overall = 100.0 if total_bytes == 0 else 100.0 * read_total / total_bytes
                file_pct = 100.0 if expected_size == 0 else 100.0 * file_read / expected_size
                print('[{:.1f}% total] {:.1f}% file, {:.2f}/{:.2f} GiB, '
                      '{:.0f} MiB/s'.format(
                          overall, file_pct, file_read / (1024 ** 3),
                          expected_size / (1024 ** 3), speed), flush=True)
                last_report = now

    if file_read != expected_size:
        raise IOError('{} changed while being read (expected {}, read {})'.format(
            path, expected_size, file_read))
    print('[DONE] {} in {:.1f}s'.format(path, time.monotonic() - file_started),
          flush=True)

elapsed = time.monotonic() - started
print('\nFinished: read {:.2f} GiB in {:.1f}s ({:.0f} MiB/s average).'.format(
    read_total / (1024 ** 3), elapsed,
    read_total / max(elapsed, 1e-9) / (1024 ** 2)), flush=True)
print('The OS may evict these cached pages later if memory is needed.', flush=True)
PY
