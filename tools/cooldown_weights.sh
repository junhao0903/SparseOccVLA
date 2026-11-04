#!/usr/bin/env bash
set -euo pipefail

# Ask Linux to discard clean page-cache pages for selected model weights.
# This is per-file and does not change file contents or clear global caches.
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

if not hasattr(os, 'posix_fadvise') or not hasattr(os, 'POSIX_FADV_DONTNEED'):
    print('ERROR: this platform does not support POSIX_FADV_DONTNEED.', file=sys.stderr)
    sys.exit(2)

paths = sys.argv[1:]
missing = [path for path in paths if not os.path.isfile(path)]
if missing:
    for path in missing:
        print('ERROR: weight file not found: {}'.format(path), file=sys.stderr)
    sys.exit(2)

for path in paths:
    fd = os.open(path, os.O_RDONLY)
    try:
        # len=0 means from offset through end of file on Linux.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    print('Requested page-cache eviction: {} ({:.2f} GiB)'.format(
        path, os.path.getsize(path) / (1024 ** 3)), flush=True)

print('Request complete. Linux treats this as advice; pinned or active pages '
      'may remain cached.', flush=True)
PY
