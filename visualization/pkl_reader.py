"""Reader for SparseOccVLA `tools/test.py --out` result files.

The result file is *not* a standard pickle.  It is a stream of independently
compressed, checksummed pickle records (see ``IncrementalVLAResultWriter`` in
``tools/test.py``)::

    MAGIC = b'SPARSEVLA-PKL-1\\n'
    then, repeated:
        struct('>Q32s')  ->  (8-byte big-endian payload size, 32-byte sha256)
        gzip( pickle(record) )

Every record is a dict with ``_record_type`` in {'metadata', 'result'}.

This module exposes ``read_pkl`` plus helpers that turn the raw occupancy
grids into compact per-voxel point clouds ready for 3D rendering.
"""

import base64
import gzip
import hashlib
import pickle
import struct

import numpy as np

MAGIC = b'SPARSEVLA-PKL-1\n'
_HEADER = struct.Struct('>Q32s')

# Occ3D / nuScenes 17-class occupancy palette (index 0..16); 17 == empty.
CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]
CLASS_COLORS = [  # (R, G, B), aligned positionally with CLASS_NAMES
    (189, 189, 189), (120, 120, 130), (255, 0, 0), (255, 130, 40),
    (255, 190, 0), (255, 110, 30), (255, 60, 150), (0, 110, 255),
    (255, 210, 50), (170, 80, 30), (210, 90, 45), (180, 185, 195),
    (205, 205, 210), (150, 155, 165), (150, 165, 85), (200, 175, 150),
    (45, 150, 65),
]
EMPTY_LABEL = 17

# Occupancy grid parameters (must match the SparseOccVLA config).
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
VOXEL_SIZE = [0.4, 0.4, 0.4]
GRID = [200, 200, 16]  # (X, Y, Z)


def read_pkl(path):
    """Parse a result file and return ``(metadata: dict, samples: list[dict])``.

    ``samples`` is the ordered list of per-frame result dicts (one per sample).
    Each dict normally carries ``occ_pred``, ``occ_forecasting``,
    ``generated_text``, ``token``, ``traj_forecasting`` and ``planning_result``.
    """
    with open(path, 'rb') as f:
        head = f.read(len(MAGIC))
    if head != MAGIC:
        return _read_plain_pickle(path)

    with open(path, 'rb') as f:
        data = f.read()

    metadata = {}
    samples = []
    off = len(MAGIC)
    while off + _HEADER.size <= len(data):
        size, digest = _HEADER.unpack(data[off:off + _HEADER.size])
        off += _HEADER.size
        payload = data[off:off + size]
        off += size
        if hashlib.sha256(payload).digest() != digest:
            raise ValueError('checksum mismatch in %s' % path)
        rec = pickle.loads(gzip.decompress(payload))
        rtype = rec.get('_record_type')
        if rtype == 'metadata':
            metadata = rec.get('metadata', rec)
        elif rtype == 'result':
            samples.append(rec.get('result', rec))
    return metadata, samples


def _read_plain_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        if obj.get('_record_type') == 'result':
            return {}, [obj.get('result', obj)]
        if 'result' in obj:
            return obj.get('metadata', {}), [obj['result']]
        return {}, list(obj.values())
    if isinstance(obj, list):
        return {}, obj
    return {}, [obj]


def occ_to_points(occ, pc_range=PC_RANGE, voxel_size=VOXEL_SIZE,
                  empty_label=EMPTY_LABEL):
    """Convert a ``(X, Y, Z)`` uint8 occupancy grid into occupied voxels.

    Returns ``(xyz, labels)`` where ``xyz`` is a float32 ``(n, 3)`` array of
    world-space voxel centers (meters, ego frame) and ``labels`` is a uint8
    ``(n,)`` array of semantic class ids.
    """
    occ = np.asarray(occ)
    if occ.ndim != 3:
        raise ValueError('expected 3D occupancy grid, got shape %s'
                         % (occ.shape,))
    mask = occ != empty_label
    idx = np.argwhere(mask)  # (n, 3) in (x, y, z) index order
    labels = occ[mask].astype(np.uint8)
    xyz = (idx.astype(np.float32) + 0.5) * np.asarray(voxel_size, np.float32) \
        + np.asarray(pc_range[:3], np.float32)
    return xyz, labels


def b64_float32(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode('ascii')


def b64_uint8(arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    return base64.b64encode(arr.tobytes()).decode('ascii')


def occ_to_payload(occ, pc_range=PC_RANGE, voxel_size=VOXEL_SIZE,
                   empty_label=EMPTY_LABEL):
    """Serialize one occupancy grid as ``{'n', 'xyz_b64', 'label_b64'}``."""
    xyz, labels = occ_to_points(occ, pc_range, voxel_size, empty_label)
    return {
        'n': int(xyz.shape[0]),
        'xyz_b64': b64_float32(xyz),
        'label_b64': b64_uint8(labels),
    }


def occ_meta():
    return {
        'pc_range': PC_RANGE,
        'voxel_size': VOXEL_SIZE,
        'grid': GRID,
        'class_names': CLASS_NAMES,
        'class_colors': CLASS_COLORS,
        'empty_label': EMPTY_LABEL,
    }
