#!/usr/bin/env python3
"""Render SparseOccVLA predictions saved by tools/test.py --out."""

import argparse
import gzip
import hashlib
import os
import pickle
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np


OCCUPANCY_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
]
CONTAINER_MAGIC = b'SPARSEVLA-PKL-1\n'
CONTAINER_HEADER = struct.Struct('>Q32s')


def _load_results(path, max_samples):
    with open(path, 'rb') as file:
        magic = file.read(len(CONTAINER_MAGIC))
        if magic == CONTAINER_MAGIC:
            results = []
            metadata = {}
            while True:
                header = file.read(CONTAINER_HEADER.size)
                if not header or len(header) != CONTAINER_HEADER.size:
                    break
                size, digest = CONTAINER_HEADER.unpack(header)
                payload = file.read(size)
                if (len(payload) != size or
                        hashlib.sha256(payload).digest() != digest):
                    print('Ignoring incomplete or damaged final record.')
                    break
                try:
                    record = pickle.loads(gzip.decompress(payload))
                except (OSError, EOFError, pickle.PickleError):
                    print('Ignoring unreadable final record.')
                    break
                if record.get('_record_type') == 'metadata':
                    metadata = record.get('metadata', {})
                elif record.get('_record_type') == 'result':
                    results.append(record['result'])
                    if max_samples is not None and len(results) >= max_samples:
                        break
            return results, metadata

    # Backward compatibility with the earlier monolithic pickle bundle.
    with open(path, 'rb') as file:
        bundle = pickle.load(file)
    if isinstance(bundle, dict) and 'results' in bundle:
        return bundle['results'][:max_samples], bundle.get('metadata', {})
    if isinstance(bundle, list):
        return bundle[:max_samples], {}
    raise ValueError('Unrecognized result pickle format')


def _topdown(volume):
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError('Expected an occupancy volume shaped [X, Y, Z], got {}'.format(volume.shape))
    occupied = volume != 17
    z_index = volume.shape[-1] - 1 - occupied[..., ::-1].argmax(axis=-1)
    top = np.take_along_axis(volume, z_index[..., None], axis=-1)[..., 0].copy()
    top[~occupied.any(axis=-1)] = 17
    return top


def _save_occupancy(volume, title, path):
    top = _topdown(volume)
    cmap = plt.get_cmap('tab20', 18)
    norm = BoundaryNorm(np.arange(-0.5, 18.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    image = ax.imshow(top.T, origin='lower', interpolation='nearest', cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel('X voxel')
    ax.set_ylabel('Y voxel')
    colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(18), fraction=0.045, pad=0.03)
    colorbar.ax.set_yticklabels(OCCUPANCY_CLASSES, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_language(predictions, path):
    with open(path, 'w', encoding='utf-8') as file:
        if not predictions:
            file.write('No language predictions were stored for this sample.\n')
            return
        for item in predictions:
            file.write('Q: {}\nA: {}\n'.format(item.get('Q', ''), item.get('A', '')))
            if item.get('G') is not None:
                file.write('Reference: {}\n'.format(item['G']))
            file.write('\n')


def _save_trajectory(planning, title, path):
    trajectory = planning.get('trajectory') if isinstance(planning, dict) else planning
    if trajectory is None:
        return False
    trajectory = np.asarray(trajectory, dtype=np.float32).squeeze()
    if trajectory.ndim != 2 or trajectory.shape[-1] < 2:
        print('Skipping trajectory with unsupported shape: {}'.format(trajectory.shape))
        return False
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.plot(trajectory[:, 0], trajectory[:, 1], '-o', markersize=3)
    ax.scatter([0], [0], marker='*', s=100, label='Ego')
    for index, (x, y) in enumerate(trajectory[:, :2]):
        ax.annotate(str(index + 1), (x, y), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('Forward (m)')
    ax.set_ylabel('Lateral (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pkl', help='Result pickle produced by tools/test.py --out')
    parser.add_argument('--out-dir', default='./vla_visuals', help='Directory for rendered files')
    parser.add_argument('--max-samples', type=int, default=8, help='Maximum number of samples to render')
    args = parser.parse_args()

    results, metadata = _load_results(args.pkl, args.max_samples)
    print('Loaded {} samples; model_use_gen_token={}'.format(
        len(results), metadata.get('model_use_gen_token', 'unknown')))

    os.makedirs(args.out_dir, exist_ok=True)
    for index, result in enumerate(results):
        token = str(result.get('token', 'sample_{:06d}'.format(index)))
        safe_token = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in token)

        language = result.get('language_prediction', result.get('generated_text'))
        _save_language(language, os.path.join(args.out_dir, safe_token + '_language.txt'))

        current_occ = result.get('occ_pred')
        if current_occ is not None:
            _save_occupancy(current_occ, token + ' | current occupancy',
                            os.path.join(args.out_dir, safe_token + '_occ_current.png'))

        forecasts = result.get('occ_forecasting')
        if forecasts is not None:
            for horizon, volume in enumerate(np.asarray(forecasts)):
                _save_occupancy(volume, '{} | occupancy forecast t+{}s'.format(token, horizon),
                                os.path.join(args.out_dir, '{}_occ_t{}.png'.format(safe_token, horizon)))

        planning = result.get('planning_result', result.get('traj_forecasting'))
        has_trajectory = _save_trajectory(
            planning, token + ' | planned trajectory',
            os.path.join(args.out_dir, safe_token + '_trajectory.png'))
        if has_trajectory:
            print('{}: trajectory visualization saved'.format(token))
        else:
            reason = planning.get('reason') if isinstance(planning, dict) else None
            print('{}: no trajectory prediction in this result{}'.format(
                token, ' ({})'.format(reason) if reason else ''))

    print('Rendered {} samples to {}'.format(len(results), os.path.abspath(args.out_dir)))


if __name__ == '__main__':
    main()
