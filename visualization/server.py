#!/usr/bin/env python3
"""Local web server for the SparseOccVLA result visualizer.

Serves the static frontend from ``visualization/static/`` and exposes a small
JSON API for listing/loading result pkl files and reading individual samples.
Runs on the Python standard library only (no third-party deps).
"""

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

from pkl_reader import read_pkl, occ_to_payload, occ_meta

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, 'static')
DEFAULT_HOST = os.environ.get('SPARSEOCC_VIZ_HOST', '0.0.0.0')
DEFAULT_PORT = int(os.environ.get('SPARSEOCC_VIZ_PORT', '8000'))

_MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.ico': 'image/x-icon',
}

_CACHE = {}
_CACHE_LOCK = threading.Lock()

_TRAJ_KEYS = ('traj_forecasting', 'trajectory', 'planning_traj')


def _cached_pkl(path):
    abspath = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(abspath)
    except OSError:
        raise FileNotFoundError(path)
    key = (abspath, mtime)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = read_pkl(abspath)
        return _CACHE[key]


def _as_matrix(value):
    if value is None:
        return None
    if isinstance(value, str):
        p = value.strip()
        if not p:
            return None
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        if p.endswith('.json'):
            with open(p) as f:
                return json.load(f)
        return np.load(p).tolist()
    arr = np.asarray(value, dtype=np.float64)
    return arr.tolist()


def _image_data_uri(value):
    if value is None:
        return None
    if isinstance(value, str) and value.startswith('data:'):
        return value
    path = value if isinstance(value, str) else None
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME.get(ext, 'image/jpeg')
    with open(path, 'rb') as f:
        raw = f.read()
    return 'data:%s;base64,%s' % (mime, base64.b64encode(raw).decode('ascii'))


def _trajectory(sample):
    for key in _TRAJ_KEYS:
        v = sample.get(key)
        if v is not None:
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                return [list(map(float, p)) for p in v]
    planning = sample.get('planning_result')
    if isinstance(planning, dict) and planning.get('available'):
        traj = planning.get('trajectory')
        if isinstance(traj, list) and traj and isinstance(traj[0], (list, tuple)):
            return [list(map(float, p)) for p in traj]
    return None


def _sample_payload(sample, idx):
    occ_pred = sample.get('occ_pred')
    if occ_pred is None or not isinstance(occ_pred, np.ndarray):
        occ_pred = None

    occ_fu = sample.get('occ_forecasting')
    occ_fu_list = None
    if isinstance(occ_fu, np.ndarray) and occ_fu.ndim == 4:
        occ_fu_list = [occ_to_payload(occ_fu[t]) for t in range(occ_fu.shape[0])]

    text = sample.get('generated_text') or sample.get('language_prediction')
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if not isinstance(text, list):
        text = None

    try:
        intrinsics = _as_matrix(sample.get('intrinsics'))
    except (OSError, ValueError):
        intrinsics = None
    try:
        extrinsics = _as_matrix(sample.get('extrinsics')
                                or sample.get('lidar2img'))
    except (OSError, ValueError):
        extrinsics = None

    token = sample.get('token')

    return {
        'idx': idx,
        'token': token if isinstance(token, str) else None,
        'text': text,
        'occ_pred': occ_to_payload(occ_pred) if occ_pred is not None else None,
        'occ_forecasting': occ_fu_list,
        'trajectory': _trajectory(sample),
        'front_image': _image_data_uri(sample.get('front_image')),
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
    }


def _list_pkls(root):
    root = os.path.abspath(root or os.path.expanduser('~'))
    if not os.path.isdir(root):
        return {'root': root, 'error': 'not a directory'}
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        depth = dirpath[len(root):].count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith('.pkl'):
                p = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = -1
                files.append({'path': p, 'size': size})
        if len(files) > 500:
            break
    files.sort(key=lambda f: f['path'])
    return {'root': root, 'files': files}


class Handler(BaseHTTPRequestHandler):
    server_version = 'SparseOccVLAViz/1.0'

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        self._send(status, json.dumps(obj))

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)

        if route == '/api/list_pkls':
            self._send_json(_list_pkls(qs.get('root', [None])[0]))
            return

        if route == '/api/load':
            path = qs.get('path', [None])[0]
            if not path:
                self._send_json({'error': 'missing path'}, 400)
                return
            try:
                metadata, samples = _cached_pkl(path)
            except FileNotFoundError:
                self._send_json({'error': 'file not found: %s' % path}, 404)
                return
            except Exception as exc:
                self._send_json({'error': str(exc)}, 500)
                return
            summaries = []
            for i, s in enumerate(samples):
                text = s.get('generated_text') or s.get('language_prediction')
                summaries.append({
                    'idx': i,
                    'token': s.get('token'),
                    'n_text': len(text) if isinstance(text, list) else 0,
                    'has_traj': _trajectory(s) is not None,
                    'has_front': bool(s.get('front_image')),
                    'has_occ_fu': isinstance(s.get('occ_forecasting'), np.ndarray)
                                  and s['occ_forecasting'].ndim == 4,
                })
            self._send_json({
                'path': os.path.abspath(path),
                'count': len(samples),
                'occ_meta': occ_meta(),
                'metadata': metadata,
                'samples': summaries,
            })
            return

        if route == '/api/sample':
            path = qs.get('path', [None])[0]
            idx = int(qs.get('idx', ['0'])[0])
            if not path:
                self._send_json({'error': 'missing path'}, 400)
                return
            try:
                metadata, samples = _cached_pkl(path)
                sample = samples[idx]
            except (FileNotFoundError, IndexError):
                self._send_json({'error': 'invalid path or index'}, 404)
                return
            except Exception as exc:
                self._send_json({'error': str(exc)}, 500)
                return
            self._send_json(_sample_payload(sample, idx))
            return

        if route == '/' or route == '':
            route = '/index.html'
        self._serve_static(route)

    def _serve_static(self, route):
        rel = route.lstrip('/')
        target = os.path.abspath(os.path.join(STATIC_DIR, rel))
        if not target.startswith(os.path.abspath(STATIC_DIR)):
            self._send(403, 'forbidden', 'text/plain')
            return
        if not os.path.isfile(target):
            self._send(404, 'not found', 'text/plain')
            return
        ext = os.path.splitext(target)[1].lower()
        with open(target, 'rb') as f:
            body = f.read()
        self._send(200, body, _MIME.get(ext, 'application/octet-stream'))


def main():
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), Handler)
    print('SparseOccVLA visualizer running:')
    print('  Local:   http://127.0.0.1:%d' % DEFAULT_PORT)
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        lan_ip = s.getsockname()[0]
        s.close()
        print('  Network: http://%s:%d' % (lan_ip, DEFAULT_PORT))
    except OSError:
        pass
    print('Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
