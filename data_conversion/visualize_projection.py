#!/usr/bin/env python3
"""Project a converted LiDAR cloud onto all six camera images."""
import argparse
import os
import pickle

import cv2
import numpy as np


def load_infos(path):
    with open(path, 'rb') as info_file:
        payload = pickle.load(info_file)
    return payload['infos'] if isinstance(payload, dict) else payload


def project_lidar_to_camera(points_lidar, camera_info, max_depth):
    """Project lidar-frame points using camera-to-lidar calibration."""
    sensor2lidar_rotation = np.asarray(camera_info['sensor2lidar_rotation'], dtype=np.float64)
    sensor2lidar_translation = np.asarray(camera_info['sensor2lidar_translation'], dtype=np.float64)
    intrinsic = np.asarray(camera_info['cam_intrinsic'], dtype=np.float64)

    # p_lidar = R_camera_to_lidar @ p_camera + t_camera_to_lidar
    # Convert row-vector lidar coordinates to row-vector camera coordinates.
    lidar2camera = np.linalg.inv(sensor2lidar_rotation).T
    points_camera = (points_lidar - sensor2lidar_translation) @ lidar2camera
    depth = points_camera[:, 2]
    front_mask = (depth > 0.1) & (depth < max_depth)
    points_camera = points_camera[front_mask]
    depth = depth[front_mask]

    pixels_h = points_camera @ intrinsic.T
    uv = pixels_h[:, :2] / pixels_h[:, 2:3]
    return uv, depth, front_mask


def colorize_depth(depth, near, far):
    if len(depth) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    normalized = np.clip((depth - near) / max(far - near, 1e-6), 0.0, 1.0)
    grayscale = (normalized * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO).reshape(-1, 3)


def render_camera_projection(image, uv, depth, max_depth, point_stride):
    height, width = image.shape[:2]
    if point_stride > 1:
        uv = uv[::point_stride]
        depth = depth[::point_stride]
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    uv = uv[inside]
    depth = depth[inside]

    colors = colorize_depth(depth, 0.1, max_depth)
    overlay = image.copy()
    for (u, v), color in zip(uv, colors):
        cv2.circle(overlay, (int(round(u)), int(round(v))), 2, color.tolist(), -1,
                   lineType=cv2.LINE_AA)
    rendered = cv2.addWeighted(image, 0.45, overlay, 0.75, 0)
    return rendered, len(uv), width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('info_pkl', help='Converted nuscenes_infos_test.pkl')
    parser.add_argument('--frame', type=int, default=0, help='Frame index in the pkl')
    parser.add_argument('--output', default='projection_check', help='Output directory')
    parser.add_argument('--point-stride', type=int, default=3,
                        help='Draw every Nth point to keep overlays legible')
    parser.add_argument('--max-depth', type=float, default=80.0,
                        help='Discard points farther than this many meters')
    args = parser.parse_args()
    if args.point_stride < 1:
        parser.error('--point-stride must be >= 1')

    infos = load_infos(args.info_pkl)
    if not 0 <= args.frame < len(infos):
        parser.error('Frame index %d outside [0, %d)' % (args.frame, len(infos)))
    info = infos[args.frame]
    point_data = np.fromfile(info['lidar_path'], dtype=np.float32)
    if point_data.size % 4:
        raise ValueError('Expected xyz+intensity point data, got %d floats' % point_data.size)
    points_lidar = point_data.reshape(-1, 4)[:, :3]

    os.makedirs(args.output, exist_ok=True)
    print('frame=%d token=%s points=%d' % (args.frame, info['token'], len(points_lidar)))
    for camera_name, camera_info in info['cams'].items():
        image = cv2.imread(camera_info['data_path'], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(camera_info['data_path'])
        uv, depth, front_mask = project_lidar_to_camera(
            points_lidar, camera_info, args.max_depth
        )
        rendered, visible_count, width, height = render_camera_projection(
            image, uv, depth, args.max_depth, args.point_stride
        )
        filename = os.path.join(args.output, camera_name + '.jpg')
        if not cv2.imwrite(filename, rendered):
            raise IOError('Could not write ' + filename)
        print('%s: size=%dx%d front=%d in_image=%d saved=%s' % (
            camera_name, width, height, int(front_mask.sum()), visible_count, filename
        ))


if __name__ == '__main__':
    main()
