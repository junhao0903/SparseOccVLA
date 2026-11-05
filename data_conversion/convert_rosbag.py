#!/usr/bin/env python3
"""Convert a ROS1 bag into a lightweight nuScenes-like inference scene."""
import argparse
import glob
import os
import pickle
from bisect import bisect_left

import cv2
import numpy as np
import rosbag
import yaml
from sensor_msgs import point_cloud2


def stamp(msg):
    return msg.header.stamp.to_sec()


def opencv_matrix(path, key):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    node = fs.getNode(key)
    value = node.mat() if not node.empty() else None
    fs.release()
    if value is None:
        raise ValueError('Missing %s in %s' % (key, path))
    return value.astype(np.float64)


def load_calibration(root, filename):
    path = os.path.join(root, 'camera', filename)
    K = opencv_matrix(path, 'K')
    D = opencv_matrix(path, 'D').reshape(-1)
    return K, D


def read_cloud(msg):
    fields = {f.name for f in msg.fields}
    names = ['x', 'y', 'z'] + (['intensity'] if 'intensity' in fields else [])
    return np.asarray(list(point_cloud2.read_points(msg, field_names=names, skip_nans=True)), dtype=np.float32)


def nearest(records, times, target, tolerance):
    i = bisect_left(times, target)
    options = [j for j in (i - 1, i) if 0 <= j < len(times)]
    if not options:
        return None
    j = min(options, key=lambda k: abs(times[k] - target))
    return records[j] if abs(times[j] - target) <= tolerance else None


def bag_files(path, recursive):
    if os.path.isfile(path):
        return [path]
    pattern = '**/*.bag' if recursive else '*.bag'
    return sorted(glob.glob(os.path.join(path, pattern), recursive=recursive))



def load_config(config_path):
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def create_scene_directory(cfg, bags):
    input_path = cfg['input']['path']
    if os.path.isfile(input_path):
        scene_name = os.path.basename(os.path.dirname(bags[0]))
    else:
        scene_name = os.path.basename(os.path.normpath(input_path))
    scene_dir = os.path.join(cfg['output']['root'], scene_name)
    os.makedirs(os.path.join(scene_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, 'sweeps', 'LIDAR_TOP'), exist_ok=True)
    return scene_name, scene_dir


def collect_topic_records(bags, cfg):
    topics = {camera['topic'] for camera in cfg['cameras'].values()}
    topics.add(cfg['keyframe_topic'])
    topics.update(cfg['lidars'])
    topics.add(cfg['localization_topic'])
    records = {topic: [] for topic in topics}

    for bag_path in bags:
        print('Indexing', bag_path)
        with rosbag.Bag(bag_path) as bag:
            for topic, msg, _ in bag.read_messages(topics=list(topics)):
                records[topic].append((stamp(msg), msg))

    for topic_records in records.values():
        topic_records.sort(key=lambda item: item[0])
    return records


def match_frame_sensors(timestamp, records, cfg):
    tolerance = float(cfg['sync_tolerance_sec'])
    matched_cameras = {}
    matched_lidars = []
    missing = []

    for camera_name, camera_cfg in cfg['cameras'].items():
        camera_records = records[camera_cfg['topic']]
        match = nearest(camera_records, [item[0] for item in camera_records], timestamp, tolerance)
        if match is None:
            missing.append(camera_name)
        else:
            matched_cameras[camera_name] = match

    for topic in cfg['lidars']:
        lidar_records = records[topic]
        match = nearest(lidar_records, [item[0] for item in lidar_records], timestamp, tolerance)
        if match is None:
            missing.append(topic)
        else:
            matched_lidars.append(match)

    localization_records = records[cfg['localization_topic']]
    localization = nearest(
        localization_records,
        [item[0] for item in localization_records],
        timestamp,
        tolerance,
    )
    if localization is None:
        missing.append('localization')

    return matched_cameras, matched_lidars, localization, missing


def rectify_camera_image(msg, source_calibration, target_intrinsic, image_size):
    raw_image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw_image is None:
        return None

    source_intrinsic, distortion = source_calibration
    output_intrinsic = target_intrinsic.copy()
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_intrinsic,
        distortion,
        None,
        output_intrinsic,
        tuple(image_size),
        cv2.CV_32FC1,
    )
    return cv2.remap(
        raw_image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def save_camera_data(scene_dir, sample_token, matched_cameras, cfg, calibrations, target_intrinsics):
    camera_infos = {}
    for camera_name, (image_timestamp, msg) in matched_cameras.items():
        image = rectify_camera_image(
            msg,
            calibrations[camera_name],
            target_intrinsics[camera_name],
            cfg['image_size'],
        )
        if image is None:
            return None

        relative_path = os.path.join('samples', camera_name, sample_token + '.jpg')
        image_path = os.path.join(scene_dir, relative_path)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if not cv2.imwrite(image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise IOError('Could not write image: ' + image_path)

        camera_infos[camera_name] = {
            'data_path': image_path,
            'timestamp': int(image_timestamp * 1e6),
            'cam_intrinsic': target_intrinsics[camera_name].copy(),
            'image_size': tuple(cfg['image_size']),
        }
    return camera_infos


def save_lidar_data(scene_dir, sample_token, lidar_matches):
    points = np.concatenate([read_cloud(msg) for _, msg in lidar_matches], axis=0)
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), np.float32)], axis=1)

    lidar_path = os.path.join(scene_dir, 'sweeps', 'LIDAR_TOP', sample_token + '.bin')
    points.astype(np.float32).tofile(lidar_path)
    return lidar_path


def make_frame_info(scene_name, timestamp, sample_token, camera_infos, lidar_path, lidar_matches, localization, infos):
    localization_timestamp, localization_msg = localization
    info = {
        'token': sample_token,
        'scene_token': scene_name,
        'scene_name': scene_name,
        'frame_idx': len(infos),
        'timestamp': int(timestamp * 1e6),
        'img_timestamp': [camera_infos[name]['timestamp'] for name in camera_infos],
        'lidar_path': lidar_path,
        'cams': camera_infos,
        'sweeps': [],
        'cam_sweeps': [],
        'localization_timestamp': int(localization_timestamp * 1e6),
        'localization': {
            'longitude': localization_msg.longitude,
            'latitude': localization_msg.latitude,
            'altitude': localization_msg.altitude,
            'velocity': [localization_msg.vel_x, localization_msg.vel_y, localization_msg.vel_z],
            'rpy': [localization_msg.roll, localization_msg.pitch, localization_msg.yaw],
        },
        'lidar_timestamps': [int(item[0] * 1e6) for item in lidar_matches],
        'can_bus': np.zeros(13, dtype=np.float32),
        'gt_planning_command': 0,
        'prev': infos[-1]['token'] if infos else '',
        'next': '',
    }
    if infos:
        infos[-1]['next'] = sample_token
    return info


def process_frames(scene_name, scene_dir, records, cfg):
    calibrations = {
        name: load_calibration(cfg['calibration_dir'], camera['calibration'])
        for name, camera in cfg['cameras'].items()
    }
    target_intrinsics = {
        name: np.asarray(cfg['target_intrinsics'][name], dtype=np.float64)
        for name in cfg['cameras']
    }
    keyframes = records[cfg['keyframe_topic']]
    target_frequency = float(cfg.get('output_frequency_hz', 10.0))
    infos = []
    dropped = {}
    last_candidate_timestamp = None

    for timestamp, _ in keyframes:
        if last_candidate_timestamp is not None:
            min_interval = 1.0 / target_frequency - 0.015
            if timestamp - last_candidate_timestamp < min_interval:
                continue

        matched_cameras, matched_lidars, localization, missing = match_frame_sensors(timestamp, records, cfg)
        if missing:
            dropped['missing_sensor'] = dropped.get('missing_sensor', 0) + 1
            continue

        sample_token = '%06d' % len(infos)
        camera_infos = save_camera_data(
            scene_dir, sample_token, matched_cameras, cfg, calibrations, target_intrinsics
        )
        if camera_infos is None:
            dropped['invalid_image'] = dropped.get('invalid_image', 0) + 1
            continue

        last_candidate_timestamp = timestamp
        lidar_path = save_lidar_data(scene_dir, sample_token, matched_lidars)
        infos.append(make_frame_info(
            scene_name,
            timestamp,
            sample_token,
            camera_infos,
            lidar_path,
            matched_lidars,
            localization,
            infos,
        ))
    return infos, dropped


def write_infos(scene_dir, infos):
    info_path = os.path.join(scene_dir, 'nuscenes_infos_test.pkl')
    with open(info_path, 'wb') as info_file:
        pickle.dump(infos, info_file, protocol=pickle.HIGHEST_PROTOCOL)
    return info_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)

    input_path = cfg['input']['path']
    bags = bag_files(input_path, cfg['input'].get('recursive_bags', False))
    if not bags:
        raise SystemExit('No rosbag files found: ' + input_path)

    scene_name, scene_dir = create_scene_directory(cfg, bags)
    records = collect_topic_records(bags, cfg)
    infos, dropped = process_frames(scene_name, scene_dir, records, cfg)
    info_path = write_infos(scene_dir, infos)

    print('Wrote %d frames to %s; dropped=%s' % (len(infos), info_path, dropped))
    print('WARNING: localization pose/can_bus and LiDAR-to-camera extrinsics still need validation before model inference.')


if __name__ == '__main__':
    main()
