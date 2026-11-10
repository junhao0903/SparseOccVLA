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
    R = opencv_matrix(path, 'R')
    T = opencv_matrix(path, 'T').reshape(3)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    sensor_link = fs.getNode('sensor_link').string()
    image_width = int(fs.getNode('imgw').real())
    image_height = int(fs.getNode('imgh').real())
    fs.release()
    return {
        'K': K,
        'D': D,
        'sensor2ego_rotation_matrix': R,
        'sensor2ego_translation': T,
        'sensor_link': sensor_link,
        'image_width': image_width,
        'image_height': image_height,
    }


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
            matched_lidars.append((topic, match))

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

    # Calibration files describe the camera at its calibrated resolution, but
    # rosbag images may have been resized before they were recorded.
    source_intrinsic = source_calibration['K'].copy()
    scale_x = raw_image.shape[1] / float(source_calibration['image_width'])
    scale_y = raw_image.shape[0] / float(source_calibration['image_height'])
    source_intrinsic[0, :] *= scale_x
    source_intrinsic[1, :] *= scale_y
    distortion = source_calibration['D']
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


def save_camera_data(scene_dir, sample_token, matched_cameras, cfg, calibrations,
                     target_intrinsics, ego_translation, ego_rotation):
    camera_infos = {}
    for camera_name, (image_timestamp, msg) in matched_cameras.items():
        calibration = calibrations[camera_name]
        geometry_mode = cfg.get('image_geometry_mode', 'nuscenes')
        raw_mode = geometry_mode in ('raw', 'raw_scaled')
        image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        if raw_mode:
            image_size = (image.shape[1], image.shape[0])
            output_intrinsic = calibration['K'].copy()
            if geometry_mode == 'raw_scaled':
                scale_x = image.shape[1] / float(calibration['image_width'])
                scale_y = image.shape[0] / float(calibration['image_height'])
                output_intrinsic[0, :] *= scale_x
                output_intrinsic[1, :] *= scale_y
            distortion = calibration['D'].copy()
        else:
            image = rectify_camera_image(
                msg,
                calibration,
                target_intrinsics[camera_name],
                cfg['image_size'],
            )
            image_size = tuple(cfg['image_size'])
            output_intrinsic = target_intrinsics[camera_name].copy()
            distortion = None

        relative_path = os.path.join('samples', camera_name, sample_token + '.jpg')
        image_path = os.path.join(scene_dir, relative_path)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if raw_mode:
            with open(image_path, 'wb') as image_file:
                image_file.write(msg.data)
        elif not cv2.imwrite(image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise IOError('Could not write image: ' + image_path)

        # Camera YAML R maps vehicle coordinates into the camera frame
        # (p_cam = R @ (p_vehicle - camera_position)). nuScenes stores the
        # opposite direction in sensor2ego/sensor2lidar, so transpose R.
        sensor2ego_rotation_matrix = calibration['sensor2ego_rotation_matrix'].T
        sensor2ego_translation = calibration['sensor2ego_translation']
        sensor2ego_rotation = quaternion_from_matrix(sensor2ego_rotation_matrix)
        ego_rotation_matrix = quaternion_to_matrix(ego_rotation)
        # The pipeline's compose_lidar2img expects this field as a rotation
        # matrix (unlike sensor2ego_rotation, which follows nuScenes' wxyz).
        sensor2global_rotation = ego_rotation_matrix @ sensor2ego_rotation_matrix
        sensor2global_translation = (
            ego_translation + ego_rotation_matrix @ sensor2ego_translation
        )
        camera_infos[camera_name] = {
            'data_path': image_path,
            'timestamp': int(image_timestamp * 1e6),
            'cam_intrinsic': output_intrinsic,
            'image_size': image_size,
            'sensor_link': calibration['sensor_link'],
            'sensor2ego_rotation': sensor2ego_rotation,
            'sensor2ego_translation': sensor2ego_translation.copy(),
            'sensor2lidar_rotation': sensor2ego_rotation_matrix.copy(),
            'sensor2lidar_translation': sensor2ego_translation.copy(),
            'sensor2global_rotation': sensor2global_rotation,
            'sensor2global_translation': sensor2global_translation,
        }
        if distortion is not None:
            camera_infos[camera_name]['cam_distortion'] = distortion
    return camera_infos


def save_lidar_data(scene_dir, sample_token, lidar_matches, cfg):
    transformed_clouds = []
    for topic, (timestamp, msg) in lidar_matches:
        points = read_cloud(msg)
        if points.shape[1] == 3:
            points = np.concatenate([points, np.zeros((len(points), 1), np.float32)], axis=1)

        calibration_path = os.path.join(
            cfg['calibration_dir'], 'lidar', cfg['lidar_calibrations'][topic]
        )
        transform = opencv_matrix(calibration_path, 'RT')
        points[:, :3] = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        transformed_clouds.append(points)

    points = np.concatenate(transformed_clouds, axis=0)

    lidar_path = os.path.join(scene_dir, 'sweeps', 'LIDAR_TOP', sample_token + '.bin')
    points.astype(np.float32).tofile(lidar_path)
    return lidar_path


def quaternion_from_rpy(roll, pitch, yaw):
    """Return an (w, x, y, z) quaternion from ROS roll/pitch/yaw angles."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def quaternion_from_matrix(rotation):
    """Convert a 3x3 rotation matrix to an (w, x, y, z) quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = [
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = [
                (matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ]
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ]
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
            ]
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def localization_to_pose(localization_msg):
    """Map localization fields to the nuScenes-style pose and 13D can_bus."""
    translation = np.asarray([
        localization_msg.mercator_x,
        localization_msg.mercator_y,
        localization_msg.mercator_z,
    ], dtype=np.float64)
    rotation = quaternion_from_rpy(
        localization_msg.roll,
        localization_msg.pitch,
        localization_msg.yaw,
    )
    acceleration = np.asarray([
        localization_msg.accel_x,
        localization_msg.accel_y,
        localization_msg.accel_z,
    ], dtype=np.float64)
    angular_velocity = np.asarray([
        localization_msg.angular_x,
        localization_msg.angular_y,
        localization_msg.angular_z,
    ], dtype=np.float64)
    can_bus = np.concatenate([translation, rotation, acceleration, angular_velocity])
    return translation, rotation, can_bus


def make_frame_info(scene_name, timestamp, sample_token, camera_infos, cam_sweeps,
                    lidar_path, lidar_matches, localization, infos):
    localization_timestamp, localization_msg = localization
    ego_translation, ego_rotation, can_bus = localization_to_pose(localization_msg)
    info = {
        'token': sample_token,
        'scene_token': scene_name,
        'scene_name': scene_name,
        'frame_idx': len(infos),
        'timestamp': int(timestamp * 1e6),
        'img_timestamp': [camera_infos[name]['timestamp'] for name in camera_infos],
        'lidar_path': lidar_path,
        'cams': camera_infos,
        # The fused point cloud messages declare base_link, so the reference
        # LiDAR frame is represented as coincident with ego/base_link.
        'lidar2ego_translation': np.zeros(3, dtype=np.float64),
        'lidar2ego_rotation': np.asarray([1.0, 0.0, 0.0, 0.0]),
        'ego2global_translation': ego_translation,
        'ego2global_rotation': ego_rotation,
        'sweeps': [],
        'cam_sweeps': cam_sweeps,
        'localization_timestamp': int(localization_timestamp * 1e6),
        'localization': {
            'longitude': localization_msg.longitude,
            'latitude': localization_msg.latitude,
            'altitude': localization_msg.altitude,
            'velocity': [localization_msg.vel_x, localization_msg.vel_y, localization_msg.vel_z],
            'rpy': [localization_msg.roll, localization_msg.pitch, localization_msg.yaw],
        },
        'pose_translation_source': 'localization_result.mercator_xyz',
        'pose_translation_is_constant_in_bag': False,
        'lidar_timestamps': [int(item[1][0] * 1e6) for item in lidar_matches],
        'can_bus': can_bus.astype(np.float32),
        'gt_planning_command': 0,
        'gt_planning_command_source': 'placeholder_no_command_topic',
        'gt_planning_command_valid': False,
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
    if cfg.get('image_geometry_mode', 'nuscenes') in ('raw', 'raw_scaled'):
        target_intrinsics = {
            name: calibration['K'] for name, calibration in calibrations.items()
        }
    else:
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
        localization_msg = localization[1]
        ego_translation, ego_rotation, _ = localization_to_pose(localization_msg)

        # Preserve the original camera-rate frames between adjacent 10 Hz
        # keyframes so the test pipeline can sample real temporal images.
        sweep_times = []
        if last_candidate_timestamp is not None:
            sweep_times = [
                sweep_timestamp
                for sweep_timestamp, _ in keyframes
                if last_candidate_timestamp < sweep_timestamp < timestamp
            ]
        cam_sweeps = []
        for sweep_index, sweep_timestamp in enumerate(reversed(sweep_times)):
            sweep_cameras = {}
            sweep_missing = False
            for camera_name, camera_cfg in cfg['cameras'].items():
                camera_records = records[camera_cfg['topic']]
                camera_match = nearest(
                    camera_records,
                    [item[0] for item in camera_records],
                    sweep_timestamp,
                    float(cfg['sync_tolerance_sec']),
                )
                if camera_match is None:
                    sweep_missing = True
                    break
                sweep_cameras[camera_name] = camera_match
            sweep_localization_records = records[cfg['localization_topic']]
            sweep_localization = nearest(
                sweep_localization_records,
                [item[0] for item in sweep_localization_records],
                sweep_timestamp,
                float(cfg['sync_tolerance_sec']),
            )
            if sweep_missing or sweep_localization is None:
                continue
            sweep_pose = localization_to_pose(sweep_localization[1])
            sweep_token = '%s_sweep_%02d' % (sample_token, sweep_index)
            sweep_info = save_camera_data(
                scene_dir,
                sweep_token,
                sweep_cameras,
                cfg,
                calibrations,
                target_intrinsics,
                sweep_pose[0],
                sweep_pose[1],
            )
            if sweep_info is not None:
                cam_sweeps.append(sweep_info)

        camera_infos = save_camera_data(
            scene_dir, sample_token, matched_cameras, cfg, calibrations,
            target_intrinsics, ego_translation, ego_rotation,
        )
        if camera_infos is None:
            dropped['invalid_image'] = dropped.get('invalid_image', 0) + 1
            continue

        last_candidate_timestamp = timestamp
        lidar_path = save_lidar_data(scene_dir, sample_token, matched_lidars, cfg)
        infos.append(make_frame_info(
            scene_name,
            timestamp,
            sample_token,
            camera_infos,
            cam_sweeps,
            lidar_path,
            matched_lidars,
            localization,
            infos,
        ))
    if infos:
        translations = np.asarray([info['ego2global_translation'] for info in infos])
        translation_is_constant = bool(np.allclose(translations, translations[0], atol=1e-6))
        for info in infos:
            info['pose_translation_is_constant_in_bag'] = translation_is_constant
    return infos, dropped


def write_infos(scene_dir, infos, cfg):
    info_path = os.path.join(scene_dir, 'nuscenes_infos_test.pkl')
    data = {
        'infos': infos,
        'metadata': {
            'version': cfg.get('target_version', 'v1.0-mini'),
            'source_format': cfg.get('source_format', 'rosbag'),
            'target_format': cfg.get('target_format', 'nuscenes'),
            'image_geometry_mode': cfg.get('image_geometry_mode', 'nuscenes'),
        },
    }
    with open(info_path, 'wb') as info_file:
        pickle.dump(data, info_file, protocol=pickle.HIGHEST_PROTOCOL)
    return info_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    parser.add_argument('--image-geometry', choices=['nuscenes', 'raw', 'raw_scaled'], default=None,
                        help='Use nuScenes target geometry, exact raw K, or raw pixels with K scaled to native size')
    parser.add_argument('--output-root', default=None,
                        help='Override output root to keep conversion variants separate')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.image_geometry is not None:
        cfg['image_geometry_mode'] = args.image_geometry
    if args.output_root is not None:
        cfg['output']['root'] = args.output_root

    input_path = cfg['input']['path']
    bags = bag_files(input_path, cfg['input'].get('recursive_bags', False))
    if not bags:
        raise SystemExit('No rosbag files found: ' + input_path)

    scene_name, scene_dir = create_scene_directory(cfg, bags)
    records = collect_topic_records(bags, cfg)
    infos, dropped = process_frames(scene_name, scene_dir, records, cfg)
    info_path = write_infos(scene_dir, infos, cfg)

    print('Wrote %d frames to %s; dropped=%s' % (len(infos), info_path, dropped))
    print('WARNING: localization pose/can_bus still need validation before model inference.')


if __name__ == '__main__':
    main()
