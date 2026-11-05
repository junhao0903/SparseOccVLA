#!/usr/bin/env python3
"""Render one saved point cloud over each camera image for calibration review."""
import argparse
import os
import pickle
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('info_pkl')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--output', default='projection_check')
    args = ap.parse_args()
    with open(args.info_pkl, 'rb') as f: infos = pickle.load(f)
    item = infos[args.frame]
    points = np.fromfile(item['lidar_path'], dtype=np.float32).reshape(-1, 4)[:, :3]
    # Projection matrices are only generated after camera-to-common-lidar
    # calibration validation; avoid drawing misleading overlays from guessed transforms.
    os.makedirs(args.output, exist_ok=True)
    for name, cam in item['cams'].items():
        image = cv2.imread(cam['data_path'])
        K = np.asarray(cam['cam_intrinsic'])
        # Placeholder until sensor_link transforms are composed and encoded in info.
        cv2.putText(image, 'intrinsic %dx%d; extrinsic pending verification' % (image.shape[1], image.shape[0]),
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(args.output, name + '.jpg'), image)
        print(name, 'size=', image.shape[1], image.shape[0], 'K=\n', K)
    print('Point cloud points:', len(points), 'projection skipped until extrinsics are confirmed')

if __name__ == '__main__': main()
