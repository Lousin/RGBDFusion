from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

import _init_paths
from core.config import cfg, update_config
from core.utils.utils import load_tango_3d_keypoints , load_camera_intrinsics

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing RAPTOR.')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                help="Modify config options using the command-line",
                default=None,
                nargs=argparse.REMAINDER)

    # additional argument unrelated to cfg
    parser.add_argument('--csvfile',
                        required=True,
                        type=str)
    parser.add_argument('--no_masks',
                        dest='load_masks',
                        action='store_false')
    parser.add_argument('--no_labels',
                        dest='load_labels',
                        action='store_false')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    datadir = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.DATANAME)
    print('Loading data from: {}'.format(datadir))

    # Load camera intrinsics
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)

    # Read Tango 3D keypoints
    keypoints_file = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.DATANAME, cfg.DATASET.KEYPOINTS)
    keypts3d = load_tango_3d_keypoints(keypoints_file)  # (26, 3) [m]


    # Where to save resized image?
    imagedir = os.path.join(datadir, cfg.DATASET.DOMAIN,
                            f'images_{cfg.DATASET.INPUT_SIZE[0]}x{cfg.DATASET.INPUT_SIZE[1]}_RGB')
    if not os.path.exists(imagedir): os.makedirs(imagedir)
    print(f'Resized images will be saved to {imagedir}')

    # Where to save resized distance maps?
    depthdir = os.path.join(datadir, cfg.DATASET.DOMAIN,
                            f'images_{cfg.DATASET.INPUT_SIZE[0]}x{cfg.DATASET.INPUT_SIZE[1]}_depth')
    if not os.path.exists(depthdir): os.makedirs(depthdir)
    print(f'Resized distance maps will be saved to {depthdir}')

    if args.load_masks:
        maskdir = os.path.join(datadir, cfg.DATASET.DOMAIN,
                               f'masks_{int(cfg.DATASET.INPUT_SIZE[0] / cfg.DATASET.OUTPUT_SIZE[0])}x{int(cfg.DATASET.INPUT_SIZE[1] / cfg.DATASET.OUTPUT_SIZE[0])}')
        if not os.path.exists(maskdir): os.makedirs(maskdir)
        print(f'Resized masks will be saved to {maskdir}')

    labels = pd.read_csv(args.csvfile, header=None)


    for idx in tqdm(range(len(labels))):

        # ---------- Read image & resize & save
        img_name = labels.loc[idx, 0]
        filename = f"scene{img_name}.png" # 构建实际的文件名
        image = cv2.imread(os.path.join(datadir, cfg.DATASET.DOMAIN, 'camera_output', filename), cv2.IMREAD_GRAYSCALE)
        image    = cv2.resize(image, tuple(cfg.DATASET.INPUT_SIZE))
        cv2.imwrite(os.path.join(imagedir, filename), image)

        # ---------- Read distance map & resize & save
        depth_filename = f"distances{img_name}.png"  # 构建实际的文件名
        depth_image = cv2.imread(os.path.join(datadir, cfg.DATASET.DOMAIN, 'distances', depth_filename), cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.resize(image, tuple(cfg.DATASET.INPUT_SIZE))
        cv2.imwrite(os.path.join(depthdir, depth_filename), depth_image)



        # ---------- Read mask & resize & save
        if args.load_masks:
            mask_filename = f"masks{img_name}.png"
            mask = cv2.imread(os.path.join(datadir, cfg.DATASET.DOMAIN, 'masks',mask_filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, tuple([int(s / cfg.DATASET.OUTPUT_SIZE[0]) for s in cfg.DATASET.INPUT_SIZE]))
            cv2.imwrite(os.path.join(maskdir, mask_filename), mask)


if __name__ == '__main__':
    main()

