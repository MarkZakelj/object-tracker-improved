import cv2
import os
from pathlib import Path
import numpy as np
import argparse
from vot.region import io as vot_io

import main_utils
import main_utils as utils

SEQUENCE_DIR = main_utils.SEQUENCE_DIR
TARGET_SEQ = 'moved_200'
INTERPOLATION_METHOD = 'RIFE'
SHOW_TRUE_MASK = False
INTERPOLATE = False
PER_CLICK = True
SHOW_RESULT = True
WRITE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_dir', default=SEQUENCE_DIR, help='path to sequences')
    parser.add_argument('--target_seq', default=TARGET_SEQ, help='target sequence')
    parser.add_argument('--write', action='store_true', default=WRITE, help='dont show images but write them to files')
    parser.add_argument('--dir', help='needed if write=True')
    args = parser.parse_args()
    video = utils.load_video(args.target_seq, interpolated=INTERPOLATE, interpolation_method=INTERPOLATION_METHOD)
    masks = utils.load_masks_raster(args.target_seq, interpolated=INTERPOLATE, interpolation_method=INTERPOLATION_METHOD)
    result_masks = utils.load_masks_raster(args.target_seq, interpolated=INTERPOLATE,
                                           interpolation_method=INTERPOLATION_METHOD, load_results=True)
    color = np.array([0, 255, 0], dtype='uint8')
    res_color = np.array([0, 0, 255], dtype='uint8')
    fps = utils.get_video_fps(args.target_seq)
    size = utils.get_video_size(args.target_seq)
    t_frame_int = round(1000 / (fps * (INTERPOLATE + 1)))
    if PER_CLICK:
        t_frame_int = 0
    print(t_frame_int)
    print('size:', size)
    Path(args.dir).mkdir(exist_ok=True, parents=True)
    for i, (ok, image) in enumerate(video):
        if not ok:
            break
        if SHOW_TRUE_MASK:
            image = utils.apply_mask_to_image(image, next(masks), color)
        if SHOW_RESULT:
            image = utils.apply_mask_to_image(image, next(result_masks), res_color)
        if args.write:
            if args.dir is None:
                raise ValueError('dir argument is expected if write=True')
            cv2.imwrite(os.path.join(args.dir, f'img_{i}.png'), image)
        else:
            cv2.imshow('frames', image)
            cv2.waitKey(t_frame_int)


if __name__ == '__main__':
    main()
