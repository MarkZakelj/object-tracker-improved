import cv2
import os
import numpy as np
from vot.region import io as vot_io
import main_utils as utils

SEQUENCE_DIR = 'tracking/sequences'
TARGET_SEQ = 'tiger'
INTERPOLATION_METHOD = 'RIFE'
SHOW_TRUE_MASK = True
INTERPOLATE = True
PER_CLICK = True


def main():
    video = utils.load_video(TARGET_SEQ, interpolated=INTERPOLATE, interpolation_method=INTERPOLATION_METHOD)
    masks = utils.load_masks_raster(TARGET_SEQ, interpolated=INTERPOLATE, interpolation_method=INTERPOLATION_METHOD)
    color = np.array([0, 255, 0], dtype='uint8')
    fps = utils.get_video_fps(TARGET_SEQ)
    size = utils.get_video_size(TARGET_SEQ)
    t_frame_int = round(1000 / (fps * (INTERPOLATE + 1)))
    if PER_CLICK:
        t_frame_int = 0
    print(t_frame_int)
    print('size:', size)
    for i, (ok, image) in enumerate(video):
        if not ok:
            break
        if SHOW_TRUE_MASK:
            image = utils.apply_mask_to_image(image, next(masks), color)
        cv2.imshow('frames', image)
        cv2.waitKey(t_frame_int)






if __name__ == '__main__':
    main()
