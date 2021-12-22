import argparse
from pathlib import Path
from interpolation.ABME import run as abme_run
from interpolation.RIFE import run as rife_run
from tqdm import tqdm
import main_utils as utils
import torch
import cv2
import os
import numpy as np
from time import sleep
from vot.region.shapes import Mask
from vot.region import io as vot_io

TARGET_SEQ = 'agility'
INTERPOLATION_METHOD = 'ABME'
SEQUENCE_DIR = utils.SEQUENCE_DIR


def generate_interpolation_groundtruth(target_seq, interpolation_method):
    # generate interpolation of masks
    interpolation_base_path = Path(SEQUENCE_DIR, target_seq)
    m0 = 'tmp_mask0.jpg'
    m2 = 'tmp_mask2.jpg'
    m1 = 'tmp_mask1.jpg'
    masks_int = []
    print(f'Generatig interpolation of groundtruth for sequence with {interpolation_method} method')
    masks = utils.load_masks_raster(target_seq)
    mask0 = cv2.cvtColor(next(masks), cv2.COLOR_GRAY2RGB) * 255
    cv2.imwrite('tmp_mask0.jpg', mask0)
    for mask2 in tqdm([cv2.cvtColor(e, cv2.COLOR_GRAY2RGB) * 255 for e in masks]):
        cv2.imwrite('tmp_mask2.jpg', mask2)
        if interpolation_method == 'ABME':
            args = argparse.Namespace(first=m0, second=m2,
                                      output=m1)
            abme_run.main(args)
        elif interpolation_method == 'RIFE':
            args = argparse.Namespace(img=[m0, m2], exp=1, ratio=0.0,
                                      rthreshold=0.02, rmaxcycles=8, modelDir='interpolation/RIFE/train_log_v6',
                                      output=m1)
            rife_run.main(namespace=args)
        tmp_int = cv2.imread(m1, cv2.IMREAD_GRAYSCALE)
        tmp_int[tmp_int > 0] = 255
        kernel = np.ones((4, 4), np.uint8)
        tmp_int = cv2.morphologyEx(tmp_int, cv2.MORPH_OPEN, kernel)
        masks_int.append(Mask(tmp_int / 255, optimize=True))
        # cv2.imwrite(m1, tmp_int)
        os.remove(m0)
        os.rename(m2, m0)
    os.remove(m0)
    os.remove(m1)
    vot_io.write_file(os.path.join(SEQUENCE_DIR, target_seq, f'groundtruth_{interpolation_method}.txt'), masks_int)


def generate_interpolation(target_seq, interpolation_method):
    # create 'interpolated' folder if it does not exist
    interpolation_base_path = Path(SEQUENCE_DIR, target_seq, 'interpolated', interpolation_method)
    interpolation_base_path.mkdir(parents=True, exist_ok=True)
    print(f'Generating Interpolation for sequence with {interpolation_method} method')
    frame_names = sorted(Path(SEQUENCE_DIR, target_seq, 'color').iterdir())
    frame1 = frame_names[0]

    sleep(0.5)
    for frame2 in tqdm(frame_names[1:]):
        if interpolation_method == 'ABME':
            args = argparse.Namespace(first=frame1.as_posix(), second=frame2.as_posix(),
                                      output=Path(interpolation_base_path, frame1.name).as_posix())
            abme_run.main(args)
        elif interpolation_method == 'RIFE':
            args = argparse.Namespace(img=[frame1.as_posix(), frame2.as_posix()], exp=1, ratio=0.0,
                                      rthreshold=0.02, rmaxcycles=8, modelDir='interpolation/RIFE/train_log_v6',
                                      output=Path(interpolation_base_path, frame1.name).as_posix())
            rife_run.main(namespace=args)
        # shift frames
        frame1 = frame2


def interpolate_all(interpolation_method, sequences=None, force=False):
    if type(interpolation_method) == str:
        interpolation_method = [interpolation_method]
    if sequences is None:
        sequences = utils.get_all_sequences()
    for seq in sequences:
        print()
        print('=====================================')
        utils.print_seq_info(seq)
        for method in interpolation_method:
            print(f'Starting interpolation with method {method}')
            if not utils.has_interpolation(seq, method) or force:
                with torch.no_grad():
                    try:
                        generate_interpolation(seq, method)
                    except RuntimeError:
                        print('sequence could not be interpolated due to runtime error (CUDA Memory).')
            else:
                print('Sequence already interpolated.')

            if not utils.has_interpolated_groundtruth(seq, method) or force:
                generate_interpolation_groundtruth(seq, method)
            else:
                print('Sequence already has interpolated ground truth')
            print('----------------')
        print('=====================================')
        print()


def main():
    # generate_interpolation_groundtruth('tiger', 'RIFE')
    interpolate_all(['RIFE'])


if __name__ == '__main__':
    main()
