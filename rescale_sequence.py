import os
import cv2
from pathlib import Path
from tqdm import tqdm
from vot.region import io as reg_io
import main_utils as utils


def main():
    sequences = utils.get_all_sequences()
    scale_factor = 0.89
    for seq in sequences:
        h, w = utils.get_video_size(seq)
        if h >= 1080 and w >= 1920:
            fps = utils.get_video_fps(seq)
            images = sorted(Path('sequences', seq, 'color').iterdir())
            new_w, new_h = round(w * scale_factor), round(h * scale_factor)
            ground_truth = os.path.join('sequences', seq, 'groundtruth.txt')
            masks = reg_io.read_file(ground_truth)
            resized_masks = [msk.resize(scale_factor) for msk in masks]
            print('shape', h, w)
            print('name', seq)
            print('fps', fps)
            reg_io.write_file(ground_truth, resized_masks)
            for img_name in tqdm(images):
                img = cv2.imread(img_name.as_posix(), cv2.IMREAD_UNCHANGED)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(img_name.as_posix(), img_resized)
                print('converted', img_name)

            print()


if __name__ == '__main__':
    main()
