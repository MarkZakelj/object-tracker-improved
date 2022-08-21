from pathlib import Path
import cv2
import os
from vot.region import io as vot_io
import numpy as np
WORKSPACE = 'TransT_M/votspace'
SEQUENCE_DIR = os.path.join(WORKSPACE, 'sequences')
RESULT_DIR = os.path.join(WORKSPACE, 'results', 'TransT_M', 'unsupervised')



def get_all_sequences():
    """return list of strings of all sequence names"""
    with Path(SEQUENCE_DIR, 'list.txt').open() as f:
        sequences = [e.strip() for e in f.readlines()]
    return sequences


def get_video_size(target_seq):
    img_file = os.path.join(SEQUENCE_DIR, target_seq, 'color', f'{"1".zfill(8)}.jpg')
    img = cv2.imread(img_file)
    size = (img.shape[0], img.shape[1])
    return size


def get_video_fps(target_seq):
    with open(os.path.join(SEQUENCE_DIR, target_seq, 'sequence')) as f:
        seq_info = [e.strip() for e in f.readlines()]
        fps = seq_info[2].replace('fps=', '')
        fps = int(fps)
        return fps


def get_n_frames(target_seq):
    n = len(list(Path(SEQUENCE_DIR, target_seq, 'color').iterdir()))
    return n


def has_interpolation(target_seq, interpolation_method):
    pth = Path(SEQUENCE_DIR, target_seq, 'interpolated', interpolation_method)
    pth_true = Path(SEQUENCE_DIR, target_seq, 'color')
    if pth.exists() and pth.is_dir():
        pth_iter = iter(sorted(pth.iterdir()))
        for img1 in sorted(pth_true.iterdir())[:-1]:
            try:
                img2 = next(pth_iter)
            except StopIteration:
                return False
            if img1.name != img2.name:
                return False
    else:
        return False
    return True


def has_interpolated_groundtruth(target_seq, interpolation_method):
    pth = os.path.join(SEQUENCE_DIR, target_seq, f'groundtruth_{interpolation_method}.txt')
    n_frames = get_n_frames(target_seq)
    try:
        with open(pth, 'r') as f:
            if len(f.readlines()) != n_frames - 1:
                return False
            return True
    except FileNotFoundError:
        return False



def load_masks_raster(target_seq, interpolated=False, interpolation_method='RIFE', load_results=False):
    ground_truth = os.path.join(SEQUENCE_DIR, target_seq, 'groundtruth.txt')
    masks = vot_io.read_trajectory(ground_truth)
    size = get_video_size(target_seq)
    if load_results:
        results = os.path.join(RESULT_DIR, target_seq, f"{target_seq}_001.txt")
        fp = open(results, 'r')
        fp.readline()
        res_masks = vot_io.read_trajectory(fp)
        masks = [masks[0]] + res_masks


    rasters = (msk.rasterize(bounds=(0, 0, size[1] - 1, size[0] - 1)) for msk in masks)
    rasters_int = None
    if interpolated:
        ground_truth_int = os.path.join(SEQUENCE_DIR, target_seq, f'groundtruth_{interpolation_method}.txt')
        masks_int = vot_io.read_trajectory(ground_truth_int)
        rasters_int = (msk.rasterize(bounds=(0, 0, size[1] - 1, size[0] - 1)) for msk in masks_int)
    for rst in rasters:
        yield rst
        if rasters_int is not None:
            yield next(rasters_int)


def load_video(target_seq, interpolated=False, interpolation_method='ABME'):
    video = cv2.VideoCapture(os.path.join(SEQUENCE_DIR, target_seq, 'color', "%08d.jpg"), cv2.CAP_IMAGES)
    video_inter = None
    if interpolated:
        video_inter = cv2.VideoCapture(
            os.path.join(SEQUENCE_DIR, target_seq, 'interpolated', interpolation_method, "%08d.jpg"), cv2.CAP_IMAGES)
    rolling = True
    while rolling:
        ok1, frame1 = video.read()
        if video_inter is not None:
            ok2, frame2 = video_inter.read()
        if ok1:
            yield ok1, frame1
        else:
            rolling = False
        if interpolated:
            if ok2:
                yield ok2, frame2
    yield False, None


def apply_mask_to_image(image, mask, color):
    masked_int_img = np.where(mask[..., None], color, image)
    out_int = cv2.addWeighted(image, 0.2, masked_int_img, 0.8, 0)
    return out_int


def print_seq_info(target_seq):
    h, w = get_video_size(target_seq)
    print(f'NAME: {target_seq}')
    print(f'SIZE: H={h} W={w}')
    print(f'N_FRAMES: {get_n_frames(target_seq)}')
    print(f'FPS: {get_video_fps(target_seq)}')
    print('-------------------------------------')
